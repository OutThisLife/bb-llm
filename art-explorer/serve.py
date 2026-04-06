"""Dashboard API — wraps existing inference, data, RL metrics, schema."""

import base64
import csv
import io
import json
import os
import threading
from pathlib import Path

import requests as http
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

DATA = Path("data")
MODELS = Path("models")
OUTPUT = Path("output")
RL = Path("rl_runs")
REFS = Path("references")
RENDERER = os.getenv("RENDERER", "http://localhost:3000")

_vlm = {"model": None, "processor": None, "device": None, "loading": False}
_vlm_lock = threading.Lock()

app = FastAPI(title="Art Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _count(d: Path, ext: str) -> int:
    return len(list(d.glob(f"*.{ext}"))) if d.exists() else 0


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _ensure_vlm():
    with _vlm_lock:
        if _vlm["model"] is not None:
            return _vlm["model"], _vlm["processor"], _vlm["device"]
        _vlm["loading"] = True

    import logging, time
    log = logging.getLogger("uvicorn")
    log.info("Loading VLM (first request — this takes 1-3 min)…")
    t0 = time.time()
    from infer import load_vlm
    m, p, d = load_vlm()
    log.info("VLM loaded in %.1fs on %s", time.time() - t0, d)

    with _vlm_lock:
        _vlm.update(model=m, processor=p, device=d, loading=False)
    return m, p, d


def _render_b64(params):
    from utils import to_prefixed

    body = to_prefixed(params) if "layers" in params else params
    try:
        r = http.post(f"{RENDERER}/api/raster", json=body, timeout=30)
        r.raise_for_status()
        return base64.b64encode(r.content).decode()
    except Exception:
        return None


# ── Stats ────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    training_log = None
    train_progress = None
    state_path = MODELS / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        training_log = [
            {k: e.get(k) for k in ("step", "loss", "eval_loss", "learning_rate", "epoch", "grad_norm")}
            for e in state.get("log_history", [])
            if e.get("loss") is not None or e.get("eval_loss") is not None
        ]
        train_progress = {
            "global_step": state.get("global_step", 0),
            "max_steps": state.get("max_steps", 0),
            "epoch": state.get("epoch", 0),
            "num_train_epochs": state.get("num_train_epochs", 0),
        }

    return {
        "images": _count(DATA / "images", "png"),
        "captions": _count(DATA / "captions", "txt"),
        "params": _count(DATA / "params", "json"),
        "outputs": _count(OUTPUT / "images", "png"),
        "rl_runs": (
            [d.name for d in sorted(RL.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)]
            if RL.exists() else []
        ),
        "refs": sum(1 for _ in open(REFS / "refs.jsonl")) if (REFS / "refs.jsonl").exists() else 0,
        "adapter_exists": (MODELS / "adapter_config.json").exists(),
        "training_log": training_log,
        "train_progress": train_progress,
        "model_loaded": _vlm["model"] is not None,
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "method": "QLoRA 4-bit NF4 · rank 16 · alpha 32",
    }


@app.get("/api/schema")
def schema():
    from utils import BOOLEAN_KEYS, CATEGORICAL_KEYS, CONTINUOUS_RANGES, LAYER_SCHEMA, SCHEMA

    def ser(spec):
        out = dict(spec)
        for k in ("range", "options", "axes"):
            if k in out and not isinstance(out[k], list):
                out[k] = dict(out[k]) if isinstance(out[k], dict) else list(out[k])
        return out

    return {
        "base": {k: ser(v) for k, v in SCHEMA.items()},
        "layer": {k: ser(v) for k, v in LAYER_SCHEMA.items()},
        "categorical": CATEGORICAL_KEYS,
        "continuous_ranges": {k: list(v) for k, v in CONTINUOUS_RANGES.items()},
        "boolean": BOOLEAN_KEYS,
    }


# ── Data ─────────────────────────────────────────────

@app.get("/api/data")
def data_list(page: int = 1, size: int = 60):
    img_dir = DATA / "images"
    if not img_dir.exists():
        return {"items": [], "total": 0, "page": page, "pages": 0}

    imgs = sorted(img_dir.glob("*.png"), key=lambda p: p.stem)
    total = len(imgs)
    start = (page - 1) * size

    items = []
    for p in imgs[start : start + size]:
        s = p.stem
        cap = DATA / "captions" / f"{s}.txt"
        items.append({
            "id": s,
            "has_caption": cap.exists(),
            "caption": cap.read_text().strip()[:200] if cap.exists() else None,
        })

    return {"items": items, "total": total, "page": page, "pages": -(-total // size)}


@app.get("/api/data/{stem}.png")
def data_image(stem: str):
    p = DATA / "images" / f"{stem}.png"
    return FileResponse(p, media_type="image/png") if p.exists() else Response(status_code=404)


@app.get("/api/data/{stem}.json")
def data_params(stem: str):
    p = DATA / "params" / f"{stem}.json"
    return JSONResponse(json.loads(p.read_text())) if p.exists() else Response(status_code=404)


# ── Outputs ──────────────────────────────────────────

@app.get("/api/outputs")
def output_list(page: int = 1, size: int = 60):
    img_dir = OUTPUT / "images"
    if not img_dir.exists():
        return {"items": [], "total": 0, "page": page, "pages": 0}

    imgs = sorted(img_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(imgs)
    start = (page - 1) * size
    return {
        "items": [{"id": p.stem} for p in imgs[start : start + size]],
        "total": total, "page": page, "pages": -(-total // size),
    }


@app.get("/api/outputs/{name}.png")
def output_image(name: str):
    p = OUTPUT / "images" / f"{name}.png"
    return FileResponse(p, media_type="image/png") if p.exists() else Response(status_code=404)


# ── References ───────────────────────────────────────

@app.get("/api/refs")
def refs():
    p = REFS / "refs.jsonl"
    if not p.exists():
        return {"refs": [], "total": 0}
    items = []
    for line in open(p):
        try:
            items.append(json.loads(line.strip()))
        except Exception:
            pass
    return {"refs": items, "total": len(items)}


# ── RL ───────────────────────────────────────────────

@app.get("/api/rl")
def rl_list():
    if not RL.exists():
        return {"runs": []}
    runs = []
    for d in sorted(RL.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        cfg = d / "config.json"
        runs.append({
            "name": d.name,
            "config": json.loads(cfg.read_text()) if cfg.exists() else {},
            "images": _count(d / "images", "png"),
        })
    return {"runs": runs}


@app.get("/api/rl/{name}/rewards")
def rl_rewards(name: str):
    p = RL / name / "rewards.csv"
    if not p.exists():
        return Response(status_code=404)
    with open(p) as f:
        return {"data": [{k: _try_float(v) for k, v in r.items()} for r in csv.DictReader(f)]}


@app.get("/api/rl/{name}/{step}.png")
def rl_image(name: str, step: str):
    p = RL / name / "images" / f"{step}.png"
    return FileResponse(p, media_type="image/png") if p.exists() else Response(status_code=404)


# ── Render Proxy ─────────────────────────────────────

@app.post("/api/render")
def proxy_render(params: dict):
    from utils import to_prefixed

    body = to_prefixed(params) if "layers" in params else params
    try:
        r = http.post(f"{RENDERER}/api/raster", json=body, timeout=30)
        r.raise_for_status()
        return Response(content=r.content, media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


# ── Inference ────────────────────────────────────────

@app.post("/api/infer/text")
def infer_text(body: dict):
    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "No prompt"}, status_code=400)

    model, processor, device = _ensure_vlm()
    from infer import SYSTEM_PROMPT, fill_defaults, parse_params, vlm_generate

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Predict scene parameters for: {prompt}"},
        ]},
    ]

    import logging, time
    log = logging.getLogger("uvicorn")
    log.info("Generating (prompt=%s…)", prompt[:60])
    t0 = time.time()
    raw = vlm_generate(model, processor, device, messages, temperature=0.7)
    log.info("VLM responded in %.1fs", time.time() - t0)
    params = fill_defaults(parse_params(raw))
    img = _render_b64(params)
    log.info("Render %s", "ok" if img else "skipped")
    return {"params": params, "image": img, "raw": raw}


@app.post("/api/infer/image")
def infer_image(file: UploadFile = File(...)):
    import tempfile

    from PIL import Image

    img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, "PNG")
        tmp = f.name

    try:
        model, processor, device = _ensure_vlm()
        from infer import SYSTEM_PROMPT, fill_defaults, parse_params, vlm_generate

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": tmp},
                {"type": "text", "text": "Predict the full scene parameters for this rendered image."},
            ]},
        ]

        raw = vlm_generate(model, processor, device, messages)
        params = fill_defaults(parse_params(raw))
        return {"params": params, "image": _render_b64(params), "raw": raw}
    finally:
        os.unlink(tmp)


# ── Model lifecycle ──────────────────────────────────

@app.get("/api/model/status")
def model_status():
    return {
        "loaded": _vlm["model"] is not None,
        "loading": _vlm["loading"],
        "adapter_exists": (MODELS / "adapter_config.json").exists(),
    }


@app.post("/api/model/unload")
def model_unload():
    with _vlm_lock:
        if _vlm["model"]:
            import torch

            del _vlm["model"], _vlm["processor"]
            _vlm.update(model=None, processor=None, device=None)
            torch.cuda.empty_cache()
    return {"loaded": False}


# ── Serve built frontend ────────────────────────────

_dist = Path(__file__).parent / "dashboard" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True))
