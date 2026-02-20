"""
Data Generation
===============
Generate synthetic (image, params) pairs via /api/raster.
Random + ref-biased breeding for coverage.
"""

import argparse
import contextlib
import json
import os
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from utils import analyze_refs, random_params, to_scene_params

DATA_DIR = Path("data")
REFS_PATH = Path("references/refs.jsonl")
API_URL = "http://localhost:3000/api/raster"
MAX_WORKERS = min(os.cpu_count() or 4, 12)


def load_refs():
    """Load refs.jsonl for breeding."""
    if not REFS_PATH.exists():
        return []
    lines = REFS_PATH.read_text().strip().split("\n")
    return [json.loads(ln) for ln in lines if ln]


_pbar_lock = threading.Lock()
_pbar = None
_session_local = threading.local()


def _get_session():
    if not hasattr(_session_local, "session"):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5)
        session.mount("http://", HTTPAdapter(max_retries=retry))
        _session_local.session = session
    return _session_local.session


def _update_progress():
    with _pbar_lock:
        if _pbar is not None:
            with contextlib.suppress(Exception):
                _pbar.update(1)


def generate_one(i, prefixed, data_dir=DATA_DIR):
    flat = to_scene_params(prefixed)
    try:
        resp = _get_session().post(API_URL, json=prefixed, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"\nSkipping {i:06d}: {e}")
        return None, None

    img_path = data_dir / f"images/{i:06d}.png"
    img_path.write_bytes(resp.content)
    flat["url"] = f"http://localhost:3000/render?c={resp.headers.get('X-Encoded-Params', '')}"
    (data_dir / f"params/{i:06d}.json").write_text(json.dumps(flat))
    _update_progress()
    return img_path, flat


def _breed_from_refs(refs, n_layers, ref_dist=None):
    """Crossover + partial inheritance from refs."""
    if len(refs) >= 2 and random.random() < 0.7:
        parents = random.sample(refs, 2)
    else:
        parents = [random.choice(refs)]

    child = random_params(n_layers=n_layers, ref_dist=ref_dist)

    # Don't let breeding corrupt mirror layer structure
    _mirror_keys = {"position", "rotation", "scale"}
    for k in child:
        if k.startswith("Groups.") and k.split("-", 1)[-1] in _mirror_keys:
            continue
        if random.random() < 0.6:
            parent = random.choice(parents)
            if k in parent:
                val = _extract_value(parent[k])
                if isinstance(val, (int, float)):
                    val = val * random.uniform(0.8, 1.2)
                    if k.endswith("geoWidth"):
                        # Re-bias inherited widths so breeding doesn't drift thick.
                        val = max(0.001, min(0.1, float(val) * (random.random() ** 1.5)))
                    if isinstance(parent[k], int):
                        val = round(val)
                child[k] = val

    return child


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _extract_value(v):
    """Handle refs stored as Leva objects: {disabled, value}."""
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def _num(v, default):
    v = _extract_value(v)
    return float(v) if isinstance(v, (int, float)) else default


_FIGMA_GEOS = ["ring", "arch", "u", "infinity", "roundedRect"]


def _apply_hard_figma_constraints(prefixed):
    """Apply strict figma-like constraints to sampled prefixed params."""
    p = {k: _extract_value(v) for k, v in dict(prefixed).items()}

    reps = int(round(_clamp(_num(p.get("Scalars.repetitions"), 260.0), 120, 460)))
    p["Scalars.repetitions"] = reps
    max_step = _clamp(12.0 / reps, 0.01, 0.08)
    axis_max = _clamp(28.0 / reps, 0.06, 0.4)

    p["Scalars.stepFactor"] = round(
        _clamp(_num(p.get("Scalars.stepFactor"), max_step * 0.7), 0.004, max_step), 4
    )
    p["Scalars.alphaFactor"] = round(_clamp(_num(p.get("Scalars.alphaFactor"), 0.8), 0.45, 1.0), 4)
    p["Scalars.scaleFactor"] = round(_clamp(_num(p.get("Scalars.scaleFactor"), 1.0), 0.85, 1.3), 4)
    p["Scalars.rotationFactor"] = round(_clamp(_num(p.get("Scalars.rotationFactor"), 0.0), -0.35, 0.35), 4)
    p["Element.geoWidth"] = round(_clamp(_num(p.get("Element.geoWidth"), 0.008), 0.001, 0.02), 4)
    p["Element.startAngle"] = round(_num(p.get("Element.startAngle"), 0.0), 4)
    gr = p.get("Element.gradientRange")
    if not isinstance(gr, list) or len(gr) != 2:
        gr = [round(random.uniform(-0.2, 0.4), 4), round(random.uniform(0.6, 1.4), 4)]
    p["Element.gradientRange"] = [round(gr[0], 4), round(gr[1], 4)]
    p["Noise.enabled"] = False
    p["Dither.enabled"] = False
    p["Spatial.origin"] = "center"
    p["Scalars.positionCoupled"] = True
    p["Scalars.positionProgression"] = random.choice(["index", "scale"])
    x_default = random.choice([-1.0, 1.0]) * random.uniform(0.04, axis_max)
    p["Spatial.xStep"] = round(_clamp(_num(p.get("Spatial.xStep"), x_default), -axis_max, axis_max), 4)
    p["Spatial.yStep"] = round(_clamp(_num(p.get("Spatial.yStep"), 0.0), -0.15, 0.15), 4)

    if p.get("Scalars.scaleProgression") not in {"exponential", "sine", "golden"}:
        p["Scalars.scaleProgression"] = random.choice(["exponential", "sine", "golden"])
    if p.get("Scalars.rotationProgression") not in {"golden-angle", "sine", "fibonacci"}:
        p["Scalars.rotationProgression"] = random.choice(["golden-angle", "sine"])
    if p.get("Scalars.alphaProgression") not in {"exponential", "linear"}:
        p["Scalars.alphaProgression"] = random.choice(["exponential", "exponential", "linear"])
    if p.get("Element.geometry") not in _FIGMA_GEOS:
        p["Element.geometry"] = random.choice(["ring", "ring", "arch", "roundedRect", "infinity"])

    for k, v in list(p.items()):
        if not k.startswith("Groups."):
            continue
        suffix = k.split("-", 1)[-1]
        if suffix == "geoWidth" and isinstance(v, (int, float)):
            p[k] = round(_clamp(float(v), 0.001, 0.02), 4)
        elif suffix == "stepFactor" and isinstance(v, (int, float)):
            p[k] = round(_clamp(float(v), 0.004, max_step * 1.5), 4)
        elif suffix == "alphaFactor" and isinstance(v, (int, float)):
            p[k] = round(_clamp(float(v), 0.45, 1.0), 4)
        elif suffix == "scaleFactor" and isinstance(v, (int, float)):
            p[k] = round(_clamp(float(v), 0.85, 1.3), 4)
        elif suffix == "rotationFactor" and isinstance(v, (int, float)):
            p[k] = round(_clamp(float(v), -0.35, 0.35), 4)
        elif suffix == "geometry" and isinstance(v, str):
            if v not in _FIGMA_GEOS:
                p[k] = random.choice(["ring", "arch", "roundedRect"])

    return p


def make_params(refs, n_layers, breed_rate=0.3, ref_dist=None, hard_figma=False):
    """Generate params: stratified categoricals + optional ref breeding."""
    layers = n_layers if n_layers >= 0 else None

    use_refs = refs if not hard_figma else []
    use_ref_dist = ref_dist if not hard_figma else None
    use_breed_rate = breed_rate if not hard_figma else 0.0

    if use_refs and random.random() < use_breed_rate:
        params = _breed_from_refs(use_refs, layers, ref_dist=use_ref_dist)
    else:
        params = random_params(n_layers=layers, stratified=True, ref_dist=use_ref_dist)

    if hard_figma:
        params = _apply_hard_figma_constraints(params)

    return params


def generate(n=2000, n_layers=-1, workers=MAX_WORKERS, hard_figma=False):
    global _pbar

    (DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "params").mkdir(parents=True, exist_ok=True)

    existing = list((DATA_DIR / "images").glob("*.png"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Continuing from index {start_idx} ({start_idx} existing)")

    refs = load_refs()
    ref_dist = analyze_refs(refs) if (refs and not hard_figma) else None
    if refs and not hard_figma:
        n_dist = len(ref_dist) if ref_dist else 0
        print(f"Including {len(refs)} refs (30% breeding, {n_dist} auto-distributions)")
    elif hard_figma:
        print("Hard figma mode: using strict constraints, no ref breeding/distribution")

    print(f"Generating {n} samples ({workers} workers)...")

    all_params = [
        make_params(refs, n_layers, ref_dist=ref_dist, hard_figma=hard_figma)
        for _ in range(n)
    ]
    random.shuffle(all_params)

    with tqdm(total=n, desc="Rendering", unit="img") as pbar:
        _pbar = pbar
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(generate_one, start_idx + i, p)
                for i, p in enumerate(all_params)
            ]
            for f in futures:
                f.result()
        _pbar = None

    print(f"Done: {n} samples in {DATA_DIR}/")


def main():
    p = argparse.ArgumentParser(description="Generate training data")
    p.add_argument("-n", type=int, default=2000, help="Number of samples")
    p.add_argument("--layers", type=int, default=-1, help="Layers (-1=random)")
    p.add_argument("--clean", action="store_true", help="Clear data first")
    p.add_argument(
        "--hard-figma",
        action="store_true",
        help="Apply strict figma-like hard constraints (same data/ output paths)",
    )
    args = p.parse_args()

    if args.clean:
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
            print("Cleared data/")

    generate(n=args.n, n_layers=args.layers, hard_figma=args.hard_figma)


if __name__ == "__main__":
    main()
