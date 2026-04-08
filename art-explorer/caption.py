"""
Caption Generation: structured (from params) + natural/brief (from VLM).
========================================================================
Output: {data}/captions.jsonl
  {"id": "...", "structured": "ring, golden-angle, ...", "prompt": "concentric rings fanning out", "brief": "spiral rings"}
"""

import argparse
import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from cli import cli

OLLAMA = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl"

PROMPT_INSTRUCTION = """\
You're a user typing a short request into a generative art tool. Look at this image and write what someone would type to get this result. Be direct and imperative — like a search query or a command.

Rules:
- 3-12 words max
- No "I want" or "make me" — just describe the thing
- Focus on what makes THIS image unique: shape, motion, color, density, symmetry, texture
- Describe the visual, not the technique
- Be specific — every image should get a different caption

Write TWO options on separate lines — one describing the visual precisely, one capturing the mood/feel.
No labels, no preamble, just the two lines."""


def _img_b64(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _clean(text):
    text = text.strip()
    for prefix in ["Here's", "Here is", "This image", "A concise", "Option"]:
        if text.startswith(prefix):
            for delim in ["\n\n", "\n", ':"', ': "', ":\n"]:
                if delim in text:
                    text = text.split(delim, 1)[-1]
                    break
    return text.strip().strip('"').strip("*").strip('"').strip()


# ── Structured caption from params (deterministic, no VLM) ──────────

def structured_caption(params: dict) -> str:
    """Build a structured caption from param values — no hallucination."""
    parts = []

    geo = params.get("geometry", "")
    if geo:
        parts.append(geo)

    reps = params.get("repetitions", 0)
    if reps > 200:
        parts.append("high density")
    elif reps > 50:
        parts.append("medium density")
    elif reps > 0:
        parts.append("low density")

    rot = params.get("rotationProgression", "")
    if rot:
        parts.append(f"{rot} rotation")

    sc = params.get("scaleProgression", "")
    if sc:
        parts.append(f"{sc} scale")

    origin = params.get("origin", "")
    if origin and origin != "center":
        parts.append(f"from {origin}")

    alpha = params.get("alphaProgression", "")
    if alpha:
        parts.append(f"{alpha} alpha")

    # Factor magnitudes — these dramatically affect the visual result
    sf = params.get("scaleFactor", 1.0)
    if sf < 0.3:
        parts.append("rapid shrink")
    elif sf > 1.5:
        parts.append("growing")

    rf = params.get("rotationFactor", 0.0)
    if abs(rf) > 0.5:
        parts.append("strong twist" if rf > 0 else "reverse twist")

    af = params.get("alphaFactor", 1.0)
    if af < 0.3:
        parts.append("fast fade")

    stf = params.get("stepFactor", 1.0)
    if stf > 1.5:
        parts.append("accelerating steps")
    elif stf < 0.1:
        parts.append("tight steps")

    # Position offsets
    xs, ys = abs(params.get("xStep", 0)), abs(params.get("yStep", 0))
    if xs > 0.5 or ys > 0.5:
        parts.append("spread" if xs > 0.5 and ys > 0.5 else "horizontal spread" if xs > 0.5 else "vertical spread")

    pp = params.get("positionProgression", "")
    if pp and pp != "index":
        parts.append(f"{pp} positioning")

    w = params.get("geoWidth", 0.041)
    if w < 0.01:
        parts.append("thin strokes")
    elif w > 0.06:
        parts.append("thick strokes")

    layers = params.get("layers", [])
    if layers:
        n = len(layers)
        scales = [l.get("scale", {}) for l in layers if isinstance(l.get("scale"), dict)]
        mirrored = any(s.get("x", 1) < 0 or s.get("y", 1) < 0 for s in scales)
        parts.append(f"{n} layer{'s' if n > 1 else ''}{' mirrored' if mirrored else ''}")

    if params.get("noiseEnabled"):
        parts.append("noise")
    if params.get("crtEnabled"):
        parts.append(f"CRT {params.get('crtMask', '')}")

    return ", ".join(parts)


# ── VLM calls ────────────────────────────────────────────────────────

def _call_ollama(img_path, prompt, model=DEFAULT_MODEL, endpoint=OLLAMA):
    resp = requests.post(
        f"{endpoint}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [_img_b64(img_path)]}],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 30},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_openrouter(img_path, prompt, model="google/gemini-2.0-flash-001"):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img_b64(img_path)}"}},
                ],
            }],
            "max_tokens": 30,
            "temperature": 0.7,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _fix_caption(text, max_words=12):
    """Clean up VLM caption artifacts and enforce word limit."""
    import re
    # Fix jammed words like "contrastAbstract" -> "contrast abstract"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Strip stray prefix fragments: a single lowercase word before a capitalized restart
    # catches "aesthetic White..." -> "White...", "Sw Circular..." -> "Circular...", "sym\"Golden..." -> "Golden..."
    text = re.sub(r'^[a-z]+["\s]+(?=[A-Z])', '', text)
    # Strip leading punctuation/quotes/commas
    text = text.lstrip('.,;:"\'\`')
    # Strip imperative openers directed at the tool
    text = re.sub(r'^(Create|Design|Make|Generate|Draw|Produce|Build)\s+(an?\s+)?', '', text, flags=re.IGNORECASE)
    # Enforce word limit
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    # Strip trailing fragment words (articles, prepositions, conjunctions)
    trail = {"a", "an", "the", "and", "or", "but", "in", "on", "of", "with", "for", "to", "from", "by", "as", "at", "its", "their", "this", "that"}
    words = text.split()
    while words and words[-1].lower().rstrip(".,;:") in trail:
        words.pop()
    text = " ".join(words).rstrip(".,;:")
    return text


def _parse_two_lines(raw: str):
    lines = [_clean(l) for l in raw.strip().splitlines() if l.strip() and not l.strip().startswith(("#", "-", "*"))]
    # strip leading numbering like "1." or "1)"
    cleaned = []
    for l in lines:
        for prefix in ["1.", "2.", "1)", "2)"]:
            if l.startswith(prefix):
                l = l[len(prefix):].strip()
                break
        cleaned.append(l)
    lines = [l for l in cleaned if l]
    prompt = _fix_caption(lines[0]) if lines else ""
    brief = _fix_caption(lines[1]) if len(lines) > 1 else prompt
    return prompt, brief


# ── Main ─────────────────────────────────────────────────────────────

def _load_existing(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def caption(data_dir="data", n=None, workers=4, backend="ollama", model=None):
    data = Path(data_dir)
    out_path = data / "captions.jsonl"
    img_dir, param_dir = data / "images", data / "params"

    images = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in img_dir.glob(ext))
    existing = _load_existing(out_path)
    todo = [p for p in images if p.stem not in existing]

    if n is not None:
        todo = todo[:n]

    if not todo:
        print(f"All {len(images)} images already captioned")
        return

    fn = _call_ollama if backend == "ollama" else _call_openrouter
    model_name = model or (DEFAULT_MODEL if backend == "ollama" else "gemini-2.0-flash")
    fn_kwargs = {"model": model_name} if model else {}

    print(f"Captioning {len(todo)} images ({backend}, {workers} workers)...")
    errors = 0

    def process(img_path):
        try:
            # Structured from params (deterministic)
            param_path = param_dir / f"{img_path.stem}.json"
            struct = ""
            if param_path.exists():
                params = json.loads(param_path.read_text())
                from utils import to_scene_params
                flat = to_scene_params(params) if any(k.startswith(("Scalars.", "Element.", "Spatial.")) for k in params) else params
                struct = structured_caption(flat)

            # VLM: prompt + brief
            raw = fn(img_path, PROMPT_INSTRUCTION, **fn_kwargs)
            prompt, brief = _parse_two_lines(raw)

            # Fall back to structured if VLM returned empty
            if not prompt:
                prompt = struct
            if not brief:
                brief = struct

            return {"id": img_path.stem, "structured": struct, "prompt": prompt, "brief": brief}
        except Exception as e:
            return str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, p): p for p in todo}
        with open(out_path, "a") as f:
            for fut in tqdm(as_completed(futures), total=len(todo), desc="Captioning"):
                result = fut.result()
                if isinstance(result, dict):
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                else:
                    errors += 1
                    if errors <= 3:
                        print(f"  Error: {result}")

    total = len(_load_existing(out_path))
    print(f"Done: {total}/{len(images)} captioned ({errors} errors)")


@cli
def main():
    p = argparse.ArgumentParser(description="Generate captions for rendered images")
    p.add_argument("--data", default="data")
    p.add_argument("-n", type=int, default=None, help="Max images to caption")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--backend", choices=["ollama", "openrouter"], default="ollama")
    p.add_argument("--model", default=None, help="Override model name")
    args = p.parse_args()
    caption(data_dir=args.data, n=args.n, workers=args.workers, backend=args.backend, model=args.model)


if __name__ == "__main__":
    main()
