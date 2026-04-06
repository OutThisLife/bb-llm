"""
Predict: VLM-powered scene parameter prediction.
  --target image.png   Image → params → render
  --text "..."         Text → params → render
  --discover           Creative generation
  --refine             + CMA-ES polish (image mode only)
"""

import argparse
import io
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils import (
    BOOLEAN_KEYS, CATEGORICAL_KEYS, CONTINUOUS_KEYS,
    LAYER_CONTINUOUS_KEYS, LAYER_OPTIONALS, MAX_LAYERS,
    decode_params, encode_params, get_device, to_prefixed,
)

IMG_SIZE = 256
DEFAULT_ENDPOINT = "http://localhost:3000/api/raster"
OUTPUT_DIR = Path("output")
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

SYSTEM_PROMPT = "You are a scene parameter predictor. Output valid JSON matching the renderer's param schema."

DISCOVER_PROMPTS = [
    "Generate scene parameters for an elegant, visually striking generative art piece. Be creative with geometry, progression types, and layer composition. Output valid JSON.",
    "Generate scene parameters for a minimal, geometric composition with interesting symmetry. Output valid JSON.",
    "Generate scene parameters for a dense, organic pattern with flowing movement. Output valid JSON.",
    "Generate novel scene parameters that would produce a beautiful, unique generative artwork. Experiment with unusual combinations. Output valid JSON.",
]

_session_local = threading.local()


def _get_session():
    if not hasattr(_session_local, "session"):
        s = requests.Session()
        s.mount("http://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))
        _session_local.session = s
    return _session_local.session


def render_api(params, endpoint=DEFAULT_ENDPOINT):
    try:
        api_params = to_prefixed(params) if "layers" in params else params
        resp = _get_session().post(endpoint, json=api_params, timeout=60)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def load_vlm(adapter_path="models"):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    proc_path = adapter_path if Path(adapter_path, "preprocessor_config.json").exists() else MODEL_ID
    return model, AutoProcessor.from_pretrained(proc_path), base.device


def vlm_generate(model, processor, device, messages, max_tokens=512, temperature=None):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    images = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "image":
                    images.append(Image.open(part["image"]).convert("RGB"))

    inputs = processor(text=[text], images=images if images else None, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen = {"do_sample": temperature is not None, "max_new_tokens": max_tokens}
    if temperature:
        gen["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    return processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_params(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{", response)
    if not match:
        raise ValueError(f"Could not parse: {response[:200]}")
    tail = response[match.start():]
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(tail)
    return obj


def fill_defaults(params: dict) -> dict:
    defaults = {
        "alphaFactor": 0.68, "alphaProgression": "exponential",
        "color": "#efeddb",
        "crtBleed": 0.4, "crtBloom": 0.15, "crtBrightness": 1.0, "crtEnabled": False,
        "crtMask": "grille", "crtMaskStrength": 0.5, "crtScale": 1.5, "crtScanlines": 0.3, "crtWarp": 0.0,
        "debug": False, "geoWidth": 0.041, "geometry": "ring",
        "gradientAngle": 0.0, "gradientRange": [0.2, 1.0],
        "noiseDensity": 0.11, "noiseEnabled": False, "noiseOpacity": 0.11, "noiseSize": 0.3,
        "origin": "top-center", "position": {"x": 0, "y": -0.5}, "positionCoupled": True,
        "positionProgression": "index", "repetitions": 75, "rotation": 0.0,
        "rotationFactor": -0.48, "rotationProgression": "linear",
        "scale": 0.4, "scaleFactor": 1.03, "scaleProgression": "exponential",
        "startAngle": 0.0, "stepFactor": 0.02, "xStep": -1.5, "yStep": 0.0,
        "layers": [{"position": {"x": 0, "y": -0.5}, "rotation": 0.0, "scale": {"x": -1.0, "y": 1.0}}],
    }

    for k, v in defaults.items():
        params.setdefault(k, v)
    for k, opts in CATEGORICAL_KEYS.items():
        if params.get(k) not in opts:
            params[k] = opts[0]
    params.setdefault("layers", [])
    return params


def to_lpips_tensor(img, size=IMG_SIZE):
    arr = np.array(img.resize((size, size))).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1


def refine_cmaes(params, target_path, endpoint=DEFAULT_ENDPOINT, max_fevals=1500):
    import lpips
    from cmaes import CatCMA

    device = get_device()
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    lpips_fn.eval()
    target_t = to_lpips_tensor(Image.open(target_path).convert("RGB")).unsqueeze(0).to(device)

    cats = {k: params.get(k, CATEGORICAL_KEYS[k][0]) for k in CATEGORICAL_KEYS}
    layers = params.get("layers", [])
    n_layers = min(len(layers), MAX_LAYERS)
    geos = [l.get("geometry", "ring") for l in layers[:n_layers]]
    presence = [[name in l for name in LAYER_OPTIONALS] for l in layers[:n_layers]]
    x0 = encode_params(params, n_layers)

    ndim = len(CONTINUOUS_KEYS) + len(BOOLEAN_KEYS) + len(LAYER_CONTINUOUS_KEYS) * MAX_LAYERS
    mean = np.full(ndim, 0.5)
    n = min(len(x0), ndim)
    mean[:n] = x0[:n]

    cat_keys = list(CATEGORICAL_KEYS.keys())
    geo_opts = CATEGORICAL_KEYS["geometry"]
    cat_sizes = [len(CATEGORICAL_KEYS[k]) for k in cat_keys] + [MAX_LAYERS + 1] + [len(geo_opts)] * MAX_LAYERS
    cat_num = np.array(cat_sizes, dtype=np.int32)

    n_cat = len(cat_num)
    cat_param = np.zeros((n_cat, int(cat_num.max())))
    lc_ci = len(cat_keys)

    def _init(ci, n_opts, idx):
        if n_opts == 1:
            cat_param[ci, 0] = 1.0
            return
        rest = 0.3 / (n_opts - 1)
        cat_param[ci, :n_opts] = rest
        cat_param[ci, idx] = 0.7

    for ci, k in enumerate(cat_keys):
        opts = CATEGORICAL_KEYS[k]
        _init(ci, len(opts), opts.index(cats[k]) if cats[k] in opts else 0)
    _init(lc_ci, MAX_LAYERS + 1, n_layers)
    for i in range(MAX_LAYERS):
        g = geos[i] if i < len(geos) else geo_opts[0]
        _init(lc_ci + 1 + i, len(geo_opts), geo_opts.index(g) if g in geo_opts else 0)

    opt = CatCMA(mean=mean, sigma=0.2, cat_num=cat_num, bounds=np.tile([0.0, 1.0], (ndim, 1)), cat_param=cat_param)
    best_score, best_params = float("inf"), params

    print(f"  CMA-ES refining ({max_fevals} fevals)...")
    fevals = 0
    while fevals < max_fevals and not opt.should_stop():
        solutions = [opt.ask() for _ in range(opt.population_size)]
        batch = []
        for x, c in solutions:
            ci = np.argmax(c, axis=1)
            cs = {k: CATEGORICAL_KEYS[k][min(int(ci[i]), len(CATEGORICAL_KEYS[k]) - 1)] for i, k in enumerate(cat_keys)}
            nl = min(int(ci[lc_ci]), MAX_LAYERS)
            lg = [geo_opts[min(int(ci[lc_ci + 1 + i]), len(geo_opts) - 1)] for i in range(nl)]
            lp = [presence[i] if i < len(presence) else [False] * len(LAYER_OPTIONALS) for i in range(nl)]
            candidate = decode_params(np.clip(x, 0, 1), cs, lg, lp, nl)
            candidate["color"] = params.get("color", candidate.get("color"))
            candidate["position"] = params.get("position", candidate.get("position"))
            for i, layer in enumerate(candidate.get("layers", [])):
                if i < len(layers) and "color" in layers[i]:
                    layer["color"] = layers[i]["color"]
            batch.append(candidate)

        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as pool:
            images = [f.result() for f in [pool.submit(render_api, p, endpoint) for p in batch]]

        scores = []
        for img in images:
            if img is None:
                scores.append(1.0)
                continue
            with torch.no_grad():
                scores.append(lpips_fn(to_lpips_tensor(img).unsqueeze(0).to(device), target_t).item())

        opt.tell([((x, c), s) for (x, c), s in zip(solutions, scores, strict=True)])
        for i, s in enumerate(scores):
            if s < best_score:
                best_score, best_params = s, batch[i]
        fevals += opt.population_size

    print(f"  Refined: LPIPS={best_score:.4f}")
    return best_params, best_score


def _save(params, rendered, name):
    for d in ("images", "params"):
        (OUTPUT_DIR / d).mkdir(parents=True, exist_ok=True)
    rendered.save(OUTPUT_DIR / f"images/{name}.png")
    json.dump(params, open(OUTPUT_DIR / f"params/{name}.json", "w"))
    print(f"  Saved: {OUTPUT_DIR / f'images/{name}.png'}")


def image_to_render(target_path, adapter, endpoint, refine):
    print(f"\nTarget: {target_path}")
    model, processor, device = load_vlm(adapter)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": target_path},
            {"type": "text", "text": "Predict the full scene parameters for this rendered image."},
        ]},
    ]

    params = fill_defaults(parse_params(vlm_generate(model, processor, device, messages)))
    del model, processor
    torch.cuda.empty_cache()

    if refine:
        params, score = refine_cmaes(params, target_path, endpoint)
        print(f"Final LPIPS: {score:.4f}")

    rendered = render_api(params, endpoint)
    if rendered is None:
        return print("Render failed.")

    _save(params, rendered, f"{Path(target_path).stem}_000")
    print(f"  geo={params.get('geometry')}, layers={len(params.get('layers', []))}")


def text_to_render(text, adapter, endpoint):
    print(f"\nPrompt: {text}")
    model, processor, device = load_vlm(adapter)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": f"Predict scene parameters for: {text}"}]},
    ]

    params = fill_defaults(parse_params(vlm_generate(model, processor, device, messages, temperature=0.7)))
    del model, processor
    torch.cuda.empty_cache()

    rendered = render_api(params, endpoint)
    if rendered is None:
        return print("Render failed.")

    _save(params, rendered, text[:40].replace(" ", "_").replace("/", "_"))


def main():
    p = argparse.ArgumentParser(description="VLM-powered scene parameter prediction")
    p.add_argument("--target", help="Image-to-render")
    p.add_argument("--text", help="Text-to-render")
    p.add_argument("--refine", action="store_true", help="CMA-ES polish (image mode)")
    p.add_argument("--adapter", default="models")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = p.parse_args()

    if args.target:
        image_to_render(args.target, args.adapter, args.endpoint, args.refine)
    elif args.text:
        text_to_render(args.text, args.adapter, args.endpoint)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
