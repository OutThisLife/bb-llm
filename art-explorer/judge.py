"""VLM-guided discovery: sample -> render -> judge -> keep."""

import argparse
import base64
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from PIL import Image, ImageDraw

from model import ParamVAE
from predict import render_api, DEFAULT_ENDPOINT, OUTPUT_DIR
from utils import (
    TASTE_FEATURE_DIM,
    decode_taste_features,
    get_device,
    load_state_dict_compat,
    to_prefixed,
)

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5vl"

D, R, B = "\033[2m", "\033[0m", "\033[1m"
G, RE, Y, C, M = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[35m"

PROMPT_TPL = (
    "This image shows {} reference artworks on top and 1 candidate "
    "on the bottom. Rate the candidate 1-10 for aesthetic quality and stylistic "
    "coherence with the references. Reply with ONLY a single number."
)


def load_refs(n=4):
    paths = sorted(Path("references").glob("figma-*.png"))
    if not paths:
        paths = sorted(Path("references").glob("*.png"))
    return [
        Image.open(p).convert("RGB")
        for p in random.sample(paths, min(n, len(paths)))
    ]


def make_grid(refs, candidate, cell=256):
    w, h = max(len(refs), 1) * cell, cell * 2 + 30
    grid = Image.new("RGB", (w, h), (30, 30, 30))
    for i, ref in enumerate(refs):
        grid.paste(ref.resize((cell, cell)), (i * cell, 0))
    ImageDraw.Draw(grid).text((4, cell + 2), "REFERENCES ^    CANDIDATE v", fill=(180, 180, 180))
    grid.paste(candidate.resize((cell, cell)), ((w - cell) // 2, cell + 30))
    return grid


def img_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def score_bar(score):
    bar = "█" * min(score, 10) + "░" * (10 - min(score, 10))
    color = G if score >= 7 else Y if score >= 5 else RE
    return f"{color}{bar}{R}"


def vlm_score(candidate, refs, model=MODEL, endpoint=OLLAMA):
    """Returns (score, raw_text, tokens_in, tokens_out)."""
    try:
        resp = requests.post(
            f"{endpoint}/api/chat",
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": PROMPT_TPL.format(len(refs)),
                    "images": [img_b64(make_grid(refs, candidate))],
                }],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 10},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["message"]["content"].strip()
        tok_in, tok_out = data.get("prompt_eval_count", 0), data.get("eval_count", 0)

        for token in text.split():
            try:
                return int(float(token.replace("/10", ""))), text, tok_in, tok_out
            except ValueError:
                continue
        return 0, text, tok_in, tok_out

    except Exception as e:
        return 0, str(e), 0, 0


def judge_one(params, refs, render_ep, vlm_model, vlm_ep):
    img = render_api(params, render_ep)
    if img is None:
        return params, None, 0, "render failed", 0, 0
    score, raw, ti, to = vlm_score(img, refs, vlm_model, vlm_ep)
    return params, img, score, raw, ti, to


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("-n", "--keep", type=int, default=10)
    pa.add_argument("-w", "--workers", type=int, default=10)
    pa.add_argument("--threshold", type=int, default=7)
    pa.add_argument("--prior", default="models/param_prior.pt")
    pa.add_argument("--model", default=MODEL)
    pa.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    pa.add_argument("--ollama", default=OLLAMA)
    pa.add_argument("--temp", type=float, default=1.0)
    pa.add_argument("--auto-ref", action="store_true")
    args = pa.parse_args()

    if not Path(args.prior).exists():
        return print(f"No prior at {args.prior}. Run: make train")

    device = get_device()
    ckpt = torch.load(args.prior, map_location=device, weights_only=False)
    prior = ParamVAE(in_dim=TASTE_FEATURE_DIM, latent_dim=ckpt.get("latent_dim", 64)).to(device)
    prior.load_state_dict(load_state_dict_compat(ckpt["model_state_dict"]))
    prior.eval()

    refs = load_refs()
    if not refs:
        return print("No reference images found")

    print(f"""
{B}┌──────────────────────────────────────────┐
│            {M}VLM JUDGE{R}{B}  ·  discovery loop   │
├──────────────────────────────────────────┤
│  {C}model{R}{B}      {args.model:<27s} │
│  {C}target{R}{B}     {args.keep} keepers{" " * (20 - len(str(args.keep)))} │
│  {C}threshold{R}{B}  >={args.threshold}/10{" " * (22 - len(str(args.threshold)))} │
│  {C}workers{R}{B}    {args.workers} parallel{" " * (18 - len(str(args.workers)))} │
│  {C}temp{R}{B}       {str(args.temp):<27s} │
│  {C}auto-ref{R}{B}   {"ON" if args.auto_ref else "OFF":<27s} │
└──────────────────────────────────────────┘{R}
""")

    out_img, out_params = OUTPUT_DIR / "images", OUTPUT_DIR / "params"
    out_img.mkdir(parents=True, exist_ok=True)
    out_params.mkdir(parents=True, exist_ok=True)
    refs_path = Path("references/refs.jsonl")

    lock = threading.Lock()
    kept, tried, file_idx = 0, 0, len(list(out_img.glob("*.png")))
    t0 = time.time()

    def on_result(params, img, score, raw, tok_in, tok_out):
        nonlocal kept, tried, file_idx
        with lock:
            tried += 1
            geo = params.get("geometry", "?")
            elapsed = time.time() - t0
            head = (
                f"{D}#{tried:<4d}{R} "
                f"{C}{kept}{R}/{args.keep}  "
                f"{D}{len(futures)} active  {tried / elapsed:.1f}/s{R}"
            )

            if img is None:
                return print(f"{head}  {RE}RENDER FAIL{R} {D}{geo}{R}")

            if score >= args.threshold:
                img.save(out_img / f"{file_idx:06d}.png")
                json.dump(params, open(out_params / f"{file_idx:06d}.json", "w"))
                if args.auto_ref:
                    open(refs_path, "a").write(json.dumps(to_prefixed(params)) + "\n")
                print(
                    f"{head}  {score_bar(score)} {B}{score:>2d}{R}/10  "
                    f"{geo:<12s} {D}{tok_in}>{tok_out}t{R}  "
                    f"{G}{B}>>> KEPT [{file_idx}]{R}"
                )
                file_idx += 1
                kept += 1
            else:
                print(
                    f"{head}  {score_bar(score)} {score:>2d}/10  "
                    f"{D}{geo:<12s} {tok_in}>{tok_out}t{R}"
                )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = set()

        def submit():
            with torch.no_grad():
                feat = prior.sample(1, device, temperature=args.temp).cpu().tolist()[0]
            futures.add(pool.submit(
                judge_one, decode_taste_features(feat),
                refs, args.endpoint, args.model, args.ollama,
            ))

        for _ in range(args.workers):
            submit()

        while kept < args.keep:
            done = next(as_completed(futures))
            futures.discard(done)
            on_result(*done.result())
            if kept < args.keep:
                submit()

        for f in as_completed(futures):
            on_result(*f.result())

    elapsed = time.time() - t0
    pct = kept / tried * 100 if tried else 0

    print(f"""
{B}┌──────────────────────────────────────────┐
│  {G}COMPLETE{R}{B}                                   │
│  {kept} kept / {tried} tried ({pct:.1f}% hit rate){" " * max(0, 13 - len(str(kept)) - len(str(tried)))} │
│  {elapsed:.0f}s elapsed  ·  {tried / elapsed:.1f} candidates/s{" " * max(0, 11 - len(f"{elapsed:.0f}"))} │
└──────────────────────────────────────────┘{R}
""")


if __name__ == "__main__":
    main()
