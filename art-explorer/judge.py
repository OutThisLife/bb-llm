"""VLM-guided discovery: VLM generates candidates -> render -> judge -> keep."""

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from infer import render_api, load_vlm, vlm_generate, parse_params, fill_defaults
from infer import DEFAULT_ENDPOINT, OUTPUT_DIR, SYSTEM_PROMPT, DISCOVER_PROMPTS
from scoring import ollama_score, score_bar, OLLAMA, JUDGE_MODEL
from utils import to_prefixed

from PIL import Image

D, R, B = "\033[2m", "\033[0m", "\033[1m"
G, RE, Y, C, M = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[35m"


def load_refs(n=4):
    paths = sorted(Path("references").glob("figma-*.png"))
    if not paths:
        paths = sorted(Path("references").glob("*.png"))
    return [Image.open(p).convert("RGB") for p in random.sample(paths, min(n, len(paths)))]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("-n", "--keep", type=int, default=10)
    pa.add_argument("-w", "--workers", type=int, default=6)
    pa.add_argument("--threshold", type=float, default=80)
    pa.add_argument("--judge-model", default=JUDGE_MODEL)
    pa.add_argument("--adapter", default="models")
    pa.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    pa.add_argument("--ollama", default=OLLAMA)
    pa.add_argument("--auto-ref", action="store_true")
    args = pa.parse_args()

    refs = load_refs()
    if not refs:
        return print("No reference images found")

    print("Loading VLM for candidate generation...")
    vlm, processor, device = load_vlm(args.adapter)

    print(f"""
{B}┌──────────────────────────────────────────┐
│        {M}VLM JUDGE{R}{B}  ·  model-led discovery  │
├──────────────────────────────────────────┤
│  {C}generator{R}{B}  fine-tuned VLM{" " * 16} │
│  {C}judge{R}{B}      {args.judge_model:<27s} │
│  {C}target{R}{B}     {args.keep} keepers{" " * (20 - len(str(args.keep)))} │
│  {C}threshold{R}{B}  >={args.threshold:.0f}/100{" " * (20 - len(f"{args.threshold:.0f}"))} │
│  {C}workers{R}{B}    {args.workers} parallel{" " * (18 - len(str(args.workers)))} │
│  {C}auto-ref{R}{B}   {"ON" if args.auto_ref else "OFF":<27s} │
└──────────────────────────────────────────┘{R}
""")

    out_img, out_params = OUTPUT_DIR / "images", OUTPUT_DIR / "params"
    out_img.mkdir(parents=True, exist_ok=True)
    out_params.mkdir(parents=True, exist_ok=True)

    def generate_candidate():
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": random.choice(DISCOVER_PROMPTS)}]},
            ]
            return fill_defaults(parse_params(vlm_generate(vlm, processor, device, messages, temperature=0.9)))
        except Exception:
            return None

    lock = threading.Lock()
    kept, tried, file_idx = 0, 0, len(list(out_img.glob("*.png")))
    t0 = time.time()

    def on_result(params, img, score, raw):
        nonlocal kept, tried, file_idx
        with lock:
            tried += 1
            geo = params.get("geometry", "?")
            elapsed = time.time() - t0
            head = f"{D}#{tried:<4d}{R} {C}{kept}{R}/{args.keep}  {D}{tried / elapsed:.1f}/s{R}"

            if img is None:
                return print(f"{head}  {RE}RENDER FAIL{R} {D}{geo}{R}")

            subs = re.findall(r"[CBDSM]:\s*(\d+(?:\.\d+)?)", raw)
            sub_str = " ".join(f"{k}:{v}" for k, v in zip("CBDSM", subs)) if len(subs) >= 5 else ""

            color = G if score >= 70 else Y if score >= 50 else RE
            if score >= args.threshold:
                img.save(out_img / f"{file_idx:06d}.png")
                json.dump(params, open(out_params / f"{file_idx:06d}.json", "w"))
                if args.auto_ref:
                    open(Path("references/refs.jsonl"), "a").write(json.dumps(to_prefixed(params)) + "\n")
                print(f"{head}  {color}{score_bar(score)}{R} {B}{score:>5.1f}{R}/100  "
                      f"{geo:<10s} {D}{sub_str}{R}  {G}{B}>>> KEPT [{file_idx}]{R}")
                file_idx += 1
                kept += 1
            else:
                print(f"{head}  {color}{score_bar(score)}{R} {score:>5.1f}/100  {D}{geo:<10s} {sub_str}{R}")

    def judge_one(params):
        img = render_api(params, args.endpoint)
        if img is None:
            return params, None, 0.0, "render failed"
        score, raw, _, _ = ollama_score(img, refs, args.judge_model, args.ollama)
        return params, img, score, raw

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = set()

        def submit():
            params = generate_candidate()
            if params:
                futures.add(pool.submit(judge_one, params))

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

    del vlm, processor

    elapsed = time.time() - t0
    print(f"""
{B}┌──────────────────────────────────────────┐
│  {G}COMPLETE{R}{B}                                   │
│  {kept} kept / {tried} tried ({kept / tried * 100 if tried else 0:.1f}% hit rate){" " * max(0, 13 - len(str(kept)) - len(str(tried)))} │
│  {elapsed:.0f}s elapsed  ·  {tried / elapsed:.1f} candidates/s{" " * max(0, 11 - len(f"{elapsed:.0f}"))} │
└──────────────────────────────────────────┘{R}
""")


if __name__ == "__main__":
    main()
