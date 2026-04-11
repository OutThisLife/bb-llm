"""
SFT Eval: render-based evaluation of fine-tuned VLM.
=====================================================
  python eval_sft.py --data data                    # JSON-only metrics
  python eval_sft.py --data data --render           # + LPIPS via renderer
  python eval_sft.py --data data --render -k 4      # rejection sampling (pick best of k)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cli import cli
from infer import (
    DEFAULT_ENDPOINT,
    SYSTEM_PROMPT,
    fill_defaults,
    load_vlm,
    parse_params,
    render_api,
    to_lpips_tensor,
    vlm_generate,
)
from train import build_dataset
from utils import CATEGORICAL_KEYS, CONTINUOUS_RANGES


def check_quality(pred, target):
    """Key overlap + range validity between predicted and target params."""
    t_keys = set(target.keys()) - {"layers"}
    p_keys = set(pred.keys()) - {"layers"}
    overlap = len(t_keys & p_keys) / len(t_keys) if t_keys else 0

    valid = total = 0
    for k, v in pred.items():
        if k in CONTINUOUS_RANGES and isinstance(v, (int, float)):
            lo, hi = CONTINUOUS_RANGES[k]
            total += 1
            valid += lo <= v <= hi
        elif k in CATEGORICAL_KEYS and isinstance(v, str):
            total += 1
            valid += v in CATEGORICAL_KEYS[k]

    return overlap, valid / total if total else 0


def coverage_report(predictions):
    """Parameter mastery: categorical + continuous range bucket coverage."""
    n_buckets = 5
    cat_seen = {k: set() for k in CATEGORICAL_KEYS}
    cont_seen = {k: set() for k in CONTINUOUS_RANGES}

    for p in predictions:
        for k in CATEGORICAL_KEYS:
            if p.get(k) in CATEGORICAL_KEYS[k]:
                cat_seen[k].add(p[k])
        for k, (lo, hi) in CONTINUOUS_RANGES.items():
            v = p.get(k)
            if isinstance(v, (int, float)) and lo <= v <= hi:
                cont_seen[k].add(min(n_buckets - 1, int((v - lo) / (hi - lo + 1e-10) * n_buckets)))

    print(f"\n{'=' * 40}")
    print("Parameter Coverage")
    total_cat = 0
    for k, seen in cat_seen.items():
        cov = len(seen) / len(CATEGORICAL_KEYS[k])
        total_cat += cov
        if cov < 1.0:
            missing = sorted(set(CATEGORICAL_KEYS[k]) - seen)[:3]
            print(f"  {k}: {cov:.0%} (missing: {', '.join(missing)})")
    print(f"  Categorical avg: {total_cat / len(cat_seen):.0%}")

    total_cont = sum(len(s) / n_buckets for s in cont_seen.values())
    print(f"  Continuous avg:  {total_cont / len(cont_seen):.0%} ({n_buckets} buckets)")
    print(f"{'=' * 40}")


def evaluate(
    data_dir: Path,
    adapter: str = "models",
    endpoint: str = DEFAULT_ENDPOINT,
    do_render: bool = False,
    k: int = 1,
    max_samples: int = 50,
):
    _, val = build_dataset(data_dir)
    val = [s for s in val if s["image_path"]][:max_samples]
    print(f"Eval: {len(val)} samples" + (f", best-of-{k}" if k > 1 else ""))

    model, processor, device = load_vlm(adapter)

    lpips_fn = None
    if do_render:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        lpips_fn.eval()

    parsed, overlaps, valids, scores = 0, [], [], []
    all_predictions = []
    render_fails = 0
    logged_sample = False

    for sample in tqdm(val, desc="Eval"):
        target = json.loads(sample["target"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": "Predict the full scene parameters for this rendered image."},
                ],
            },
        ]

        best_lpips, best_ol, best_rv = float("inf"), 0.0, 0.0
        any_parsed = False
        best_pred = None

        for _ in range(k):
            temp = 0.7 if k > 1 else None
            try:
                resp = vlm_generate(model, processor, device, messages, temperature=temp)
                params = fill_defaults(parse_params(resp))
                any_parsed = True
            except Exception:
                continue

            ol, rv = check_quality(params, target)
            if ol > best_ol:
                best_ol, best_rv, best_pred = ol, rv, params

            if not logged_sample:
                pred_keys = sorted(set(params.keys()) - {"layers"})
                tgt_keys = sorted(set(target.keys()) - {"layers"})
                print(f"\n  Pred keys: {pred_keys[:8]}...")
                print(f"  Tgt keys:  {tgt_keys[:8]}...")
                logged_sample = True

            if do_render and lpips_fn is not None:
                rendered = render_api(params, endpoint)
                if rendered is None:
                    render_fails += 1
                    if render_fails == 1:
                        print(f"\n  First render fail -- is localhost:3000 running?")
                    continue
                target_img = Image.open(sample["image_path"]).convert("RGB")
                with torch.no_grad():
                    score = lpips_fn(
                        to_lpips_tensor(rendered).unsqueeze(0).to(device),
                        to_lpips_tensor(target_img).unsqueeze(0).to(device),
                    ).item()
                if score < best_lpips:
                    best_lpips = score

        if any_parsed:
            parsed += 1
            overlaps.append(best_ol)
            valids.append(best_rv)
            if best_pred:
                all_predictions.append(best_pred)
        if best_lpips < float("inf"):
            scores.append(best_lpips)

    del model, processor
    torch.cuda.empty_cache()

    n = len(val)
    print(f"\n{'=' * 40}")
    print(f"Parse rate:   {parsed}/{n} ({parsed / n * 100:.1f}%)")
    if overlaps:
        print(f"Key overlap:  {np.mean(overlaps):.1%}")
        print(f"Range valid:  {np.mean(valids):.1%}")
    if do_render:
        ok = n * k - render_fails
        print(f"Renders:      {ok}/{n * k} ({render_fails} failed)")
    if scores:
        arr = np.array(scores)
        print(f"LPIPS mean:   {arr.mean():.4f} +/- {arr.std():.4f}")
        print(f"LPIPS median: {np.median(arr):.4f}")
        print(f"  <0.2:       {(arr < 0.2).sum()}/{len(arr)}")
        print(f"  <0.1:       {(arr < 0.1).sum()}/{len(arr)}")
    print(f"{'=' * 40}")

    if all_predictions:
        coverage_report(all_predictions)


@cli
def main():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned VLM on held-out data")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--adapter", default="models")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--render", action="store_true", help="Render predictions, compute LPIPS")
    p.add_argument("-k", type=int, default=1, help="Rejection sampling: generate k, pick best")
    p.add_argument("-n", type=int, default=50, help="Max eval samples")
    args = p.parse_args()
    evaluate(args.data, args.adapter, args.endpoint, args.render, args.k, args.n)


if __name__ == "__main__":
    main()
