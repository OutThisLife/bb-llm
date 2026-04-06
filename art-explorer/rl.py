"""
RL Training: GRPO with composite render-in-loop reward.
  python rl.py                        # style
  python rl.py --target x.png         # inverse
  python rl.py --mode explore --judge  # explore + aesthetic

Saves to rl_runs/{run}/ — view at bb-particles /refs/runs.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from infer import (
    DEFAULT_ENDPOINT, DISCOVER_PROMPTS, SYSTEM_PROMPT,
    fill_defaults, load_vlm, parse_params, render_api, to_lpips_tensor, vlm_generate,
)
from scoring import ollama_score
from utils import CATEGORICAL_KEYS

# Per-mode reward weights. Keys: lpips_ref, lpips_target, aesthetic
W_DEFAULT = {
    "style":   {"lpips_ref": 1.0, "lpips_target": 0.0, "aesthetic": 0.0},
    "inverse": {"lpips_ref": 0.0, "lpips_target": 1.0, "aesthetic": 0.0},
    "explore": {"lpips_ref": 1.0, "lpips_target": 0.0, "aesthetic": 0.0},
}
W_JUDGE = {
    "style":   {"lpips_ref": 0.8, "lpips_target": 0.0, "aesthetic": 0.2},
    "inverse": {"lpips_ref": 0.0, "lpips_target": 1.0, "aesthetic": 0.0},
    "explore": {"lpips_ref": 0.3, "lpips_target": 0.0, "aesthetic": 0.7},
}


def best_lpips(rendered, targets, lpips_fn):
    """CPU LPIPS — avoids stacking AlexNet on the same GPU as the 7B VLM."""
    if rendered is None or not targets:
        return 0.0
    r = to_lpips_tensor(rendered).unsqueeze(0)
    with torch.no_grad():
        return max(-lpips_fn(r, to_lpips_tensor(t).unsqueeze(0)).item() for t in targets)


def composite_score(img, refs, targets, lpips_fn, w, judge_refs=None):
    """Multi-signal reward. Returns (total, {component: value})."""
    if img is None or np.array(img).mean() < 5:
        return -1.0, {"lpips_ref": -1, "lpips_target": -1, "aesthetic": -1}

    c = {
        "lpips_ref": best_lpips(img, refs, lpips_fn),
        "lpips_target": best_lpips(img, targets, lpips_fn),
        "aesthetic": 0.0,
    }

    if w.get("aesthetic", 0) > 0 and judge_refs:
        score, _, _, _ = ollama_score(img, judge_refs)
        c["aesthetic"] = (score - 50) / 50

    return sum(w.get(k, 0) * v for k, v in c.items()), c


def _mask_prefix(labels, marker):
    row = labels[0].tolist()
    n, m = len(marker), len(row)
    pos = next((i for i in range(m - n, -1, -1) if row[i : i + n] == marker), -1)
    if pos >= 0:
        labels[:, : pos + n] = -100


def rl_batch(processor, messages, completion, device):
    msgs = [*messages, {"role": "assistant", "content": completion}]
    text = processor.apply_chat_template(msgs, tokenize=False)

    imgs = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "image":
                    imgs.append(Image.open(part["image"]).convert("RGB"))

    batch = processor(
        text=[text], images=imgs if imgs else None,
        padding=True, truncation=True, max_length=4096, return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    labels = batch["input_ids"].clone()
    _mask_prefix(labels, processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


def completion_logps(model, batch):
    labels = batch["labels"]
    out = model(**{k: v for k, v in batch.items() if k != "labels"})
    logits, ids = out.logits[:, :-1], batch["input_ids"][:, 1:]
    mask = labels[:, 1:].ne(-100)
    lp = torch.log_softmax(logits, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return lp, mask


class Stats:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.ema = None
        self.best_ever = -float("inf")
        self.stale = 0
        self.pf = self.rf = self.n = self.steps = self.zero_std = 0
        self.geos: dict[str, int] = {}

    def update(self, rewards, pf, rf, params_list):
        valid = [r for r in rewards if r > -1.0]
        best = float(np.max(rewards))
        improved = best > self.best_ever

        mean = float(np.mean(valid)) if valid else -1.0
        self.ema = mean if self.ema is None else self.alpha * mean + (1 - self.alpha) * self.ema

        if improved:
            self.best_ever, self.stale = best, 0
        else:
            self.stale += 1

        self.pf += pf
        self.rf += rf
        self.n += len(rewards)
        self.steps += 1

        for p in params_list:
            g = p.get("geometry", "?")
            self.geos[g] = self.geos.get(g, 0) + 1

        return improved

    @property
    def diversity(self):
        if not self.geos:
            return 0.0
        total = sum(self.geos.values())
        probs = [c / total for c in self.geos.values()]
        return -sum(p * np.log(p + 1e-10) for p in probs) / np.log(len(CATEGORICAL_KEYS["geometry"]))

    def summary(self):
        zf = f"{self.zero_std}/{self.steps}" if self.steps else "0"
        return (
            f"  ema={self.ema:+.3f}  best_ever={self.best_ever:+.3f}  "
            f"diversity={self.diversity:.0%}  fail={self.pf + self.rf}/{self.n}  "
            f"stale={self.stale}  zero_std={zf}"
        )


def make_messages(mode, target_path=None, ref_paths=None):
    if mode == "inverse" and target_path:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": target_path},
                {"type": "text", "text": "Predict the full scene parameters for this rendered image."},
            ]},
        ]

    content = []
    if ref_paths:
        content.append({"type": "image", "image": random.choice(ref_paths)})
        content.append({"type": "text", "text": "Generate scene parameters inspired by this reference artwork. Output valid JSON."})
    else:
        content.append({"type": "text", "text": random.choice(DISCOVER_PROMPTS)})

    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def grpo_step(model, processor, device, optimizer, messages, G, refs, targets,
              lpips_fn, w, endpoint, track_kl, stats, judge_refs=None):
    completions, params_list, images = [], [], []
    pf = rf = 0

    for _ in range(G):
        resp, params, img = "", {}, None
        try:
            resp = vlm_generate(model, processor, device, messages, temperature=0.8, max_tokens=512)
            params = fill_defaults(parse_params(resp))
        except Exception:
            pf += 1
        if resp and params:
            img = render_api(params, endpoint)
            if img is None:
                rf += 1
        completions.append(resp)
        params_list.append(params)
        images.append(img)

    scored = [composite_score(img, refs, targets, lpips_fn, w, judge_refs) for img in images]
    raw = np.array([s[0] for s in scored])
    comps = [s[1] for s in scored]

    valid = np.array([img is not None and np.array(img).mean() >= 5 for img in images])
    parsed = np.array([bool(completions[i] and params_list[i]) for i in range(G)])
    train_m = parsed & valid

    adv = np.zeros(G)
    m = int(train_m.sum())
    skipped = False
    reward_std = 0.0

    if m > 1:
        v = raw[train_m]
        reward_std = float(v.std())
        if reward_std < 1e-4:
            skipped = True
            stats.zero_std += 1
        else:
            adv[train_m] = np.clip((v - v.mean()) / (reward_std + 1e-8), -2.0, 2.0)
    elif m == 1:
        adv[train_m] = 1.0

    old_lp = {}
    kl_acc, n_kl = 0.0, 0

    if not skipped:
        for i in range(G):
            if not train_m[i]:
                continue
            batch = rl_batch(processor, messages, completions[i], device)
            if track_kl:
                with torch.no_grad():
                    lp, mask = completion_logps(model, batch)
                    old_lp[i] = (lp.detach(), mask)
            scale = max(m, 1)
            (float(adv[i]) * model(**batch).loss / scale).backward()
            n_kl += 1

        if n_kl:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if track_kl and old_lp:
                for i, (lp0, mask0) in old_lp.items():
                    batch = rl_batch(processor, messages, completions[i], device)
                    with torch.no_grad():
                        lp1, mask1 = completion_logps(model, batch)
                    k = min(lp0.shape[1], lp1.shape[1], mask0.shape[1], mask1.shape[1])
                    d = (lp0[:, :k] - lp1[:, :k]).abs()
                    mx = mask0[:, :k] & mask1[:, :k]
                    if mx.any():
                        kl_acc += d[mx].mean().item()
                kl_acc /= len(old_lp)

    improved = stats.update(raw.tolist(), pf, rf, params_list)
    j = int(raw.argmax())

    vc = [c for i, c in enumerate(comps) if train_m[i]]
    mc = {k: float(np.mean([c[k] for c in vc])) for k in vc[0]} if vc else {}

    return {
        "mean": float(raw[valid].mean()) if valid.any() else -1.0,
        "best": float(raw.max()),
        "kl": kl_acc, "n": int(train_m.sum()), "pf": pf, "rf": rf,
        "std": reward_std, "skipped": skipped, "improved": improved,
        "best_img": images[j], "best_params": params_list[j],
        "components": mc,
    }


def latest_rl_adapter():
    root = Path("rl_runs")
    if not root.is_dir():
        return None
    cfgs = [p for p in root.glob("*/adapter/adapter_config.json") if p.is_file()]
    return str(max(cfgs, key=lambda p: p.stat().st_mtime).parent) if cfgs else None


def train(
    mode="style", target_path=None, adapter="auto", endpoint=DEFAULT_ENDPOINT,
    steps=100, G=8, lr=5e-5, kl_coeff=0.05, warmup=5, save_every=10,
    run_name=None, use_judge=False,
):
    w = (W_JUDGE if use_judge else W_DEFAULT)[mode]
    run = Path("rl_runs") / (run_name or f"{mode}_{int(time.time())}")

    for d in ("images", "params", "checkpoints"):
        (run / d).mkdir(parents=True, exist_ok=True)

    ref_paths = sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg")
        for d in (Path("references"), Path("data/refs/images"))
        if d.is_dir() for p in d.glob(ext)
    )
    ref_paths = [str(p) for p in ref_paths[:16]]
    refs = [Image.open(p).convert("RGB") for p in ref_paths] if w["lpips_ref"] > 0 else []
    targets = [Image.open(target_path).convert("RGB")] if target_path else []
    judge_refs = [Image.open(p).convert("RGB") for p in ref_paths[:4]] if use_judge and ref_paths else None

    if adapter == "auto":
        chained = latest_rl_adapter()
        adapter = chained or "models"
        if chained:
            print(f"  chaining from {adapter}")

    model, processor, device = load_vlm(adapter)
    model.train()
    for n, p in model.named_parameters():
        if "lora" in n.lower():
            p.requires_grad = True

    import lpips as lpips_mod
    lpips_fn = lpips_mod.LPIPS(net="alex", verbose=False).cpu().eval()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))
    track_kl = kl_coeff > 0

    log = open(run / "rewards.csv", "w", newline="")
    csv_w = csv.writer(log)
    csv_w.writerow([
        "step", "mean", "best", "ema", "kl", "n", "pf", "rf",
        "lpips_ref", "lpips_target", "aesthetic", "std", "skipped",
        "div", "lr", "dt",
    ])

    with open(run / "config.json", "w") as f:
        json.dump({
            "mode": mode, "target": target_path, "steps": steps, "G": G,
            "lr": lr, "kl": kl_coeff, "warmup": warmup, "w": w,
            "refs": len(refs), "judge": use_judge,
        }, f, indent=2)

    stats = Stats()
    print(f"\n  {mode}{'+judge' if use_judge else ''}  steps={steps}  G={G}  lr={lr}  kl={kl_coeff}  warmup={warmup}")
    print(f"  refs={len(refs)}  targets={len(targets)}  w={w}")
    print(f"  {run}/\n", flush=True)

    for step in range(steps):
        t0 = time.time()

        r = grpo_step(
            model, processor, device, optimizer,
            make_messages(mode, target_path, ref_paths if refs else None),
            G, refs, targets, lpips_fn, w, endpoint, track_kl, stats, judge_refs,
        )

        scheduler.step()
        dt = time.time() - t0
        mc = r["components"]

        csv_w.writerow([
            step, f"{r['mean']:.4f}", f"{r['best']:.4f}", f"{stats.ema:.4f}",
            f"{r['kl']:.4f}", r["n"], r["pf"], r["rf"],
            f"{mc.get('lpips_ref', 0):.4f}", f"{mc.get('lpips_target', 0):.4f}",
            f"{mc.get('aesthetic', 0):.4f}", f"{r['std']:.4f}", int(r["skipped"]),
            f"{stats.diversity:.2f}", f"{scheduler.get_last_lr()[0]:.2e}", f"{dt:.0f}",
        ])
        log.flush()

        if r["best_img"]:
            r["best_img"].save(run / f"images/{step:06d}.png")
            with open(run / f"params/{step:06d}.json", "w") as f:
                json.dump({**r["best_params"], "_reward": round(r["best"], 4)}, f)

        bar = int(max(0, (r["best"] + 1) * 10))
        print(
            f"  [{step:>4d}/{steps}]  "
            f"r={r['mean']:+.3f}  best={r['best']:+.3f}  ema={stats.ema:+.3f}  "
            f"kl={r['kl']:.3f}  "
            f"{'█' * bar}{'░' * (20 - bar)}  "
            f"{'SKIP ' if r['skipped'] else ''}{r['n']}/{G}{'*' if r['improved'] else ''}  "
            f"{dt:.0f}s"
        )

        if save_every > 0 and (step + 1) % save_every == 0:
            model.save_pretrained(str(run / f"checkpoints/step_{step:04d}"))
            processor.save_pretrained(str(run / f"checkpoints/step_{step:04d}"))

        if stats.stale >= 20:
            print(f"  !! stale for {stats.stale} steps")

    log.close()
    print(f"\n{stats.summary()}\n  geos: {stats.geos}")

    model.save_pretrained(str(run / "adapter"))
    processor.save_pretrained(str(run / "adapter"))
    print(f"  saved {run / 'adapter'}")

    from harvest import harvest
    harvest(threshold=-0.3, source=run)
    print(f"  auto-harvested winners from {run.name}")


def main():
    p = argparse.ArgumentParser(description="GRPO RL with composite render-in-loop reward")
    p.add_argument("--mode", choices=["style", "inverse", "explore"], default="style")
    p.add_argument("--target", help="Target image (inverse mode)")
    p.add_argument("--adapter", default="auto")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--kl", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--judge", action="store_true", help="Enable aesthetic judge scoring")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    if args.mode == "inverse" and not args.target:
        p.error("--target required for inverse mode")

    train(
        mode=args.mode, target_path=args.target, adapter=args.adapter,
        endpoint=args.endpoint, steps=args.steps, G=args.group,
        lr=args.lr, kl_coeff=args.kl, warmup=args.warmup,
        save_every=args.save_every, run_name=args.name, use_judge=args.judge,
    )


if __name__ == "__main__":
    main()
