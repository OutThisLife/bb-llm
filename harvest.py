"""Harvest: move winning RL samples into SFT training pool."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from cli import cli

DATA_DIR = Path("data")
RL_DIR = Path("rl_runs")


def params_hash(params):
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:16]


def existing_hashes():
    hashes = set()
    param_dir = DATA_DIR / "params"
    if not param_dir.exists():
        return hashes
    for p in param_dir.glob("*.json"):
        try:
            hashes.add(params_hash(json.loads(p.read_text())))
        except Exception:
            continue
    return hashes


def harvest(threshold=-0.3, dry_run=False, source=None):
    """Harvest RL winners into data/. source=Path for a single run, None for all."""
    scan = source if source else RL_DIR
    if not scan.exists():
        return print(f"No {scan}/ found")

    for d in ("images", "params"):
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)

    existing = existing_hashes()
    next_id = max(
        (int(p.stem) for p in (DATA_DIR / "images").glob("*.png")),
        default=-1,
    ) + 1

    glob = "params/*.json" if source else "*/params/*.json"
    candidates = []

    for pf in sorted(scan.glob(glob)):
        try:
            params = json.loads(pf.read_text())
        except Exception:
            continue

        reward = params.get("_reward", -999)
        if reward < threshold:
            continue

        img = pf.parent.parent / "images" / f"{pf.stem}.png"
        if not img.exists():
            continue

        h = params_hash(params)
        if h in existing:
            continue

        candidates.append((pf, img, params, reward, h))

    if not candidates:
        return

    candidates.sort(key=lambda x: -x[3])
    added = 0

    for pf, img, params, reward, h in candidates:
        if h in existing:
            continue

        run = pf.parent.parent.name
        clean = {k: v for k, v in params.items() if not k.startswith("_")}
        clean["_provenance"] = {"source_run": run, "reward": reward, "step": pf.stem}

        tid = f"{next_id:06d}"
        if not dry_run:
            shutil.copy2(img, DATA_DIR / f"images/{tid}.png")
            (DATA_DIR / f"params/{tid}.json").write_text(json.dumps(clean))

        existing.add(h)
        next_id += 1
        added += 1
        print(f"  {run}/{pf.stem} → {tid}  r={reward:+.3f}")

    if added:
        print(f"  harvested {added} → data/ (total: {next_id})")


@cli
def main():
    p = argparse.ArgumentParser(description="Harvest RL winners into SFT data pool")
    p.add_argument("--threshold", type=float, default=-0.3, help="Min reward")
    p.add_argument("--run", type=Path, default=None, help="Specific run dir")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    harvest(threshold=args.threshold, dry_run=args.dry_run, source=args.run)


if __name__ == "__main__":
    main()
