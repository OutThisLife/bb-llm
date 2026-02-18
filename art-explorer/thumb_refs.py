"""Generate cached thumbnails + params for references."""

import argparse
import hashlib
import json
from pathlib import Path

import requests

from utils import to_scene_params

REFS_PATH = Path("references/refs.jsonl")
IMG_DIR = Path("data/refs/images")
PARAM_DIR = Path("data/refs/params")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:3000")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not REFS_PATH.exists():
        print("No refs found")
        return

    lines = [ln for ln in REFS_PATH.read_text().splitlines() if ln.strip()]
    if not lines:
        print("No refs found")
        return

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    PARAM_DIR.mkdir(parents=True, exist_ok=True)

    api = f"{args.base_url.rstrip('/')}/api/raster?size={args.size}"
    valid_tids = set()
    done, skipped, failed = 0, 0, 0

    for i, raw in enumerate(lines, 1):
        try:
            params = json.loads(raw)
        except Exception:
            failed += 1
            continue

        tid = hashlib.sha1(raw.encode()).hexdigest()[:16]
        valid_tids.add(tid)
        img = IMG_DIR / f"{tid}.jpg"
        par = PARAM_DIR / f"{tid}.json"

        # Always ensure params exist (cheap, no API call)
        if not par.exists():
            par.write_text(json.dumps(to_scene_params(params)))

        if img.exists() and not args.force:
            skipped += 1
            continue

        try:
            r = requests.post(api, json=params, timeout=60)
            r.raise_for_status()
            img.write_bytes(r.content)

            flat = to_scene_params(params)
            flat["url"] = f"{args.base_url}/render?c={r.headers.get('X-Encoded-Params', '')}"
            par.write_text(json.dumps(flat))

            done += 1
        except Exception as e:
            failed += 1
            print(f"Failed line {i}: {e}")

    # Prune stale refs no longer in refs.jsonl
    pruned = 0
    for d in (IMG_DIR, PARAM_DIR):
        for f in d.iterdir():
            if f.stem not in valid_tids:
                f.unlink()
                pruned += 1

    print(f"thumb-refs: {done} new, {skipped} cached, {failed} failed, {pruned} pruned")


if __name__ == "__main__":
    main()
