"""Generate cached thumbnails for references."""

import argparse
import hashlib
import json
from pathlib import Path

import requests

REFS_PATH = Path("references/refs.jsonl")
OUT_DIR = Path("data/refs/images")


def iter_ref_lines():
    if not REFS_PATH.exists():
        return []
    return [line for line in REFS_PATH.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:3000")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    lines = iter_ref_lines()
    if not lines:
        print("No refs found")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = f"{args.base_url.rstrip('/')}/api/raster?size={args.size}"

    done = 0
    skipped = 0
    failed = 0

    for i, raw in enumerate(lines, 1):
        try:
            params = json.loads(raw)
        except Exception:
            failed += 1
            print(f"Invalid JSON at line {i}")
            continue

        tid = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        out = OUT_DIR / f"{tid}.jpg"

        if out.exists() and not args.force:
            skipped += 1
            continue

        try:
            r = requests.post(api, json=params, timeout=60)
            r.raise_for_status()
            out.write_bytes(r.content)
            done += 1
        except Exception as e:
            failed += 1
            print(f"Failed line {i}: {e}")

    print(f"thumb-refs: generated={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
