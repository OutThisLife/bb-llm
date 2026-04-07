"""Data generation: (image, params) pairs via /api/raster."""

import argparse
import contextlib
import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from utils import random_params, to_scene_params

DATA = Path("data")
API = "http://localhost:3000/api/raster"
WORKERS = 2
FAIL_THRESHOLD = 5

_pbar_lock = threading.Lock()
_pbar = None
_local = threading.local()
_fails = 0
_flock = threading.Lock()


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.mount("http://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))
    return _local.s


def _wait_healthy():
    global _fails
    print(f"\n⚠ {_fails} fails, waiting for renderer…")

    for i in range(30):
        time.sleep(min(5 * (i + 1), 30))
        try:
            data = requests.get(f"{API}?health", timeout=5).json()
            if data.get("ok") and not data.get("recycling"):
                with _flock:
                    _fails = 0
                print(f"  healthy after {5 * (i + 1)}s")
                return
        except Exception:
            pass

    print("⚠ still unhealthy, continuing…")


def generate_one(i, prefixed):
    global _fails

    with _flock:
        should_wait = _fails >= FAIL_THRESHOLD
    if should_wait:
        _wait_healthy()

    resp = None
    for attempt in range(3):
        try:
            resp = _session().post(API, json=prefixed, timeout=60)
            if resp.status_code == 503:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                with _flock:
                    _fails += 1
                print(f"\nSkip {i:06d}: {e}")
                return
            time.sleep(5 * (attempt + 1))

    if not resp or resp.status_code != 200:
        with _flock:
            _fails += 1
        return

    with _flock:
        _fails = 0

    img_path = DATA / f"images/{i:06d}.png"
    img_path.write_bytes(resp.content)

    if np.array(Image.open(img_path)).mean() < 5:
        img_path.unlink()
        return

    flat = to_scene_params(prefixed)
    flat["url"] = f"http://localhost:3000/render?c={resp.headers.get('X-Encoded-Params', '')}"
    (DATA / f"params/{i:06d}.json").write_text(json.dumps(flat))

    with _pbar_lock:
        if _pbar:
            with contextlib.suppress(Exception):
                _pbar.update(1)


def generate(n=2000, workers=WORKERS):
    global _pbar

    for d in ("images", "params"):
        (DATA / d).mkdir(parents=True, exist_ok=True)

    existing = list((DATA / "images").glob("*.png"))
    start = max((int(p.stem) for p in existing), default=-1) + 1
    if start:
        print(f"Continuing from {start} ({len(existing)} images)")

    print(f"Generating {n} samples ({workers} workers)…")

    params = [random_params(stratified=True) for _ in range(n)]
    random.shuffle(params)

    with tqdm(total=n, desc="Rendering", unit="img") as pbar:
        _pbar = pbar
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(generate_one, start + i, p) for i, p in enumerate(params)]
            for f in futs:
                f.result()
        _pbar = None

    print(f"Done: {n} samples in {DATA}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()

    if args.clean and DATA.exists():
        shutil.rmtree(DATA)

    generate(n=args.n)
