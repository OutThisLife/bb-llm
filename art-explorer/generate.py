"""
Data Generation: (image, params) pairs via /api/raster.
Stratified random sampling for full parameter space coverage.
"""

import argparse
import contextlib
import json
import os
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

DATA_DIR = Path("data")
API_URL = "http://localhost:3000/api/raster"
HEALTH_URL = "http://localhost:3000/api/raster?health"
MAX_WORKERS = 2
FAIL_BACKOFF_THRESHOLD = 5
FAIL_BACKOFF_MAX = 120

_pbar_lock = threading.Lock()
_pbar = None
_session_local = threading.local()
_consecutive_fails = 0
_fail_lock = threading.Lock()


def _get_session():
    if not hasattr(_session_local, "session"):
        s = requests.Session()
        s.mount("http://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))
        _session_local.session = s
    return _session_local.session


def _wait_for_healthy():
    global _consecutive_fails

    print(f"\n⚠ {_consecutive_fails} fails, waiting for renderer…")

    for attempt in range(30):
        time.sleep(min(5 * (attempt + 1), 30))
        try:
            r = requests.get(HEALTH_URL, timeout=5)
            data = r.json() if r.ok else {}
            if data.get("ok") and not data.get("recycling"):
                with _fail_lock:
                    _consecutive_fails = 0
                print(f"  renderer healthy after {(attempt+1)*5}s")
                return
        except Exception:
            pass

    print("⚠ renderer still unhealthy, continuing…")


def generate_one(i, prefixed):
    global _consecutive_fails
    flat = to_scene_params(prefixed)

    with _fail_lock:
        if _consecutive_fails >= FAIL_BACKOFF_THRESHOLD:
            _wait_for_healthy()

    resp = None
    for attempt in range(3):
        try:
            resp = _get_session().post(API_URL, json=prefixed, timeout=60)
            if resp.status_code == 503:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        except Exception as e:
            resp = None
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            with _fail_lock:
                _consecutive_fails += 1
            print(f"\nSkipping {i:06d}: {e}")
            return

    if resp is None or resp.status_code != 200:
        with _fail_lock:
            _consecutive_fails += 1
        return

    with _fail_lock:
        _consecutive_fails = 0

    img_path = DATA_DIR / f"images/{i:06d}.png"
    img_path.write_bytes(resp.content)

    if np.array(Image.open(img_path)).mean() < 5:
        img_path.unlink()
        return

    flat["url"] = f"http://localhost:3000/render?c={resp.headers.get('X-Encoded-Params', '')}"
    (DATA_DIR / f"params/{i:06d}.json").write_text(json.dumps(flat))

    with _pbar_lock:
        if _pbar:
            with contextlib.suppress(Exception):
                _pbar.update(1)


def generate(n=2000, workers=MAX_WORKERS):
    global _pbar

    for d in ("images", "params"):
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)

    existing = list((DATA_DIR / "images").glob("*.png"))
    start = max((int(p.stem) for p in existing), default=-1) + 1
    if start:
        print(f"Continuing from {start} ({len(existing)} images)")

    print(f"Generating {n} samples ({workers} workers)...")

    all_params = [random_params(stratified=True) for _ in range(n)]
    random.shuffle(all_params)

    with tqdm(total=n, desc="Rendering", unit="img") as pbar:
        _pbar = pbar
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(generate_one, start + i, p) for i, p in enumerate(all_params)]
            for f in futs:
                f.result()
        _pbar = None

    print(f"Done: {n} samples in {DATA_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate training data")
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--clean", action="store_true", help="Clear data first")
    args = p.parse_args()

    if args.clean and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print("Cleared data/")

    generate(n=args.n)
