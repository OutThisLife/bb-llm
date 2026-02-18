"""
Data Generation
===============
Generate synthetic (image, params) pairs via /api/raster.
Random + ref-biased breeding for coverage.
"""

import argparse
import contextlib
import json
import os
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from utils import random_params, to_scene_params

DATA_DIR = Path("data")
REFS_PATH = Path("references/refs.jsonl")
API_URL = "http://localhost:3000/api/raster"
MAX_WORKERS = min(os.cpu_count() or 4, 12)


def load_refs():
    """Load refs.jsonl for breeding."""
    if not REFS_PATH.exists():
        return []
    lines = REFS_PATH.read_text().strip().split("\n")
    return [json.loads(ln) for ln in lines if ln]


_pbar_lock = threading.Lock()
_pbar = None
_session_local = threading.local()


def _get_session():
    if not hasattr(_session_local, "session"):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5)
        session.mount("http://", HTTPAdapter(max_retries=retry))
        _session_local.session = session
    return _session_local.session


def _update_progress():
    with _pbar_lock:
        if _pbar is not None:
            with contextlib.suppress(Exception):
                _pbar.update(1)


def generate_one(i, prefixed, data_dir=DATA_DIR):
    flat = to_scene_params(prefixed)
    try:
        resp = _get_session().post(API_URL, json=prefixed, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"\nSkipping {i:06d}: {e}")
        return None, None

    img_path = data_dir / f"images/{i:06d}.png"
    img_path.write_bytes(resp.content)
    flat["url"] = f"http://localhost:3000/render?c={resp.headers.get('X-Encoded-Params', '')}"
    (data_dir / f"params/{i:06d}.json").write_text(json.dumps(flat))
    _update_progress()
    return img_path, flat


def _breed_from_refs(refs, n_layers):
    """Crossover + partial inheritance from refs."""
    if len(refs) >= 2 and random.random() < 0.7:
        parents = random.sample(refs, 2)
    else:
        parents = [random.choice(refs)]

    child = random_params(n_layers=n_layers)

    # Don't let breeding corrupt mirror layer structure
    _mirror_keys = {"position", "rotation", "scale"}
    for k in child:
        if k.startswith("Groups.") and k.split("-", 1)[-1] in _mirror_keys:
            continue
        if random.random() < 0.6:
            parent = random.choice(parents)
            if k in parent:
                val = parent[k]
                if isinstance(val, (int, float)):
                    val = val * random.uniform(0.8, 1.2)
                    if isinstance(parent[k], int):
                        val = round(val)
                child[k] = val

    return child


def make_params(refs, n_layers, breed_rate=0.3):
    """Generate params: stratified categoricals + optional ref breeding."""
    layers = n_layers if n_layers >= 0 else None

    if refs and random.random() < breed_rate:
        return _breed_from_refs(refs, layers)

    return random_params(n_layers=layers, stratified=True)


def generate(n=2000, n_layers=-1, workers=MAX_WORKERS):
    global _pbar

    (DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "params").mkdir(parents=True, exist_ok=True)

    existing = list((DATA_DIR / "images").glob("*.png"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Continuing from index {start_idx} ({start_idx} existing)")

    refs = load_refs()
    if refs:
        print(f"Including {len(refs)} refs (30% breeding)")

    print(f"Generating {n} samples ({workers} workers)...")

    all_params = [make_params(refs, n_layers) for _ in range(n)]
    random.shuffle(all_params)

    with tqdm(total=n, desc="Rendering", unit="img") as pbar:
        _pbar = pbar
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(generate_one, start_idx + i, p)
                for i, p in enumerate(all_params)
            ]
            for f in futures:
                f.result()
        _pbar = None

    print(f"Done: {n} samples in {DATA_DIR}/")


def main():
    p = argparse.ArgumentParser(description="Generate training data")
    p.add_argument("-n", type=int, default=2000, help="Number of samples")
    p.add_argument("--layers", type=int, default=-1, help="Layers (-1=random)")
    p.add_argument("--clean", action="store_true", help="Clear data first")
    args = p.parse_args()

    if args.clean:
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
            print("Cleared data/")

    generate(n=args.n, n_layers=args.layers)


if __name__ == "__main__":
    main()
