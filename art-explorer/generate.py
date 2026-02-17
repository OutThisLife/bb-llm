"""
Data Generation
===============
Generate synthetic (image, params) pairs via /api/raster.
Supports random, ref-biased, and VLM-score-weighted breeding.
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

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from utils import random_params, to_scene_params

DATA_DIR = Path("data")
REFS_PATH = Path("references/refs.jsonl")
SCORED_PATH = Path("references/refs-scored.jsonl")
API_URL = "http://localhost:3000/api/raster"
MAX_WORKERS = min(os.cpu_count() or 4, 12)


def load_refs():
    """Load refs.jsonl for breeding."""
    if not REFS_PATH.exists():
        return []
    lines = REFS_PATH.read_text().strip().split("\n")
    return [json.loads(ln) for ln in lines if ln]


def load_scored_refs():
    """Load refs-scored.jsonl with weights for biased sampling."""
    if not SCORED_PATH.exists():
        print("refs-scored.jsonl not found. Run `make score-refs` first.")
        return [], []
    entries = []
    for ln in SCORED_PATH.read_text().strip().split("\n"):
        if ln:
            entries.append(json.loads(ln))
    params = [e["params"] for e in entries]
    scores = [e["score"] for e in entries]
    return params, scores


_pbar_lock = threading.Lock()
_pbar = None
_trace = None

_session_local = threading.local()


def _write_trace(msg):
    with _pbar_lock:
        (_pbar.write(msg) if _pbar else print(msg))


def _get_session():
    """Get thread-local session with retry."""
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


def clear_data():
    """Remove existing data."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print("Cleared data/")


def generate_one(i, prefixed, data_dir=DATA_DIR):
    flat = to_scene_params(prefixed)
    t0 = time.perf_counter()
    w = threading.current_thread().name
    if _trace:
        _trace(f"[start] {i:06d} {w}")

    try:
        resp = _get_session().post(API_URL, json=prefixed, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        if _trace:
            _trace(f"[fail] {i:06d} {w} {time.perf_counter()-t0:.1f}s {e}")
        print(f"\nSkipping {i:06d}: {e}")
        return None, None

    img_path = data_dir / f"images/{i:06d}.png"
    img_path.write_bytes(resp.content)
    flat["url"] = f"http://localhost:3000/render?c={resp.headers.get('X-Encoded-Params', '')}"
    (data_dir / f"params/{i:06d}.json").write_text(json.dumps(flat))

    _update_progress()
    if _trace:
        _trace(f"[done] {i:06d} {w} {time.perf_counter()-t0:.1f}s")
    return img_path, flat


def _breed_from_refs(refs, n_layers, weights=None):
    """Crossover + partial inheritance from refs."""
    if weights:
        if len(refs) >= 2 and random.random() < 0.7:
            parents = random.choices(refs, weights=weights, k=2)
        else:
            parents = random.choices(refs, weights=weights, k=1)
    else:
        if len(refs) >= 2 and random.random() < 0.7:
            parents = random.sample(refs, 2)
        else:
            parents = [random.choice(refs)]

    child = random_params(n_layers=n_layers)

    for k in child:
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


def make_params(refs, n_layers, weights=None, breed_rate=0.3):
    """Generate params: stratified categoricals + optional ref breeding."""
    layers = n_layers if n_layers >= 0 else None

    if refs and random.random() < breed_rate:
        return _breed_from_refs(refs, layers, weights)

    return random_params(n_layers=layers, stratified=True)


def generate(
    n=2000,
    n_layers=0,
    include_refs=True,
    workers=MAX_WORKERS,
    scored=False,
    out_dir=None,
    trace_workers=False,
):
    global _pbar, _trace
    _trace = _write_trace if trace_workers else None

    data_dir = Path(out_dir) if out_dir else DATA_DIR

    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "params").mkdir(parents=True, exist_ok=True)

    # Continue from highest existing index
    existing = list((data_dir / "images").glob("*.png"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Continuing from index {start_idx} ({start_idx} existing)")

    # Load refs (scored or regular)
    weights = None
    breed_rate = 0.3

    if scored:
        refs, weights = load_scored_refs()
        breed_rate = 0.7
        if refs:
            print(f"Scored mode: {len(refs)} refs, weighted breeding @ 70%")
    elif include_refs:
        refs = load_refs()
        if refs:
            print(f"Including {len(refs)} reference params (30% breeding)")
    else:
        refs = []

    print(f"Generating {n} samples (parallel, {workers} workers)...")

    all_params = [make_params(refs, n_layers, weights, breed_rate) for _ in range(n)]
    random.shuffle(all_params)

    with tqdm(total=n, desc="Rendering", unit="img") as pbar:
        _pbar = pbar
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(generate_one, start_idx + i, p, data_dir)
                for i, p in enumerate(all_params)
            ]
            for f in futures:
                f.result()
        _pbar = None

    print(f"Done: {n} samples in {data_dir}/")


def main():
    p = argparse.ArgumentParser(description="Generate training data")
    p.add_argument("-n", type=int, default=2000, help="Number of samples")
    p.add_argument(
        "--layers", type=int, default=-1,
        help="Layers per image (-1=random, 0=none, 1+=fixed)",
    )
    p.add_argument("--clean", action="store_true", help="Clear existing data before generating")
    p.add_argument("--scored", action="store_true", help="Use VLM-scored refs for weighted breeding")
    p.add_argument("--trace-workers", action="store_true", help="Verbose per-job worker timing logs")
    p.add_argument("--out", default=None, help="Output directory (default: data/)")
    args = p.parse_args()

    if args.clean:
        clear_data()

    generate(
        n=args.n,
        n_layers=args.layers,
        scored=args.scored,
        workers=1 if args.scored else MAX_WORKERS,
        out_dir=args.out,
        trace_workers=args.trace_workers,
    )


if __name__ == "__main__":
    main()
