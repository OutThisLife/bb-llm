"""
Phase 1: Data Generation
========================
Generate synthetic (image, params) pairs via /api/raster.
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
DATA_SCORED_DIR = Path("data-scored")
REFS_PATH = Path("references/refs.jsonl")
SCORED_PATH = Path("references/refs-scored.jsonl")
API_URL = "http://localhost:3000/api/raster"
# Dynamic workers: scale with CPU, cap at 8
MAX_WORKERS = min(8, max(2, os.cpu_count() or 4))


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


# Thread-safe progress
_pbar_lock = threading.Lock()
_pbar = None

# Thread-local session for connection reuse
_session_local = threading.local()


def _get_session():
    """Get thread-local session with retry."""
    if not hasattr(_session_local, "session"):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5)
        session.mount("http://", HTTPAdapter(max_retries=retry))
        _session_local.session = session
    return _session_local.session


def _update_progress():
    """Thread-safe progress bar update."""
    with _pbar_lock:
        if _pbar is not None:
            with contextlib.suppress(Exception):
                _pbar.update(1)


def clear_data():
    """Remove existing data."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print("Cleared data/")


def setup_preview():
    """Setup matplotlib preview window."""
    import matplotlib

    with contextlib.suppress(Exception):
        matplotlib.use("QtAgg")

    import matplotlib.pyplot as plt

    plt.ion()
    plt.rcParams.update({"font.family": "monospace", "toolbar": "None"})

    fig = plt.figure(figsize=(6, 6), facecolor="#1a1a1a")

    # Image axis
    ax_img = fig.add_axes((0.05, 0.15, 0.9, 0.8))
    ax_img.axis("off")

    # Metadata axis
    ax_meta = fig.add_axes((0.05, 0.02, 0.9, 0.1))
    ax_meta.axis("off")

    fig.canvas.manager.set_window_title("Data Gen Preview")  # type: ignore[union-attr]
    fig.canvas.mpl_connect("close_event", lambda _: os._exit(0))

    return fig, ax_img, ax_meta


def update_preview(fig, ax_img, ax_meta, img_path, flat, i, n):
    """Update preview with current image and metadata."""
    from PIL import Image

    ax_img.clear()
    ax_img.imshow(Image.open(img_path), aspect="auto")
    ax_img.axis("off")
    ax_img.set_title(f"{i + 1}/{n}", color="#666", fontsize=10, pad=5)

    # Show key params
    geo = flat.get("geometry", "?")
    rep = flat.get("repetitions", "?")
    layers = len(flat.get("layers", []))

    ax_meta.clear()
    ax_meta.axis("off")
    meta_text = f"geo={geo}  rep={rep}  layers={layers}"
    ax_meta.text(
        0.5,
        0.5,
        meta_text,
        transform=ax_meta.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#888",
        family="monospace",
    )

    fig.canvas.draw()
    fig.canvas.flush_events()


def generate_one(i, prefixed):
    """Generate a single (image, params) pair via API."""
    flat = to_scene_params(prefixed)

    try:
        session = _get_session()
        resp = session.post(API_URL, json=flat, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"\nSkipping {i:06d}: {e}")
        return None, None

    # Save image
    img_path = DATA_DIR / f"images/{i:06d}.png"
    img_path.write_bytes(resp.content)

    # Get URL from header
    encoded = resp.headers.get("X-Encoded-Params", "")
    url = f"http://localhost:3000/render?c={encoded}"

    # Save params + url
    flat["url"] = url
    with open(DATA_DIR / f"params/{i:06d}.json", "w") as f:
        json.dump(flat, f)

    _update_progress()
    return img_path, flat


def _breed_from_refs(refs, n_layers, weights=None):
    """Crossover + partial inheritance from refs."""
    # Pick 1-2 parents (weighted if scores provided)
    if weights:
        # Weighted random selection
        if len(refs) >= 2 and random.random() < 0.7:
            parents = random.choices(refs, weights=weights, k=2)
        else:
            parents = random.choices(refs, weights=weights, k=1)
    else:
        # Uniform random
        if len(refs) >= 2 and random.random() < 0.7:
            parents = random.sample(refs, 2)
        else:
            parents = [random.choice(refs)]

    # Start with random params as base
    child = random_params(n_layers=n_layers)

    # Inherit ~60% of params from parents (crossover)
    for k in child:
        if random.random() < 0.6:  # 60% inherit, 40% stay random
            parent = random.choice(parents)
            if k in parent:
                val = parent[k]
                # Perturb numeric values ±20%
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
    preview=False,
    include_refs=True,
    workers=MAX_WORKERS,
    scored=False,
    data_dir=None,
):
    """Generate n synthetic (image, params) pairs via API."""
    global _pbar, DATA_DIR

    # Use appropriate data directory
    if data_dir:
        DATA_DIR = data_dir
    elif scored:
        DATA_DIR = DATA_SCORED_DIR

    (DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "params").mkdir(parents=True, exist_ok=True)

    # Continue from highest existing index
    existing = list((DATA_DIR / "images").glob("*.png"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Continuing from index {start_idx} ({start_idx} existing)")

    # Load refs (scored or regular)
    weights = None
    breed_rate = 0.3

    if scored:
        refs, weights = load_scored_refs()
        breed_rate = 0.7  # Higher breeding rate for scored mode
        if refs:
            print(f"Scored mode: {len(refs)} refs, weighted breeding @ 70%")
    elif include_refs:
        refs = load_refs()
        if refs:
            print(f"Including {len(refs)} reference params (30% breeding)")
    else:
        refs = []

    # Preview mode: serial (so we can display)
    if preview:
        fig, ax_img, ax_meta = setup_preview()
        pbar = tqdm(range(n), desc="Generating", unit="img")

        for i in pbar:
            prefixed = make_params(refs, n_layers, weights, breed_rate)
            img_path, flat = generate_one(start_idx + i, prefixed)
            if img_path:
                update_preview(fig, ax_img, ax_meta, img_path, flat, i, n)

        print(f"Done: {n} samples in {DATA_DIR}/")
        return

    # Parallel mode: fast
    print(f"Generating {n} samples (parallel, {workers} workers)...")

    # Pre-generate all params (fast, CPU-bound)
    all_params = [make_params(refs, n_layers, weights, breed_rate) for _ in range(n)]

    # Shuffle to break any sequential patterns in random state
    random.shuffle(all_params)

    # Parallel API calls - progress updated from workers via _update_progress()
    with tqdm(total=n, desc="Rendering", unit="img") as pbar:
        _pbar = pbar
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(generate_one, start_idx + i, p)
                for i, p in enumerate(all_params)
            ]
            # Wait for all to complete, check for errors
            for f in futures:
                f.result()
        _pbar = None

    print(f"Done: {n} samples in {DATA_DIR}/")


def main():
    p = argparse.ArgumentParser(description="Generate training data for inverse model")
    p.add_argument("-n", type=int, default=2000, help="Number of samples")
    p.add_argument(
        "--layers",
        type=int,
        default=-1,
        help="Layers per image (-1=random, 0=none, 1+=fixed)",
    )
    p.add_argument(
        "--clean", action="store_true", help="Clear existing data before generating"
    )
    p.add_argument("--preview", action="store_true", help="Show preview window")
    p.add_argument(
        "--scored",
        action="store_true",
        help="Use refs-scored.jsonl with weighted breeding → data-scored/",
    )
    args = p.parse_args()

    # Scored mode: clean data-scored/, preview by default, serial
    if args.scored:
        if args.clean:
            if DATA_SCORED_DIR.exists():
                shutil.rmtree(DATA_SCORED_DIR)
                print("Cleared data-scored/")
        generate(
            n=args.n,
            n_layers=args.layers,
            preview=True,  # always preview in scored mode
            scored=True,
            workers=1,  # serial for experimentation
        )
        return

    if args.clean:
        clear_data()

    generate(n=args.n, n_layers=args.layers, preview=args.preview)


if __name__ == "__main__":
    main()
