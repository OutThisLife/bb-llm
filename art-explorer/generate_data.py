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
from explore import load_refs, random_params
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from utils import to_scene_params

DATA_DIR = Path("data")
API_URL = "http://localhost:3000/api/raster"
# Dynamic workers: scale with CPU, cap at 8
MAX_WORKERS = min(8, max(2, os.cpu_count() or 4))

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


def _breed_from_refs(refs, n_layers):
    """Crossover + partial inheritance from refs."""
    # Pick 1-2 parents
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


def make_params(refs, n_layers):
    """Generate params, optionally biased by refs via crossover."""
    layers = n_layers if n_layers >= 0 else None

    # 30% chance to breed from refs
    if refs and random.random() < 0.05:
        return _breed_from_refs(refs, layers)

    return random_params(n_layers=layers)


def generate(n=2000, n_layers=0, preview=False, include_refs=True, workers=MAX_WORKERS):
    """Generate n synthetic (image, params) pairs via API."""
    global _pbar

    (DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "params").mkdir(parents=True, exist_ok=True)

    # Continue from highest existing index
    existing = list((DATA_DIR / "images").glob("*.png"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Continuing from index {start_idx} ({start_idx} existing)")

    refs = load_refs() if include_refs else []

    if refs:
        print(f"Including {len(refs)} reference params (30% breeding)")

    # Preview mode: serial (so we can display)
    if preview:
        fig, ax_img, ax_meta = setup_preview()
        pbar = tqdm(range(n), desc="Generating", unit="img")

        for i in pbar:
            prefixed = make_params(refs, n_layers)
            img_path, flat = generate_one(start_idx + i, prefixed)
            if img_path:
                update_preview(fig, ax_img, ax_meta, img_path, flat, i, n)

        print(f"Done: {n} samples")
        return

    # Parallel mode: fast
    print(f"Generating {n} samples (parallel, {workers} workers)...")

    # Pre-generate all params (fast, CPU-bound)
    all_params = [make_params(refs, n_layers) for _ in range(n)]

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

    print(f"Done: {n} samples")


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
    args = p.parse_args()

    if args.clean:
        clear_data()

    generate(n=args.n, n_layers=args.layers, preview=args.preview)


if __name__ == "__main__":
    main()
