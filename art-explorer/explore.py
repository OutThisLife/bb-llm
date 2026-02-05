"""
Generative Art Parameter Explorer
=================================
CMA-ES optimization with VLM scoring (Qwen2-VL).

Flow:
  1. Define parameter space (single unified schema)
  2. Encode params → URL for render server
  3. CMA-ES proposes candidates in [0,1]^n
  4. Decode to params, render, screenshot, score
  5. Feed scores back to CMA-ES
  6. Repeat until convergence

Run:
  python explore.py --preview --gens 10 --pop 4
"""

import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

from utils import clamp

# ============================================================================
# 1. UNIFIED SCHEMA
#    Single source of truth for all parameters
# ============================================================================

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

COLORS = [
    "#FFFDDD",
    "#FFFEF0",
    "#FFFFF0",
    "#FFF8DC",
    "#FFEFD5",
    "#FFE4B5",
    "#F5DEB3",
    "#EEE8AA",
    "#E8D9A0",
    "#D4C896",
]

# Schema: { name: { type, range?, options?, cma? } }
#   type: "int" | "float" | "bool" | "cat" | "obj"
#   range: (min, max) for numeric types
#   options: [...] for categorical
#   cma: True (use range) | (min, max) for optimizer | False/omit (not optimized)

SCHEMA = {
    # Scalars
    "Scalars.repetitions": {
        "type": "int",
        "range": (1, 500),
        "cma": (50, 500),
        "bias_min": 15,
    },
    "Scalars.alphaFactor": {"type": "float", "range": (0, 1), "cma": True},
    "Scalars.scaleFactor": {"type": "float", "range": (0, 2), "cma": (0.5, 3)},
    "Scalars.rotationFactor": {"type": "float", "range": (-1, 1), "cma": (-0.5, 0.5)},
    "Scalars.stepFactor": {"type": "float", "range": (0.02, 2), "cma": (0, 1)},
    "Scalars.positionCoupled": {"type": "bool"},
    "Scalars.scaleProgression": {
        "type": "cat",
        "options": ["linear", "exponential", "additive", "fibonacci", "golden", "sine"],
    },
    "Scalars.rotationProgression": {
        "type": "cat",
        "options": ["linear", "golden-angle", "fibonacci", "sine"],
    },
    "Scalars.alphaProgression": {
        "type": "cat",
        "options": ["exponential", "linear", "inverse"],
    },
    "Scalars.positionProgression": {"type": "cat", "options": ["index", "scale"]},
    # Spatial
    "Spatial.xStep": {"type": "float", "range": (-2, 2), "cma": (-3, 3)},
    "Spatial.yStep": {"type": "float", "range": (-2, 2), "cma": (-3, 3)},
    "Spatial.origin": {
        "type": "cat",
        "options": [
            "center",
            "top-center",
            "bottom-center",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ],
    },
    # Scene (fixed to center - layers handle spatial variation)
    "Scene.rotation": {"type": "float", "fixed": 0},
    "Scene.position": {"type": "obj", "fixed": {"x": 0, "y": 0}},
    "Scene.debug": {"type": "bool", "fixed": False},
    "Scene.transform": {"type": "bool", "fixed": False},
    # Element
    "Element.geometry": {
        "type": "cat",
        "options": [
            "ring",
            "bar",
            "line",
            "arch",
            "u",
            "spiral",
            "wave",
            "infinity",
            "square",
            "roundedRect",
        ],
    },
    "Element.color": {"type": "color"},
    # Noise effect
    "Noise.enabled": {"type": "bool", "default": False},
    "Noise.density": {"type": "float", "range": (0, 1), "default": 0.11},
    "Noise.opacity": {"type": "float", "range": (0, 1), "default": 0.55},
    "Noise.size": {"type": "float", "range": (0.1, 10), "default": 1.0},
}

# Layer schema (instantiated with prefix "Groups.g{i}.g{i}-")
LAYER_SCHEMA = {
    "rotation": {"type": "float", "range": (-3.14159, 3.14159), "cma": (-1, 1)},
    "position": {
        "type": "obj",
        "axes": {"x": (-2, 2), "y": (-2, 2)},
        "cma": {"x": (-1, 1), "y": (-1, 1)},
    },
    "scale": {
        "type": "obj",
        "axes": {"x": (-2, 2), "y": (-2, 2)},
        "cma": {"x": (0.5, 2), "y": (0.5, 2)},
    },
    # Optional (30% chance)
    "stepFactor": {"type": "float", "range": (0, 2), "optional": 0.3},
    "alphaFactor": {"type": "float", "range": (0, 1), "optional": 0.3},
    "scaleFactor": {"type": "float", "range": (0, 2), "optional": 0.3},
    "rotationFactor": {"type": "float", "range": (-1, 1), "optional": 0.3},
    "color": {"type": "color", "optional": 0.2},
    "geometry": {
        "type": "cat",
        "options": SCHEMA["Element.geometry"]["options"],
        "optional": 0.4,
    },
}


# ============================================================================
# 2. DERIVED STRUCTURES (computed from schema)
# ============================================================================


def _build_cma_dims():
    """Build CMA dimension list from schema."""
    dims = []
    for name, spec in SCHEMA.items():
        if not spec.get("cma"):
            continue
        cma = spec["cma"]
        if spec["type"] == "obj":
            for axis, bounds in (
                cma if isinstance(cma, dict) else spec["axes"]
            ).items():
                lo, hi = bounds if isinstance(cma, dict) else spec["axes"][axis]
                dims.append((f"{name}.{axis}", lo, hi))
        else:
            lo, hi = cma if isinstance(cma, tuple) else spec["range"]
            dims.append((name, lo, hi))

    # Add layer 0 dims
    for suffix, spec in LAYER_SCHEMA.items():
        if spec.get("optional") or not spec.get("cma"):
            continue
        cma = spec["cma"]
        pre = "Groups.g0.g0-"
        if spec["type"] == "obj":
            for axis, bounds in (
                cma if isinstance(cma, dict) else spec["axes"]
            ).items():
                lo, hi = bounds if isinstance(cma, dict) else spec["axes"][axis]
                dims.append((f"{pre}{suffix}.{axis}", lo, hi))
        else:
            lo, hi = cma if isinstance(cma, tuple) else spec["range"]
            dims.append((f"{pre}{suffix}", lo, hi))

    return dims


CMA_DIMS = _build_cma_dims()

# Optional suffixes for encoding
OPT_SUFFIXES = {f"-{k}" for k, v in LAYER_SCHEMA.items() if v.get("optional")}


# ============================================================================
# 3. REFERENCE LOADING
# ============================================================================


def load_refs():
    """Load manual reference params from refs.jsonl."""
    # Try both relative paths (running from art-explorer/ or repo root)
    for p in [
        Path("references/refs.jsonl"),
        Path("art-explorer/references/refs.jsonl"),
    ]:
        if p.exists():
            lines = p.read_text().strip().split("\n")
            return [json.loads(ln) for ln in lines if ln]
    return []  # Empty list, not [{}]


def load_best_from_history(min_score=6):
    """Load high-scoring params from ratings.jsonl (self-improvement)."""
    p = Path("art_data/ratings.jsonl")
    if not p.exists():
        return []
    best = []
    for ln in p.read_text().strip().split("\n"):
        if not ln:
            continue
        try:
            r = json.loads(ln)
            if r.get("score", 0) >= min_score:
                best.append(r["params"])
        except ValueError:
            pass
    return best


def load_ref():
    """Pick a random seed from refs + history."""
    refs = load_refs()
    historical = load_best_from_history(min_score=6)
    pool = refs + historical
    return random.choice(pool) if pool else {}


def save_to_refs(params):
    """Append params to refs.jsonl for future runs."""
    with open("references/refs.jsonl", "a") as f:
        f.write(json.dumps(params) + "\n")


# ============================================================================
# 4. RANDOM PARAM GENERATION
# ============================================================================


def _random_value(spec):
    """Generate random value from spec."""
    t = spec["type"]
    if t == "int":
        lo, hi = spec["range"]
        # bias_min: 80% chance to sample from [bias_min, hi], 20% from [lo, bias_min)
        if "bias_min" in spec and random.random() < 0.8:
            return random.randint(spec["bias_min"], hi)
        return random.randint(lo, hi)
    if t == "float":
        return round(random.uniform(*spec["range"]), 4)
    if t == "bool":
        return random.choice([True, False])
    if t == "cat":
        return random.choice(spec["options"])
    if t == "color":
        return random.choice(COLORS)
    if t == "obj":
        return {k: round(random.uniform(*v), 4) for k, v in spec["axes"].items()}
    return None


def random_layer(i):
    """Generate random params for layer i."""
    pre = f"Groups.g{i}.g{i}-"
    p = {}

    for name, spec in LAYER_SCHEMA.items():
        prob = spec.get("optional", 1.0)
        if random.random() < prob:
            val = _random_value(spec)
            if name == "position" and isinstance(val, dict):
                # Clamp to visible range
                val = {"x": max(-1, min(1, val["x"])), "y": max(-1, min(1, val["y"]))}
            p[f"{pre}{name}"] = val

    return p


def random_params(n_layers=None):
    """Generate fully random params."""
    import math

    params = {}

    for name, spec in SCHEMA.items():
        if "fixed" in spec:
            params[name] = spec["fixed"]
        else:
            params[name] = _random_value(spec)

    # Prevent exponential blowout: scaleFactor^repetitions < 1000
    sf = params.get("Scalars.scaleFactor", 1)
    reps = params.get("Scalars.repetitions", 65)
    if sf > 1.01:
        max_reps = int(6.9 / math.log(sf))  # log(1000) ≈ 6.9
        params["Scalars.repetitions"] = min(reps, max(10, max_reps))

    # Layers: 1-5 (weighted toward fewer)
    n = (
        n_layers
        if n_layers is not None
        else random.choices(range(1, 6), weights=[4, 3, 2, 1, 1])[0]
    )

    for i in range(n):
        params.update(random_layer(i))

    return params


# ============================================================================
# 5. URL ENCODING
# ============================================================================


def zigzag(n):
    """Zigzag encoding: maps signed → unsigned (0, -1, 1, -2, 2 → 0, 1, 2, 3, 4)."""
    return (n << 1) ^ (n >> 31)


def encode_params(params):
    """Encode params dict to URL-safe string."""
    buf = []

    def write_uvar(v):
        v = int(v)
        while v >= 128:
            buf.append((v & 127) | 128)
            v >>= 7
        buf.append(v & 127)

    for key, val in params.items():
        kb = key.encode()
        buf.append(len(kb))
        buf.extend(kb)

        opt = 8 if any(key.endswith(s) for s in OPT_SUFFIXES) else 0

        if isinstance(val, bool):
            buf.extend([1 | opt, 1 if val else 0])
        elif isinstance(val, (int, float)):
            buf.append(0 | opt)
            write_uvar(zigzag(int(round(val * 1000))))
        elif isinstance(val, str):
            buf.append(2 | opt)
            sb = val.encode()
            write_uvar(len(sb))
            buf.extend(sb)
        elif isinstance(val, dict):
            buf.append(3 | opt)
            ob = json.dumps(val).encode()
            write_uvar(len(ob))
            buf.extend(ob)

    out = ""
    for i in range(0, len(buf), 3):
        c = buf[i] << 16
        c |= buf[i + 1] << 8 if i + 1 < len(buf) else 0
        c |= buf[i + 2] if i + 2 < len(buf) else 0
        out += CHARS[(c >> 18) & 63] + CHARS[(c >> 12) & 63]
        if i + 1 < len(buf):
            out += CHARS[(c >> 6) & 63]
        if i + 2 < len(buf):
            out += CHARS[c & 63]

    return out


def params_to_url(params):
    return f"http://localhost:3000/render?c={encode_params(params)}"


# ============================================================================
# 6. CMA-ES VECTOR CONVERSION
# ============================================================================


def params_to_vec(params):
    """Convert params dict → normalized [0,1] vector for CMA-ES."""
    vec = []
    for name, lo, hi in CMA_DIMS:
        if name.endswith(".x") or name.endswith(".y"):
            obj, axis = name.rsplit(".", 1)
            v = params.get(obj, {}).get(axis, (lo + hi) / 2)
        else:
            v = params.get(name, (lo + hi) / 2)
        normalized = (v - lo) / (hi - lo) if hi > lo else 0.5
        vec.append(clamp(normalized, 0, 1))
    return vec


def vec_to_params(vec, explore=True):
    """Convert [0,1] vector → params dict, with optional exploration."""
    ref = load_ref()
    params = ref.copy()

    # Apply CMA-ES vector
    for i, (name, lo, hi) in enumerate(CMA_DIMS):
        v = lo + clamp(vec[i], 0, 1) * (hi - lo)
        if name.endswith(".x") or name.endswith(".y"):
            obj, axis = name.rsplit(".", 1)
            params.setdefault(obj, {})[axis] = round(v, 4)
        else:
            params[name] = int(round(v)) if "repetitions" in name else round(v, 4)

    # Random exploration
    if explore:
        for name, spec in SCHEMA.items():
            if spec["type"] == "cat" and random.random() < 0.5:
                params[name] = random.choice(spec["options"])
            elif (
                spec["type"] == "bool" and "fixed" not in spec and random.random() < 0.3
            ):
                params[name] = random.choice([True, False])

        if random.random() < 0.3:
            params["Element.color"] = random.choice(COLORS)

        if random.random() < 0.5:
            for i in range(random.randint(1, 3)):
                params.update(random_layer(i))

    params["Element.color"] = ref.get("Element.color", "#FFFDDD")
    params["Scene.debug"] = False
    params["Scene.transform"] = False

    return params


# ============================================================================
# 7. DATA PERSISTENCE
# ============================================================================

DATA_DIR = Path("art_data")
RATINGS_FILE = DATA_DIR / "ratings.jsonl"


def init_data_dir():
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "screenshots").mkdir(exist_ok=True)


def save_rating(params, score, screenshot, url):
    with open(RATINGS_FILE, "a") as f:
        f.write(
            json.dumps(
                {
                    "params": params,
                    "score": score,
                    "screenshot": screenshot,
                    "url": url,
                    "ts": datetime.now().isoformat(),
                }
            )
            + "\n"
        )


def clear_data():
    RATINGS_FILE.unlink(missing_ok=True)
    for f in (DATA_DIR / "screenshots").glob("*.png"):
        f.unlink()


# ============================================================================
# 8. PREVIEW WINDOW
# ============================================================================


def setup_preview():
    import os

    import matplotlib

    try:
        matplotlib.use("QtAgg")
    except Exception:
        pass

    import matplotlib.pyplot as plt

    state = {"key": None}

    def on_close(_):
        print("\nWindow closed.")
        os._exit(0)

    plt.ion()
    plt.rcParams.update({"font.family": "monospace", "toolbar": "None"})

    fig = plt.figure(figsize=(6.5, 4), facecolor="#1a1a1a")
    ax_img = fig.add_axes([0, 0, 0.55, 1])
    ax_info = fig.add_axes([0.56, 0, 0.44, 1])

    for ax in (ax_img, ax_info):
        ax.axis("off")
    ax_info.set_facecolor("#1a1a1a")

    fig.canvas.manager.set_window_title("Art Explorer  [y=save  n=skip]")
    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", lambda e: state.update(key=e.key))

    return fig, ax_img, ax_info, state


def update_preview(fig, ax_img, ax_info, shot_path, params, gen, j, pop, gens):
    from PIL import Image

    ax_img.clear()
    ax_img.imshow(Image.open(shot_path), aspect="auto")
    ax_img.axis("off")

    ax_info.clear()
    ax_info.axis("off")
    ax_info.set_facecolor("#1a1a1a")

    layers = []
    for i in range(4):
        if f"Groups.g{i}.g{i}-position" in params:
            geo = params.get(f"Groups.g{i}.g{i}-geometry", "·")
            layers.append(geo[:3] if geo != "·" else "·")
    layers_str = ",".join(layers) if layers else "none"

    info = (
        f"GEN {gen + 1}/{gens}  |  {j + 1}/{pop}\n{'─' * 20}\n\n"
        f"shape     {params.get('Element.geometry', '?')}\n"
        f"reps      {params.get('Scalars.repetitions', '?')}\n"
        f"scaleFac  {params.get('Scalars.scaleFactor', '?')}\n"
        f"rotFac    {params.get('Scalars.rotationFactor', '?')}\n"
        f"origin    {params.get('Spatial.origin', '?')}\n"
        f"layers    {layers_str}\n\n"
        f"{'─' * 20}\n\n[y] save  [n] skip"
    )

    ax_info.text(
        0.05,
        0.95,
        info,
        transform=ax_info.transAxes,
        fontsize=10,
        color="#666",
        va="top",
        family="monospace",
    )

    fig.canvas.draw()
    fig.canvas.flush_events()

    return info


# ============================================================================
# 9. MAIN OPTIMIZATION LOOP
# ============================================================================


async def run_optimization(gens=50, pop=8, preview=False):
    import cma
    from playwright.async_api import async_playwright
    from score import score_image

    init_data_dir()

    fig, ax_img, ax_info, state = (None, None, None, {"key": None})
    if preview:
        fig, ax_img, ax_info, state = setup_preview()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1200, "height": 900})

        refs = load_refs()
        historical = load_best_from_history(min_score=6)
        print(f"\n{'=' * 50}")
        print(f"CMA-ES: {len(CMA_DIMS)} dims, pop={pop}, gens={gens}")
        print(f"Refs: {len(refs)} manual + {len(historical)} from history (score>=6)")
        print(f"{'=' * 50}\n")

        es = cma.CMAEvolutionStrategy(
            params_to_vec(load_ref()),
            0.08,
            {
                "popsize": pop,
                "bounds": [0, 1],
                "verbose": -9,
                "tolx": 1e-12,
                "tolfun": 1e-12,
            },
        )

        best = (0, None)

        for g in range(gens):
            solutions = es.ask()
            scores = []

            for j, sol in enumerate(solutions):
                params = vec_to_params(sol)
                url = params_to_url(params)

                await page.goto(url)
                await page.wait_for_timeout(2000)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                shot_path = DATA_DIR / f"screenshots/{ts}_g{g}_{j}.png"
                await page.screenshot(path=str(shot_path))

                score = None
                info = ""

                if fig:
                    import time

                    state["key"] = None
                    info = update_preview(
                        fig, ax_img, ax_info, shot_path, params, g, j, pop, gens
                    )

                    for _ in range(3):
                        fig.canvas.flush_events()
                        if state["key"]:
                            break
                        time.sleep(0.03)

                    if state["key"] == "n":
                        score = 1
                        ax_info.texts[-1].set_text(
                            info.replace("[y] save  [n] skip", "SKIPPED")
                        )
                        fig.canvas.draw()
                        print(f"[g{g}:{j}] skipped")
                    elif state["key"] == "y":
                        save_to_refs(params)
                        print(f"[g{g}:{j}] *SAVED*")

                if score is None:
                    import threading
                    import time

                    if fig:
                        ax_info.texts[-1].set_text(
                            info.replace("[y] save  [n] skip", "scoring... [n]=abort")
                        )
                        ax_info.texts[-1].set_color("#888")
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        state["key"] = None

                    result = [None]

                    def do_score():
                        result[0] = score_image(str(shot_path))

                    t = threading.Thread(target=do_score, daemon=True)
                    t.start()

                    while t.is_alive():
                        if fig:
                            fig.canvas.flush_events()
                            if state["key"] == "n":
                                score = 1
                                ax_info.texts[-1].set_text(
                                    info.replace("[y] save  [n] skip", "ABORTED")
                                )
                                fig.canvas.draw()
                                print(f"[g{g}:{j}] aborted")
                                break
                        time.sleep(0.05)

                    if score is None:
                        score = result[0][0] if result[0] else 1

                scores.append(score)
                save_rating(params, score, str(shot_path.relative_to(DATA_DIR)), url)

                if fig and state["key"] != "n":
                    color = "#fffddd" if score >= 5 else "#ff6b6b"
                    ax_info.texts[-1].set_text(
                        info.replace("[y] save  [n] skip", f"SCORE  {score}/10")
                    )
                    ax_info.texts[-1].set_color(color)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                if score >= 9 and state.get("key") != "y":
                    save_to_refs(params)
                    print(f"[g{g}:{j}] {score}/10 *AUTO-SAVED*")
                elif state.get("key") not in ("y", "n"):
                    if score > best[0]:
                        best = (score, params)
                        print(f"[g{g}:{j}] {score}/10 *BEST*")
                    else:
                        print(f"[g{g}:{j}] {score}/10")

                if score > best[0]:
                    best = (score, params)

            es.tell(solutions, [-s for s in scores])
            avg = sum(scores) / len(scores)
            print(f"--- Gen {g + 1}: avg={avg:.1f}, best={best[0]} ---\n")

        await browser.close()

    print(f"\nDone: {gens * pop} evals, best={best[0]}/10")
    if best[1]:
        print(f"URL: {params_to_url(best[1])}")


# ============================================================================
# 10. CLI
# ============================================================================


def main():
    p = argparse.ArgumentParser(
        description="Generative art explorer with CMA-ES + VLM scoring"
    )
    p.add_argument("--gens", type=int, default=50, help="Number of generations")
    p.add_argument("--pop", type=int, default=8, help="Population size per generation")
    p.add_argument(
        "--preview", action="store_true", help="Show matplotlib preview window"
    )
    p.add_argument(
        "--clean", action="store_true", help="Clear ratings/screenshots before run"
    )
    p.add_argument(
        "--test", action="store_true", help="Print random params + URL, then exit"
    )
    args = p.parse_args()

    if args.test:
        params = random_params()
        print("Params:", json.dumps(params, indent=2))
        print(f"\n{params_to_url(params)}")
        return

    if args.clean:
        clear_data()

    asyncio.run(run_optimization(args.gens, args.pop, args.preview))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
