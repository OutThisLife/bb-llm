"""
Generative art parameter explorer with VLM feedback.
CMA-ES optimization against Qwen2-VL scoring.
"""

import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path


# === Config ===

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

PARAMS = {
    "Scalars.repetitions": (1, 500, 1),
    "Scalars.alphaFactor": (0, 1, 0.01),
    "Scalars.scaleFactor": (0, 2, 0.01),
    "Scalars.rotationFactor": (-1, 1, 0.01),
    "Scalars.stepFactor": (0, 2, 0.01),
    "Spatial.xStep": (-2, 2, 0.01),
    "Spatial.yStep": (-2, 2, 0.01),
    "Scene.rotation": (-3.14159, 3.14159, 0.01),
}

OBJECT_PARAMS = {
    "Scene.position": {"x": (-2, 2), "y": (-2, 2)},
}

LAYER_PARAMS = {"rotation": (-3.14159, 3.14159, 0.01)}
LAYER_OBJECT_PARAMS = {
    "position": {"x": (-2, 2), "y": (-2, 2)},
    "scale": {"x": (-2, 2), "y": (-2, 2)},
}
LAYER_OPTIONAL = {
    "stepFactor": (0, 2, 0.01),
    "alphaFactor": (0, 1, 0.01),
    "scaleFactor": (0, 2, 0.01),
    "rotationFactor": (-1, 1, 0.01),
}

CATEGORICAL = {
    "Element.geometry": [
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
    "Scalars.scaleProgression": [
        "linear",
        "exponential",
        "additive",
        "fibonacci",
        "golden",
        "sine",
    ],
    "Scalars.rotationProgression": ["linear", "golden-angle", "fibonacci", "sine"],
    "Scalars.alphaProgression": ["exponential", "linear", "inverse"],
    "Scalars.positionProgression": ["index", "scale"],
    "Spatial.origin": [
        "center",
        "top-center",
        "bottom-center",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ],
}

BOOLEANS = ["Scalars.positionCoupled"]

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

CMA_DIMS = [
    ("Scalars.repetitions", 50, 500),
    ("Scalars.alphaFactor", 0, 1),
    ("Scalars.scaleFactor", 0.5, 3),
    ("Scalars.rotationFactor", -0.5, 0.5),
    ("Scalars.stepFactor", 0, 1),
    ("Spatial.xStep", -3, 3),
    ("Spatial.yStep", -3, 3),
    ("Scene.rotation", -1, 1),
    ("Scene.position.x", -1, 1),
    ("Scene.position.y", -1, 1),
    ("Groups.g0.g0-rotation", -1, 1),
    ("Groups.g0.g0-position.x", -1, 1),
    ("Groups.g0.g0-position.y", -1, 1),
    ("Groups.g0.g0-scale.x", 0.5, 2),
    ("Groups.g0.g0-scale.y", 0.5, 2),
]


# === References ===


def load_refs() -> list[dict]:
    p = Path("references/refs.jsonl")
    return (
        [json.loads(ln) for ln in p.read_text().strip().split("\n") if ln]
        if p.exists()
        else [{}]
    )


def load_ref() -> dict:
    refs = load_refs()
    return random.choice(refs) if refs else {}


# === Param Generation ===

clamp = lambda v, lo, hi: max(lo, min(hi, v))


GEOMETRIES = CATEGORICAL["Element.geometry"]


def random_layer(i: int) -> dict:
    pre = f"Groups.g{i}.g{i}-"
    p = {}
    for name, (lo, hi, _) in LAYER_PARAMS.items():
        p[f"{pre}{name}"] = round(random.uniform(lo, hi), 2)
    for name, axes in LAYER_OBJECT_PARAMS.items():
        p[f"{pre}{name}"] = {k: round(random.uniform(*v), 2) for k, v in axes.items()}
    for name, (lo, hi, _) in LAYER_OPTIONAL.items():
        if random.random() < 0.3:
            p[f"{pre}{name}"] = round(random.uniform(lo, hi), 2)
    if random.random() < 0.2:
        p[f"{pre}color"] = random.choice(COLORS)
    if random.random() < 0.4:  # 40% chance of geometry override per layer
        p[f"{pre}geometry"] = random.choice(GEOMETRIES)
    return p


def random_params(n_layers=None) -> dict:
    params = {}

    for name, (lo, hi, step) in PARAMS.items():
        v = random.uniform(lo, hi)
        params[name] = int(round(v)) if step == 1 else round(v, 2)

    for name, axes in OBJECT_PARAMS.items():
        params[name] = {
            k: round(random.uniform(lo, hi), 2) for k, (lo, hi) in axes.items()
        }

    for name, opts in CATEGORICAL.items():
        params[name] = random.choice(opts)

    for name in BOOLEANS:
        params[name] = random.choice([True, False])

    params["Element.color"] = random.choice(COLORS)
    params["Scene.debug"] = False
    params["Scene.transform"] = False

    n = (
        n_layers
        if n_layers is not None
        else random.choices(range(5), weights=[4, 3, 2, 1, 1])[0]
    )
    for i in range(n):
        params.update(random_layer(i))

    return params


# === URL Encoding ===


def zigzag(n):
    return (n << 1) ^ (n >> 31)


def encode_params(params: dict) -> str:
    buf = []

    def write_uvar(v):
        v = int(v)
        while v >= 128:
            buf.append((v & 127) | 128)
            v >>= 7
        buf.append(v & 127)

    OPT_SUFFIXES = {
        "-stepFactor",
        "-alphaFactor",
        "-scaleFactor",
        "-rotationFactor",
        "-color",
        "-geometry",
    }

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
        c = (
            buf[i] << 16
            | (buf[i + 1] << 8 if i + 1 < len(buf) else 0)
            | (buf[i + 2] if i + 2 < len(buf) else 0)
        )
        out += CHARS[(c >> 18) & 63] + CHARS[(c >> 12) & 63]
        if i + 1 < len(buf):
            out += CHARS[(c >> 6) & 63]
        if i + 2 < len(buf):
            out += CHARS[c & 63]

    return out


def params_to_url(params: dict) -> str:
    return f"http://localhost:3000/render?c={encode_params(params)}"


# === CMA-ES ===


def params_to_vec(params: dict) -> list:
    vec = []
    for name, lo, hi in CMA_DIMS:
        if name.endswith(".x") or name.endswith(".y"):
            obj, axis = name.rsplit(".", 1)
            v = params.get(obj, {}).get(axis, (lo + hi) / 2)
        else:
            v = params.get(name, (lo + hi) / 2)
        vec.append(clamp((v - lo) / (hi - lo) if hi > lo else 0.5, 0, 1))
    return vec


def vec_to_params(vec: list, explore=True) -> dict:
    ref = load_ref()
    params = ref.copy()

    for i, (name, lo, hi) in enumerate(CMA_DIMS):
        v = lo + clamp(vec[i], 0, 1) * (hi - lo)
        if name.endswith(".x") or name.endswith(".y"):
            obj, axis = name.rsplit(".", 1)
            params.setdefault(obj, {})[axis] = round(v, 2)
        else:
            params[name] = (
                int(round(v)) if name == "Scalars.repetitions" else round(v, 2)
            )

    if explore:
        for name, opts in CATEGORICAL.items():
            if random.random() < 0.5:
                params[name] = random.choice(opts)
        for name in BOOLEANS:
            if random.random() < 0.3:
                params[name] = random.choice([True, False])
        if random.random() < 0.3:
            params["Element.color"] = random.choice(COLORS)
        # Layers with geometry overrides help achieve complex Figma-like compositions
        if random.random() < 0.5:  # 50% chance to add layers
            for i in range(random.randint(1, 3)):  # 1-3 layers
                params.update(random_layer(i))

    params["Element.color"] = ref.get("Element.color", "#FFFDDD")
    params["Scene.debug"] = False
    params["Scene.transform"] = False

    return params


# === Session ===


class Session:
    def __init__(self, data_dir="art_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "screenshots").mkdir(exist_ok=True)
        self.ratings_file = self.data_dir / "ratings.jsonl"

    def save(self, params, score, screenshot, url):
        with open(self.ratings_file, "a") as f:
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

    def clear(self):
        self.ratings_file.unlink(missing_ok=True)
        for f in (self.data_dir / "screenshots").glob("*.png"):
            f.unlink()


# === Runner ===


async def run_cma(session: Session, gens=50, pop=8, preview=False):
    import cma
    from playwright.async_api import async_playwright
    from score import score_image

    fig, ax_img, ax_info, state = None, None, None, {"key": None}

    if preview:
        import matplotlib
        import os

        try:
            matplotlib.use("QtAgg")
        except:
            pass
        import matplotlib.pyplot as plt

        def on_close(_):
            print("\nWindow closed.")
            os._exit(0)  # immediate exit, no cleanup callbacks

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

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1200, "height": 900})

        print(
            f"\n{'='*50}\nCMA-ES: {len(CMA_DIMS)} dims, pop={pop}, gens={gens}\n{'='*50}\n"
        )

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
                shot_path = session.data_dir / f"screenshots/{ts}_g{g}_{j}.png"
                await page.screenshot(path=str(shot_path))

                # Preview + hotkey check
                score = None
                if fig:
                    from PIL import Image
                    import time

                    state["key"] = None
                    ax_img.clear()
                    ax_img.imshow(Image.open(shot_path), aspect="auto")
                    ax_img.axis("off")

                    ax_info.clear()
                    ax_info.axis("off")
                    ax_info.set_facecolor("#1a1a1a")

                    # Count layers and their geometries
                    layers = []
                    for i in range(4):
                        key = f"Groups.g{i}.g{i}-position"
                        if key in params:
                            geo = params.get(f"Groups.g{i}.g{i}-geometry", "·")
                            layers.append(geo[:3] if geo != "·" else "·")
                    layers_str = ",".join(layers) if layers else "none"

                    info = (
                        f"GEN {g+1}/{gens}  |  {j+1}/{pop}\n{'─'*20}\n\n"
                        f"shape     {params.get('Element.geometry', '?')}\n"
                        f"reps      {params.get('Scalars.repetitions', '?')}\n"
                        f"scaleFac  {params.get('Scalars.scaleFactor', '?')}\n"
                        f"rotFac    {params.get('Scalars.rotationFactor', '?')}\n"
                        f"origin    {params.get('Spatial.origin', '?')}\n"
                        f"layers    {layers_str}\n\n"
                        f"{'─'*20}\n\n[y] save  [n] skip"
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

                    # Quick check for skip before scoring
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
                        with open("references/refs.jsonl", "a") as f:
                            f.write(json.dumps(params) + "\n")
                        print(f"[g{g}:{j}] *SAVED*")

                # Score if not skipped - run in thread so we can abort
                if score is None:
                    import threading

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
                        if result[0]:
                            score, _ = result[0]
                        else:
                            score = 1  # thread crashed
                            print(f"[g{g}:{j}] scoring failed")

                scores.append(score)
                session.save(
                    params, score, str(shot_path.relative_to(session.data_dir)), url
                )

                if fig and state["key"] != "n":
                    ax_info.texts[-1].set_text(
                        info.replace(
                            "[y] save  [n] skip", f"SCORE  {score}/10"
                        ).replace("scoring...", f"SCORE  {score}/10")
                    )
                    ax_info.texts[-1].set_color("#fffddd" if score >= 5 else "#ff6b6b")
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                if state.get("key") not in ("y", "n"):
                    if score > best[0]:
                        best = (score, params)
                        print(f"[g{g}:{j}] {score}/10 *BEST*")
                    else:
                        print(f"[g{g}:{j}] {score}/10")
                elif score > best[0]:
                    best = (score, params)

            es.tell(solutions, [-s for s in scores])
            print(
                f"--- Gen {g+1}: avg={sum(scores)/len(scores):.1f}, best={best[0]} ---\n"
            )

        await browser.close()
        print(f"\nDone: {gens*pop} evals, best={best[0]}/10")
        if best[1]:
            print(f"URL: {params_to_url(best[1])}")


# === CLI ===


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--preview", action="store_true")
    p.add_argument(
        "--clean", action="store_true", help="Clear ratings/screenshots before run"
    )
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if args.test:
        params = random_params()
        print("Params:", json.dumps(params, indent=2))
        print(f"\n{params_to_url(params)}")
        return

    session = Session()
    if args.clean:
        session.clear()
    asyncio.run(run_cma(session, args.gens, args.pop, args.preview))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
