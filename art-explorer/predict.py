"""
Predict: Inverse inference + gradient/CMA-ES refinement + Explore mode
=====================================================================
Inverse mode: target image → params (CNN + gradient descent + CMA-ES polish)
Explore mode: sample from trained param prior → taste rerank → render top N
"""

import argparse
import io
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cma
import lpips
import numpy as np
import requests
import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")
from model import ForwardModel, InverseModel, MAX_LAYERS, ParamVAE, TasteModel
from PIL import Image
from requests.adapters import HTTPAdapter
from torchvision import transforms
from urllib3.util.retry import Retry

from utils import (
    BOOLEAN_KEYS,
    CATEGORICAL_KEYS,
    CONTINUOUS_KEYS,
    LAYER_CONTINUOUS_KEYS,
    LAYER_OPTIONALS,
    TASTE_FEATURE_DIM,
    decode_params,
    decode_taste_features,
    encode_taste_features,
    encode_params,
    get_device,
    normalize_continuous,
    normalize_layer,
    get_layer_geometry_idx,
    get_layer_presence,
    load_state_dict_compat,
    reconstruct_params,
    to_prefixed,
    to_scene_params,
)

# ============================================================================
# Constants
# ============================================================================

IMG_SIZE = 252  # divisible by DINOv2 patch size (14)
FWD_SIZE = 256  # forward model output size
DEFAULT_ENDPOINT = "http://localhost:3000/api/raster"
MAX_WORKERS = min(os.cpu_count() or 4, 12)
OUTPUT_DIR = Path("output")

# ============================================================================
# Helpers
# ============================================================================

_session_local = threading.local()


def _get_session():
    if not hasattr(_session_local, "session"):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5)
        session.mount("http://", HTTPAdapter(max_retries=retry))
        _session_local.session = session
    return _session_local.session


def render_api(params, endpoint=DEFAULT_ENDPOINT):
    try:
        session = _get_session()
        api_params = to_prefixed(params) if "layers" in params else params
        resp = session.post(endpoint, json=api_params, timeout=60)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def render_batch(params_list, endpoint=DEFAULT_ENDPOINT):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(render_api, p, endpoint) for p in params_list]
        return [f.result() for f in futures]


def load_image(path, size=IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(path).convert("RGB"))


def to_lpips_tensor(img, size=IMG_SIZE):
    arr = np.array(img.resize((size, size))).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2 - 1


def score_taste(params_list, taste_model, device):
    if taste_model is None or not params_list:
        return [0.0] * len(params_list)
    feats = torch.tensor(
        [encode_taste_features(p) for p in params_list], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        logits = taste_model(feats)
        probs = torch.sigmoid(logits)
        if probs.dim() == 0:
            return [probs.item()]
        return probs.cpu().tolist()


# ============================================================================
# Param vector ↔ forward model inputs
# ============================================================================


def _vec_to_forward_inputs(vec, categoricals, layer_geos, layer_presence, n_layers, device):
    n_cont = len(CONTINUOUS_KEYS)
    n_bool = len(BOOLEAN_KEYS)
    n_lcont = len(LAYER_CONTINUOUS_KEYS)

    cont = vec[:n_cont].unsqueeze(0)
    bools = vec[n_cont:n_cont + n_bool].unsqueeze(0)

    cat_idx = {}
    for k, v in categoricals.items():
        opts = CATEGORICAL_KEYS[k]
        idx = opts.index(v) if v in opts else 0
        cat_idx[k] = torch.tensor([idx], dtype=torch.long, device=device)

    lc = torch.tensor([n_layers], dtype=torch.long, device=device)

    lt, lp, lg = [], [], []
    offset = n_cont + n_bool
    for i in range(MAX_LAYERS):
        if i < n_layers:
            start = offset + i * n_lcont
            lt.append(vec[start:start + n_lcont].unsqueeze(0))
            pres = layer_presence[i] if i < len(layer_presence) else [False] * len(LAYER_OPTIONALS)
            lp.append(torch.tensor([pres], dtype=torch.float32, device=device))
            geo = layer_geos[i] if i < len(layer_geos) else None
            geo_opts = CATEGORICAL_KEYS["geometry"]
            geo_idx = geo_opts.index(geo) if geo and geo in geo_opts else 0
            lg.append(torch.tensor([geo_idx], dtype=torch.long, device=device))
        else:
            lt.append(torch.full((1, n_lcont), 0.5, device=device))
            lp.append(torch.zeros(1, len(LAYER_OPTIONALS), device=device))
            lg.append(torch.tensor([0], dtype=torch.long, device=device))

    return cont, cat_idx, bools, lc, lt, lp, lg


def _params_to_forward_batch(flat_params_list, device):
    batch_cont, batch_bool, batch_lc = [], [], []
    batch_cat = {k: [] for k in CATEGORICAL_KEYS}
    batch_lt = [[] for _ in range(MAX_LAYERS)]
    batch_lp = [[] for _ in range(MAX_LAYERS)]
    batch_lg = [[] for _ in range(MAX_LAYERS)]

    for params in flat_params_list:
        batch_cont.append(torch.tensor(normalize_continuous(params), dtype=torch.float32))
        for k, opts in CATEGORICAL_KEYS.items():
            val = params.get(k, opts[0])
            batch_cat[k].append(opts.index(val) if val in opts else 0)
        batch_bool.append(torch.tensor(
            [1.0 if params.get(b, False) else 0.0 for b in BOOLEAN_KEYS], dtype=torch.float32,
        ))
        layers = params.get("layers", [])
        batch_lc.append(min(len(layers), MAX_LAYERS))
        for i in range(MAX_LAYERS):
            if i < len(layers):
                batch_lt[i].append(torch.tensor(normalize_layer(layers[i]), dtype=torch.float32))
                batch_lp[i].append(torch.tensor(get_layer_presence(layers[i]), dtype=torch.float32))
                batch_lg[i].append(get_layer_geometry_idx(layers[i]))
            else:
                batch_lt[i].append(torch.tensor([0.5] * len(LAYER_CONTINUOUS_KEYS), dtype=torch.float32))
                batch_lp[i].append(torch.zeros(len(LAYER_OPTIONALS), dtype=torch.float32))
                batch_lg[i].append(0)

    return (
        torch.stack(batch_cont).to(device),
        {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in batch_cat.items()},
        torch.stack(batch_bool).to(device),
        torch.tensor(batch_lc, dtype=torch.long).to(device),
        [torch.stack(batch_lt[i]).to(device) for i in range(MAX_LAYERS)],
        [torch.stack(batch_lp[i]).to(device) for i in range(MAX_LAYERS)],
        [torch.tensor(batch_lg[i], dtype=torch.long).to(device) for i in range(MAX_LAYERS)],
    )


# ============================================================================
# Gradient-based optimizer (through differentiable forward model)
# ============================================================================


def optimize_gradient(
    forward_model, lpips_model, target_tensor, device,
    categoricals, layer_geos, layer_presence, n_layers,
    x0, steps=300, lr=0.01,
):
    vec = nn.Parameter(torch.tensor(x0, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([vec], lr=lr)

    best_loss, best_vec = float("inf"), x0.copy()

    for step in range(steps):
        opt.zero_grad()
        clamped = vec.clamp(0, 1)
        inputs = _vec_to_forward_inputs(
            clamped, categoricals, layer_geos, layer_presence, n_layers, device,
        )
        img = forward_model(*inputs)
        loss = lpips_model(img * 2 - 1, target_tensor).mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            l = loss.item()
            if l < best_loss:
                best_loss = l
                best_vec = vec.clamp(0, 1).detach().cpu().numpy().copy()

    return best_vec, best_loss


# ============================================================================
# Scorers (for inverse mode)
# ============================================================================


class ForwardScorer:
    def __init__(self, target_path, device, forward_model):
        self.device = device
        self.forward_model = forward_model
        self.lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
        self.lpips_model.eval()
        target = Image.open(target_path).convert("RGB").resize((FWD_SIZE, FWD_SIZE))
        self.target_tensor = to_lpips_tensor(target, FWD_SIZE).unsqueeze(0).to(device)

    def score_params(self, flat_params_list):
        if not flat_params_list:
            return []
        inputs = _params_to_forward_batch(flat_params_list, self.device)
        with torch.no_grad():
            pred_img = self.forward_model(*inputs)
            pred_lpips = pred_img * 2 - 1
            target_exp = self.target_tensor.expand(len(flat_params_list), -1, -1, -1)
            dists = self.lpips_model(target_exp, pred_lpips)
            result = dists.squeeze(-1).squeeze(-1).squeeze(-1)
            if result.dim() == 0:
                return [result.item()]
            return result.tolist()


class APIScorer:
    def __init__(self, target_path, device, endpoint=DEFAULT_ENDPOINT):
        self.device = device
        self.endpoint = endpoint
        self.lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
        self.lpips_model.eval()
        target = Image.open(target_path).convert("RGB").resize((FWD_SIZE, FWD_SIZE))
        self.target_tensor = to_lpips_tensor(target, FWD_SIZE).unsqueeze(0).to(device)

    def score_params(self, flat_params_list):
        images = render_batch(flat_params_list, self.endpoint)
        result = [1.0] * len(flat_params_list)
        valid = [(i, img) for i, img in enumerate(images) if img is not None]
        if valid:
            indices, imgs = zip(*valid, strict=True)
            with torch.no_grad():
                tensors = torch.stack([to_lpips_tensor(img, FWD_SIZE) for img in imgs]).to(self.device)
                target_exp = self.target_tensor.expand(len(imgs), -1, -1, -1)
                dists = self.lpips_model(target_exp, tensors)
                d = dists.squeeze(-1).squeeze(-1).squeeze(-1)
                scores = [d.item()] if d.dim() == 0 else d.tolist()
            for idx, sc in zip(indices, scores, strict=True):
                result[idx] = sc
        return result


# ============================================================================
# CMA-ES optimizer
# ============================================================================


def optimize_cmaes(
    scorer, categoricals, layer_geos, layer_presence, n_layers,
    x0=None, sigma0=0.3, max_fevals=2000, verbose=True,
):
    ndim = len(CONTINUOUS_KEYS) + len(BOOLEAN_KEYS) + len(LAYER_CONTINUOUS_KEYS) * n_layers
    if x0 is None:
        x0 = np.full(ndim, 0.5)

    opts = {
        "bounds": [0, 1],
        "maxfevals": max_fevals,
        "tolx": 1e-6,
        "tolfun": 1e-6,
        "verbose": -9,
        "CMA_active": True,
    }
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    best_score, best_params = float("inf"), None
    gen = 0

    while not es.stop():
        solutions = es.ask()
        params_list = [
            decode_params(np.array(sol), categoricals, layer_geos, layer_presence, n_layers)
            for sol in solutions
        ]
        scores = scorer.score_params(params_list)
        es.tell(solutions, scores)

        gen_best = min(scores)
        if gen_best < best_score:
            best_score = gen_best
            best_params = params_list[scores.index(gen_best)]

        gen += 1
        if verbose and gen % 10 == 0:
            print(f"  gen {gen:3d} | best={best_score:.4f} | sigma={es.sigma:.4f}")

    if verbose:
        print(f"  Converged: {es.stop()}")

    return best_params, best_score


# ============================================================================
# CNN inference
# ============================================================================


def cnn_predict(target_path, model_path="models/inverse_model.pt"):
    device = get_device()
    model = InverseModel(pretrained=False).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_tensor = load_image(target_path).unsqueeze(0).to(device)

    with torch.no_grad():
        cont, cat, boolean, layer_count, layer_params, layer_pres, layer_geos = model(img_tensor)

    params = reconstruct_params(
        cont[0],
        {k: v[0] for k, v in cat.items()},
        boolean[0],
        layer_count[0],
        [lp[0] for lp in layer_params],
        [lp[0] for lp in layer_pres],
        [lg[0] for lg in layer_geos],
    )

    categoricals = {k: params[k] for k in CATEGORICAL_KEYS}
    n_layers = len(params.get("layers", []))
    layer_geo_list = [layer.get("geometry") for layer in params.get("layers", [])]
    layer_pres_list = [
        [k in layer for k in LAYER_OPTIONALS] for layer in params.get("layers", [])
    ]

    x0 = encode_params(params, n_layers)
    return x0, categoricals, layer_geo_list, layer_pres_list, n_layers, params


# ============================================================================
# Inverse mode
# ============================================================================


def inverse_mode(
    target_path,
    model_path="models/inverse_model.pt",
    forward_path="models/forward_model.pt",
    endpoint=DEFAULT_ENDPOINT,
    max_fevals=2000,
):
    """CNN predict → config screening → gradient descent → CMA-ES polish."""
    print(f"\nTarget: {target_path}")
    device = get_device()
    print(f"Device: {device}")

    forward_model = None
    if Path(forward_path).exists():
        forward_model = ForwardModel().to(device)
        ckpt = torch.load(forward_path, map_location=device, weights_only=False)
        forward_model.load_state_dict(load_state_dict_compat(ckpt["model_state_dict"]))
        forward_model.eval()
        if device.type == "cuda":
            forward_model = torch.compile(forward_model)
        print(f"Forward model: {forward_path}")

    api_scorer = APIScorer(target_path, device, endpoint)

    best_params, best_score = None, float("inf")
    best_cats, best_geos, best_pres, best_nl = None, None, None, 0
    screen_fevals = 300

    if Path(model_path).exists():
        print(f"CNN predict from {model_path}...")
        x0, cats, geos, presence, n_layers, init_params = cnn_predict(target_path, model_path)

        init_img = render_api(init_params, endpoint)
        if init_img is None:
            print("API render failed — is the renderer running?")
            raise SystemExit(1)

        configs = [(cats, geos, presence, n_layers, x0, 0.2, "cnn")]
        for nl in [0, 1, 2, 3]:
            if nl == n_layers:
                continue
            configs.append((cats, [cats["geometry"]] * nl,
                           [[True] * len(LAYER_OPTIONALS)] * nl,
                           nl, None, 0.3, f"alt-{nl}L"))
    else:
        print("No inverse model — screening categoricals")
        configs = []
        for nl in [1, 2, 0]:
            c = {k: random.choice(v) for k, v in CATEGORICAL_KEYS.items()}
            g = [random.choice(CATEGORICAL_KEYS["geometry"]) for _ in range(nl)]
            p = [[True] * len(LAYER_OPTIONALS)] * nl
            configs.append((c, g, p, nl, None, 0.3, f"rand-{nl}L"))

    print(f"\nStage 1: Screening {len(configs)} configs ({screen_fevals} fevals each)...")
    for cats_c, geos_c, pres_c, nl_c, x0_c, sig, label in configs:
        params, score = optimize_cmaes(
            api_scorer, cats_c, geos_c, pres_c, nl_c,
            x0=x0_c, sigma0=sig, max_fevals=screen_fevals, verbose=False,
        )
        print(f"  [{label}] layers={nl_c} → LPIPS={score:.4f}")
        if score < best_score:
            best_score = score
            best_params = params
            best_cats, best_geos, best_pres, best_nl = cats_c, geos_c, pres_c, nl_c

    if best_params is None:
        print("Optimization failed.")
        return

    if forward_model is not None:
        print(f"\nStage 2: Gradient refinement (300 steps)...")
        x0_grad = encode_params(best_params, best_nl)

        refined_vec, grad_score = optimize_gradient(
            forward_model, api_scorer.lpips_model, api_scorer.target_tensor, device,
            best_cats, best_geos, best_pres, best_nl,
            x0_grad, steps=300, lr=0.01,
        )
        print(f"  Gradient LPIPS={grad_score:.4f} (was {best_score:.4f})")
        x0_refine = refined_vec
        sigma_polish = 0.05
    else:
        print(f"\nNo forward model — skipping gradient refinement")
        x0_refine = encode_params(best_params, best_nl)
        sigma_polish = 0.1

    print(f"\nStage 3: CMA-ES polish ({max_fevals} fevals)...")
    best_params, best_score = optimize_cmaes(
        api_scorer, best_cats, best_geos, best_pres, best_nl,
        x0=x0_refine, sigma0=sigma_polish, max_fevals=max_fevals,
    )

    rendered = render_api(best_params, endpoint)
    if rendered is None:
        print("Final API render failed.")
        return

    out_img = OUTPUT_DIR / "images"
    out_params = OUTPUT_DIR / "params"
    out_img.mkdir(parents=True, exist_ok=True)
    out_params.mkdir(parents=True, exist_ok=True)

    stem = Path(target_path).stem
    rendered.save(out_img / f"{stem}_000.png")
    with open(out_params / f"{stem}_000.json", "w") as f:
        json.dump(best_params, f)

    print(f"\nResult: LPIPS={best_score:.4f}")
    print(f"  geo={best_params.get('geometry', '?')}, layers={len(best_params.get('layers', []))}")
    print(f"  Saved: {out_img / f'{stem}_000.png'}")


# ============================================================================
# Explore mode (param prior sampling + taste rerank)
# ============================================================================


def explore_mode(
    n_results=20,
    n_candidates=500,
    endpoint=DEFAULT_ENDPOINT,
    taste_path="models/taste_model.pt",
    prior_path="models/param_prior.pt",
    sample_temp=1.0,
):
    """Sample from trained param prior, rerank by taste, render top N."""
    print("\nExplore mode")

    if not Path(prior_path).exists():
        print(f"No param prior at {prior_path}. Run: make train")
        return

    device = get_device()

    # Load prior
    ckpt = torch.load(prior_path, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 64)
    prior = ParamVAE(in_dim=TASTE_FEATURE_DIM, latent_dim=latent_dim).to(device)
    prior.load_state_dict(load_state_dict_compat(ckpt["model_state_dict"]))
    prior.eval()
    print(f"Prior: {prior_path} (latent={latent_dim})")

    # Load taste model (optional reranker)
    taste_model = None
    if Path(taste_path).exists():
        taste_model = TasteModel(in_dim=TASTE_FEATURE_DIM).to(device)
        tckpt = torch.load(taste_path, map_location=device, weights_only=False)
        taste_model.load_state_dict(load_state_dict_compat(tckpt["model_state_dict"]))
        taste_model.eval()
        print(f"Taste: {taste_path}")

    # Sample
    print(f"Sampling {n_candidates} candidates (temp={sample_temp})...")
    with torch.no_grad():
        feats = prior.sample(n_candidates, device, temperature=sample_temp).cpu().tolist()
    candidates = [decode_taste_features(f) for f in feats]

    # Taste rerank
    scores = score_taste(candidates, taste_model, device)
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])

    # Render + save top N
    out_img = OUTPUT_DIR / "images"
    out_params = OUTPUT_DIR / "params"
    out_img.mkdir(parents=True, exist_ok=True)
    out_params.mkdir(parents=True, exist_ok=True)

    existing = list(out_img.glob("*.png"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Continuing from index {start_idx}")

    print(f"\nRendering top {n_results}...")
    saved = 0
    for idx in ranked:
        if saved >= n_results:
            break
        params = candidates[idx]
        rendered = render_api(params, endpoint)
        if rendered is None:
            continue

        file_idx = start_idx + saved
        rendered.save(out_img / f"{file_idx:06d}.png")
        with open(out_params / f"{file_idx:06d}.json", "w") as f:
            json.dump(params, f)

        print(f"  [{file_idx}] taste={scores[idx]:.4f} geo={params.get('geometry', '?')}")
        saved += 1

    print(f"\nSaved {saved} to {OUTPUT_DIR}/ ({start_idx + saved} total)")
    print("Curate with: make save-ref ID=out:0")


# ============================================================================
# CLI
# ============================================================================


def main():
    p = argparse.ArgumentParser(description="Inverse prediction + exploration")
    p.add_argument("--target", help="Target image path (inverse mode)")
    p.add_argument("--explore", action="store_true", help="Explore mode")
    p.add_argument("-n", type=int, default=20, help="Number of explore results")
    p.add_argument("--candidates", type=int, default=500, help="Explore candidate pool")
    p.add_argument("--model", default="models/inverse_model.pt")
    p.add_argument("--forward", default="models/forward_model.pt")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--fevals", type=int, default=2000, help="CMA-ES evaluations")
    p.add_argument("--taste", default="models/taste_model.pt")
    p.add_argument("--prior", default="models/param_prior.pt")
    p.add_argument("--sample-temp", type=float, default=1.0, help="Prior sample temperature")
    args = p.parse_args()

    if args.target:
        inverse_mode(
            target_path=args.target,
            model_path=args.model,
            forward_path=args.forward,
            endpoint=args.endpoint,
            max_fevals=args.fevals,
        )
    elif args.explore:
        explore_mode(
            n_results=args.n,
            n_candidates=args.candidates,
            endpoint=args.endpoint,
            taste_path=args.taste,
            prior_path=args.prior,
            sample_temp=args.sample_temp,
        )
    else:
        p.print_help()


if __name__ == "__main__":
    main()
