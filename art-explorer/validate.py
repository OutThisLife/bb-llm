"""
Phase 3: Validation
===================
Validate inverse model on ref.png and figma-*.png references.
Writes good predictions (SSIM > threshold) to refs.jsonl for feedback loop.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import requests
import torch
from model import InverseModel, reconstruct_params
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from utils import to_prefixed


def load_model(model_path="models/inverse_model.pt"):
    """Load trained inverse model."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = InverseModel(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def load_and_transform(image_path):
    """Load and transform image for inference."""
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return transform(img)


def render_params(params, endpoint="http://localhost:3000/api/raster"):
    """Render params via API."""
    resp = requests.post(endpoint, json=params, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def compute_ssim(img1, img2):
    """Compute SSIM between two PIL images."""
    # Resize to same size
    size = (min(img1.width, img2.width), min(img1.height, img2.height))
    img1 = img1.resize(size).convert("RGB")
    img2 = img2.resize(size).convert("RGB")

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    return ssim(arr1, arr2, channel_axis=2, data_range=255)


def append_to_refs(params, refs_path="references/refs.jsonl"):
    """Append params to refs.jsonl for feedback loop."""
    # Try both paths (running from repo root or art-explorer/)
    for p in [Path(refs_path), Path(f"art-explorer/{refs_path}")]:
        if p.parent.exists():
            prefixed = to_prefixed(params)
            with open(p, "a") as f:
                f.write(json.dumps(prefixed) + "\n")
            print(f"→ Added to {p} for next generation")
            return
    print("Warning: couldn't find refs.jsonl location")


def validate_single(model, device, image_path, endpoint, save_dir=None):
    """Validate on a single image."""
    print(f"\n{'=' * 50}")
    print(f"Validating: {image_path}")

    # Inference
    img_tensor = load_and_transform(image_path).unsqueeze(0).to(device)
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

    print("Predicted params:")
    for k, v in sorted(params.items()):
        if k not in ("layers", "debug"):
            print(f"  {k}: {v}")

    # Show layer predictions
    layers = params.get("layers", [])
    print(f"  layers: {len(layers)}")
    for i, layer in enumerate(layers):
        pos = layer.get("position", {})
        scl = layer.get("scale", {})
        rot = layer.get("rotation", 0)
        extras = []
        for k in ("stepFactor", "alphaFactor", "scaleFactor", "rotationFactor", "geometry"):
            if k in layer:
                extras.append(f"{k}={layer[k]}")
        extra_str = f" {' '.join(extras)}" if extras else ""
        print(
            f"    [{i}] pos=({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}) "
            f"scale=({scl.get('x', 1):.2f}, {scl.get('y', 1):.2f}) rot={rot:.2f}{extra_str}"
        )

    # Render
    try:
        reconstructed = render_params(params, endpoint)
    except Exception as e:
        print(f"Render failed: {e}")
        print("Aborting - is bb-particles running?")
        raise SystemExit(1)

    # Compare
    original = Image.open(image_path).convert("RGB")
    score = compute_ssim(original, reconstructed)

    print(f"\nSSIM: {score:.4f}")

    if score >= 0.85:
        print("✓ PASS (>0.85)")
        append_to_refs(params)  # Feedback: save good predictions
    elif score >= 0.7:
        print("○ REFINE (0.7-0.85) - use as CMA-ES seed")
        append_to_refs(params)  # Feedback: save decent predictions too
    else:
        print("✗ DEBUG (<0.7) - check training/data")

    # Save reconstruction
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        name = Path(image_path).stem
        reconstructed.save(save_dir / f"{name}_reconstructed.png")
        print(f"Saved: {save_dir / f'{name}_reconstructed.png'}")

    return score, params


def validate_all(model_path, endpoint, refs_dir="references", save_dir="validation"):
    """Validate on all reference images."""
    model, device = load_model(model_path)
    refs_dir = Path(refs_dir)

    results = []

    # ref.png (exact params exist)
    ref_path = refs_dir / "ref.png"
    if ref_path.exists():
        score, params = validate_single(model, device, ref_path, endpoint, save_dir)
        results.append(("ref.png", score, "exact"))

    # figma-*.png (best achievable)
    for figma in sorted(refs_dir.glob("figma-*.png")):
        score, params = validate_single(model, device, figma, endpoint, save_dir)
        results.append((figma.name, score, "external"))

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    for name, score, type_ in results:
        if score is None:
            print(f"{name}: FAILED")
        else:
            status = "✓" if score >= 0.85 else ("○" if score >= 0.7 else "✗")
            print(f"{name}: {score:.4f} {status} ({type_})")


def main():
    p = argparse.ArgumentParser(description="Validate inverse model")
    p.add_argument("--model", default="models/inverse_model.pt")
    p.add_argument("--image", help="Single image to validate")
    p.add_argument("--endpoint", default="http://localhost:3000/api/raster")
    p.add_argument("--refs", default="references", help="References directory")
    p.add_argument("--save", default="validation", help="Save reconstructions to")
    args = p.parse_args()

    if args.image:
        model, device = load_model(args.model)
        validate_single(model, device, args.image, args.endpoint, args.save)
    else:
        validate_all(args.model, args.endpoint, args.refs, args.save)


if __name__ == "__main__":
    main()
