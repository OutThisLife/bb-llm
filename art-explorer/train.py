"""
Train Forward + Inverse Models
==============================
Phase 1: Forward model (params → image) on pixel + perceptual loss.
Phase 2: Inverse model (image → params) end-to-end through frozen forward model.
         Supports quality-filtered data (--quality) for the two-dataset lifecycle.
"""

import argparse
import json
import random
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from model import ForwardModel, InverseModel, MAX_LAYERS, TasteModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from utils import (
    BOOLEAN_KEYS,
    CATEGORICAL_KEYS,
    CONTINUOUS_KEYS,
    LAYER_CONTINUOUS_KEYS,
    LAYER_OPTIONALS,
    TASTE_FEATURE_DIM,
    auto_batch_size,
    auto_workers,
    encode_taste_features,
    get_device,
    get_layer_geometry_idx,
    get_layer_presence,
    load_state_dict_compat,
    normalize_continuous,
    normalize_layer,
)

IMG_SIZE = 252  # divisible by DINOv2 patch size (14)
FWD_SIZE = 256
REFS_PATH = Path("references/refs.jsonl")
SCORED_REFS_PATH = Path("references/refs-scored.jsonl")

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    _normalize,
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    _normalize,
])

RAW_TRANSFORM = transforms.Compose([
    transforms.Resize((FWD_SIZE, FWD_SIZE)),
    transforms.ToTensor(),
])


class ParamsDataset(Dataset):
    def __init__(self, data_dirs, transform=None, raw_transform=None):
        """Load from one or more data directories."""
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.images = []
        for d in data_dirs:
            d = Path(d)
            self.images.extend(sorted(d.glob("images/*.png")))
        self.transform = transform or TRAIN_TRANSFORM
        self.raw_transform = raw_transform or RAW_TRANSFORM

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            return self._load(idx)
        except Exception:
            return self._load(random.randint(0, len(self) - 1))

    def _load(self, idx):
        img_path = self.images[idx]
        param_path = img_path.parent.parent / "params" / f"{img_path.stem}.json"

        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img)
        img_raw = self.raw_transform(img)

        with open(param_path) as f:
            params = json.load(f)

        cont = torch.tensor(normalize_continuous(params), dtype=torch.float32)

        cat = {}
        for name, opts in CATEGORICAL_KEYS.items():
            val = params.get(name, opts[0])
            cat[name] = opts.index(val) if val in opts else 0

        bools = torch.tensor(
            [params.get(b, True) for b in BOOLEAN_KEYS], dtype=torch.float32
        )

        layers = params.get("layers", [])
        layer_count = min(len(layers), MAX_LAYERS)

        layer_targets, layer_presence_targets, layer_geo_targets = [], [], []
        for i in range(MAX_LAYERS):
            if i < len(layers):
                layer_targets.append(torch.tensor(normalize_layer(layers[i]), dtype=torch.float32))
                layer_presence_targets.append(torch.tensor(get_layer_presence(layers[i]), dtype=torch.float32))
                layer_geo_targets.append(get_layer_geometry_idx(layers[i]))
            else:
                layer_targets.append(torch.tensor([0.5] * len(LAYER_CONTINUOUS_KEYS), dtype=torch.float32))
                layer_presence_targets.append(torch.zeros(len(LAYER_OPTIONALS), dtype=torch.float32))
                layer_geo_targets.append(0)

        return (img_transformed, img_raw, cont, cat, bools,
                layer_count, layer_targets, layer_presence_targets, layer_geo_targets)


def collate_fn(batch):
    (imgs, imgs_raw, conts, cats, bools,
     layer_counts, layer_targets, layer_pres, layer_geos) = zip(*batch)

    imgs = torch.stack(imgs)
    imgs_raw = torch.stack(imgs_raw)
    conts = torch.stack(conts)
    bools = torch.stack(bools)
    layer_counts = torch.tensor(layer_counts, dtype=torch.long)

    cat_tensors = {
        name: torch.tensor([c[name] for c in cats], dtype=torch.long)
        for name in CATEGORICAL_KEYS
    }

    layer_tensors = [torch.stack([lt[i] for lt in layer_targets]) for i in range(MAX_LAYERS)]
    layer_pres_tensors = [torch.stack([lp[i] for lp in layer_pres]) for i in range(MAX_LAYERS)]
    layer_geo_tensors = [torch.tensor([lg[i] for lg in layer_geos], dtype=torch.long) for i in range(MAX_LAYERS)]

    return (imgs, imgs_raw, conts, cat_tensors, bools,
            layer_counts, layer_tensors, layer_pres_tensors, layer_geo_tensors)


# ============================================================================
# Taste model dataset + training
# ============================================================================


class TasteDataset(Dataset):
    def __init__(self, features, labels, weights=None):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.w = torch.tensor(weights if weights is not None else [1.0] * len(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


def _load_jsonl(path):
    if not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    return [json.loads(ln) for ln in text.split("\n") if ln]


def _collect_negative_params(data_dirs, target_count):
    candidates = []
    for d in data_dirs:
        d = Path(d)
        candidates.extend(sorted((d / "params").glob("*.json")))
    if not candidates:
        return []
    if len(candidates) > target_count:
        candidates = random.sample(candidates, target_count)
    out = []
    for p in candidates:
        try:
            params = json.loads(p.read_text())
            params.pop("url", None)
            out.append(params)
        except Exception:
            continue
    return out


def train_taste(
    data_dirs="data",
    epochs=12,
    batch_size=0,
    lr=3e-4,
    save_path="models/taste_model.pt",
):
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]
    device = get_device()
    if batch_size <= 0:
        batch_size = max(64, auto_batch_size(model_vram_mb=40))
    print(f"[Taste] Device: {device}")

    refs = _load_jsonl(REFS_PATH)
    if not refs:
        print("[Taste] No refs found. Add refs first.")
        return

    scored = _load_jsonl(SCORED_REFS_PATH)
    scored_map = {
        json.dumps(e.get("params", {}), sort_keys=True): float(e.get("score", 0.8))
        for e in scored
    }

    pos_features, pos_labels, pos_weights = [], [], []
    for ref in refs:
        pos_features.append(encode_taste_features(ref))
        pos_labels.append(1.0)
        key = json.dumps(ref, sort_keys=True)
        pos_weights.append(max(0.2, scored_map.get(key, 1.0)))

    neg = _collect_negative_params(data_dirs, target_count=max(len(pos_features) * 4, 2000))
    if not neg:
        print("[Taste] No negative pool in data dirs. Generate data first.")
        return

    neg_features = [encode_taste_features(p) for p in neg]
    neg_labels = [0.0] * len(neg_features)
    neg_weights = [1.0] * len(neg_features)

    features = pos_features + neg_features
    labels = pos_labels + neg_labels
    weights = pos_weights + neg_weights
    dataset = TasteDataset(features, labels, weights)

    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    n_workers = auto_workers()
    loader_args = dict(
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=n_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    print(f"[Taste] Dataset: {train_size} train, {val_size} val")
    print(f"[Taste] Positives: {len(pos_features)}, negatives: {len(neg_features)}")

    model = TasteModel(in_dim=TASTE_FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0

        pbar = tqdm(train_loader, desc=f"[Taste] Epoch {epoch + 1}/{epochs}")
        for x, y, w in pbar:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)

            logits = model(x)
            loss = (criterion(logits, y) * w).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == y).sum().item()
            seen += y.numel()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_acc = correct / max(1, seen)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for x, y, w in val_loader:
                x = x.to(device)
                y = y.to(device)
                w = w.to(device)
                logits = model(x)
                loss = (criterion(logits, y) * w).mean()
                val_loss += loss.item()
                pred = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (pred == y).sum().item()
                val_seen += y.numel()
        val_loss /= len(val_loader)
        val_acc = val_correct / max(1, val_seen)

        print(
            f"[Taste] Epoch {epoch + 1}: loss={train_loss:.4f}, acc={train_acc*100:.1f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.1f}%"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "feature_dim": TASTE_FEATURE_DIM,
                },
                save_path,
            )
            print(f"  Saved (best val_loss={val_loss:.4f})")

    print(f"[Taste] Done. Best val_loss={best_val:.4f} -> {save_path}")


# ============================================================================
# Phase 1: Train Forward Model
# ============================================================================


def train_forward(
    data_dirs="data",
    epochs=20,
    batch_size=0,
    lr=2e-4,
    save_path="models/forward_model.pt",
):
    if batch_size <= 0:
        batch_size = auto_batch_size(model_vram_mb=100)
    device = get_device()
    print(f"[Forward] Device: {device}")

    full_dataset = ParamsDataset(data_dirs)
    if len(full_dataset) == 0:
        print("No data found. Run generate.py first.")
        return

    val_size = max(1, len(full_dataset) // 10)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset = ParamsDataset(data_dirs, transform=VAL_TRANSFORM)
    print(f"[Forward] Dataset: {train_size} train, {val_size} val")

    n_workers = auto_workers()
    loader_args = dict(
        batch_size=batch_size, collate_fn=collate_fn,
        num_workers=n_workers, pin_memory=device.type == "cuda", persistent_workers=True,
    )
    loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    print(f"[Forward] batch={batch_size}, workers={n_workers}")

    model = ForwardModel().to(device)
    if device.type == "cuda":
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad = False

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, total_l1, total_lpips = 0, 0, 0

        pbar = tqdm(loader, desc=f"[Fwd] Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            (imgs, imgs_raw, conts, cat_targets, bool_targets,
             layer_counts, layer_targets, layer_pres_targets, layer_geo_targets) = batch

            imgs_raw = imgs_raw.to(device)
            conts = conts.to(device)
            bool_targets = bool_targets.to(device)
            layer_counts = layer_counts.to(device)
            cat_dev = {k: v.to(device) for k, v in cat_targets.items()}
            lt_dev = [lt.to(device) for lt in layer_targets]
            lp_dev = [lp.to(device) for lp in layer_pres_targets]
            lg_dev = [lg.to(device) for lg in layer_geo_targets]

            with torch.amp.autocast(device.type, enabled=use_amp):
                pred_img = model(conts, cat_dev, bool_targets, layer_counts, lt_dev, lp_dev, lg_dev)
                l1_loss = F.l1_loss(pred_img, imgs_raw)
                perc_loss = lpips_model(pred_img * 2 - 1, imgs_raw * 2 - 1).mean()
                loss = l1_loss + 0.5 * perc_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_lpips += perc_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", l1=f"{l1_loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        avg_l1 = total_l1 / len(loader)
        avg_lpips = total_lpips / len(loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                (_, imgs_raw, conts, cat_targets, bool_targets,
                 layer_counts, layer_targets, layer_pres_targets, layer_geo_targets) = batch
                imgs_raw = imgs_raw.to(device)
                conts = conts.to(device)
                bool_targets = bool_targets.to(device)
                layer_counts = layer_counts.to(device)
                cat_dev = {k: v.to(device) for k, v in cat_targets.items()}
                lt_dev = [lt.to(device) for lt in layer_targets]
                lp_dev = [lp.to(device) for lp in layer_pres_targets]
                lg_dev = [lg.to(device) for lg in layer_geo_targets]
                pred_img = model(conts, cat_dev, bool_targets, layer_counts, lt_dev, lp_dev, lg_dev)
                val_loss += F.l1_loss(pred_img, imgs_raw).item()
        val_loss /= len(val_loader)

        print(
            f"[Fwd] Epoch {epoch + 1}: loss={avg_loss:.4f} "
            f"(l1={avg_l1:.4f}, lpips={avg_lpips:.4f}), val_l1={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss},
                save_path,
            )
            print(f"  Saved (best val_l1={val_loss:.4f})")

    print(f"[Forward] Done. Best val_l1={best_val_loss:.4f} → {save_path}")


# ============================================================================
# Phase 2: Train Inverse Model
# ============================================================================


def train_inverse(
    data_dirs="data",
    epochs=10,
    batch_size=0,
    lr=1e-4,
    save_path="models/inverse_model.pt",
    forward_path="models/forward_model.pt",
    perceptual_weight=0.5,
    freeze_backbone=3,
):
    if batch_size <= 0:
        batch_size = auto_batch_size(model_vram_mb=400)  # DINOv2 + forward model
    device = get_device()
    print(f"[Inverse] Device: {device}")

    full_dataset = ParamsDataset(data_dirs)
    if len(full_dataset) == 0:
        print("No data found.")
        return

    val_size = max(1, len(full_dataset) // 10)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset = ParamsDataset(data_dirs, transform=VAL_TRANSFORM)
    print(f"[Inverse] Dataset: {train_size} train, {val_size} val")

    n_workers = auto_workers()
    loader_args = dict(
        batch_size=batch_size, collate_fn=collate_fn,
        num_workers=n_workers, pin_memory=device.type == "cuda", persistent_workers=True,
    )
    loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    print(f"[Inverse] batch={batch_size}, workers={n_workers}")

    inverse = InverseModel(pretrained=True).to(device)

    # Freeze DINOv2 backbone for initial epochs
    if freeze_backbone > 0:
        for p in inverse.backbone.parameters():
            p.requires_grad = False
        print(f"[Inverse] DINOv2 backbone frozen for {freeze_backbone} epochs")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, inverse.parameters()), lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Forward model (frozen) for perceptual loss
    forward_model, lpips_model = None, None
    if Path(forward_path).exists():
        forward_model = ForwardModel().to(device)
        ckpt = torch.load(forward_path, map_location=device, weights_only=False)
        forward_model.load_state_dict(load_state_dict_compat(ckpt["model_state_dict"]))
        forward_model.eval()
        for p in forward_model.parameters():
            p.requires_grad = False
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
        lpips_model.eval()
        for p in lpips_model.parameters():
            p.requires_grad = False
        print(f"[Inverse] Forward model loaded (frozen)")
    else:
        print(f"[Inverse] No forward model — param-space loss only")

    # Loss weights
    w = {"cont": 1.0, "cat": 1.0, "bool": 0.5, "lc": 1.0,
         "layer": 0.5, "lpres": 0.3, "lgeo": 0.3}

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        # Unfreeze backbone after freeze_backbone epochs
        if epoch == freeze_backbone and freeze_backbone > 0:
            for p in inverse.backbone.parameters():
                p.requires_grad = True
            # Rebuild optimizer with all params + lower LR for backbone
            optimizer = torch.optim.AdamW([
                {"params": inverse.backbone.parameters(), "lr": lr * 0.1},
                {"params": [p for n, p in inverse.named_parameters()
                           if not n.startswith("backbone")]},
            ], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_backbone,
            )
            print(f"[Inverse] Backbone unfrozen (lr={lr * 0.1:.1e})")

        inverse.train()
        total_loss, total_param, total_perc = 0, 0, 0
        cat_correct = {k: 0 for k in CATEGORICAL_KEYS}
        layer_count_correct, cat_total = 0, 0

        perc_w = perceptual_weight * min(1.0, (epoch + 1) / max(1, epochs // 2))

        pbar = tqdm(loader, desc=f"[Inv] Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            (imgs, imgs_raw, cont_targets, cat_targets, bool_targets,
             layer_count_targets, layer_targets, layer_pres_targets, layer_geo_targets) = batch

            imgs = imgs.to(device)
            imgs_raw = imgs_raw.to(device)
            cont_targets = cont_targets.to(device)
            bool_targets = bool_targets.to(device)
            layer_count_targets = layer_count_targets.to(device)

            with torch.amp.autocast(device.type, enabled=use_amp):
                (cont_pred, cat_pred, bool_pred, layer_count_pred,
                 layer_preds, layer_pres_preds, layer_geo_preds) = inverse(imgs)

                # Param-space loss
                param_loss = w["cont"] * F.mse_loss(cont_pred, cont_targets)

                for k, pred in cat_pred.items():
                    target = cat_targets[k].to(device)
                    param_loss = param_loss + w["cat"] * F.cross_entropy(pred, target)
                    cat_correct[k] += (pred.argmax(1) == target).sum().item()

                param_loss = param_loss + w["bool"] * F.binary_cross_entropy_with_logits(
                    bool_pred, bool_targets
                )
                param_loss = param_loss + w["lc"] * F.cross_entropy(
                    layer_count_pred, layer_count_targets
                )
                layer_count_correct += (layer_count_pred.argmax(1) == layer_count_targets).sum().item()

                for i in range(MAX_LAYERS):
                    has_layer = (layer_count_targets > i).float()
                    mask = has_layer.unsqueeze(1)
                    if mask.sum() == 0:
                        continue
                    lt = layer_targets[i].to(device)
                    param_loss = param_loss + w["layer"] * (
                        ((layer_preds[i] - lt) ** 2 * mask).sum() / mask.sum()
                    )
                    pt = layer_pres_targets[i].to(device)
                    param_loss = param_loss + w["lpres"] * (
                        (F.binary_cross_entropy_with_logits(
                            layer_pres_preds[i], pt, reduction="none") * mask
                        ).sum() / mask.sum()
                    )
                    gt = layer_geo_targets[i].to(device)
                    geo_present = pt[:, LAYER_OPTIONALS.index("geometry")].bool() & has_layer.bool()
                    if geo_present.sum() > 0:
                        param_loss = param_loss + w["lgeo"] * F.cross_entropy(
                            layer_geo_preds[i][geo_present], gt[geo_present]
                        )

                loss = param_loss

                # Perceptual loss through forward model
                if forward_model is not None and perc_w > 0:
                    cont_clamped = cont_pred.clamp(0, 1)
                    cat_idx = {k: pred.argmax(1) for k, pred in cat_pred.items()}
                    lc_idx = layer_count_pred.argmax(1)
                    lc_list = [lp.clamp(0, 1) for lp in layer_preds]
                    lp_list = list(layer_pres_preds)
                    lg_list = [lg.argmax(1) for lg in layer_geo_preds]

                    fwd_img = forward_model(
                        cont_clamped, cat_idx, bool_pred, lc_idx,
                        lc_list, lp_list, lg_list,
                    )
                    perc_loss = lpips_model(fwd_img * 2 - 1, imgs_raw * 2 - 1).mean()
                    loss = loss + perc_w * perc_loss
                    total_perc += perc_loss.item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_param += param_loss.item()
            cat_total += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = total_loss / len(loader)
        avg_param = total_param / len(loader)
        avg_perc = total_perc / len(loader) if forward_model else 0
        avg_cat_acc = sum(v / cat_total * 100 for v in cat_correct.values()) / len(CATEGORICAL_KEYS)
        layer_acc = layer_count_correct / cat_total * 100

        # Validation
        inverse.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs_v = batch[0].to(device)
                cont_t = batch[2].to(device)
                cont_p, *_ = inverse(imgs_v)
                val_loss += F.mse_loss(cont_p, cont_t).item()
        val_loss /= len(val_loader)

        print(
            f"[Inv] Epoch {epoch + 1}: loss={avg_loss:.4f} "
            f"(param={avg_param:.4f}, perc={avg_perc:.4f}), "
            f"val_mse={val_loss:.4f}, cat={avg_cat_acc:.1f}%, layers={layer_acc:.1f}%"
        )

        torch.save(
            {"epoch": epoch, "model_state_dict": inverse.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss},
            save_path,
        )

    print(f"[Inverse] Done → {save_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    p = argparse.ArgumentParser(description="Train forward + inverse models")
    p.add_argument("--data", nargs="+", default=["data"], help="Data directories (multiple for cumulative)")
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--inverse-only", action="store_true")
    p.add_argument("--taste-only", action="store_true", help="Train taste model (refs vs generated pool)")
    p.add_argument("--forward-epochs", type=int, default=20)
    p.add_argument("--inverse-epochs", type=int, default=10)
    p.add_argument("--taste-epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=0, help="Batch size (0=auto)")
    p.add_argument("--lr", type=float, default=2e-4, help="LR (forward)")
    p.add_argument("--inv-lr", type=float, default=1e-4, help="LR (inverse)")
    p.add_argument("--taste-lr", type=float, default=3e-4, help="LR (taste)")
    p.add_argument("--perc-weight", type=float, default=0.5, help="Perceptual loss weight")
    p.add_argument("--freeze-backbone", type=int, default=3, help="Epochs to freeze DINOv2 backbone")
    args = p.parse_args()

    if args.taste_only:
        print("=" * 50)
        print("Training Taste Model")
        print("=" * 50)
        train_taste(
            data_dirs=args.data,
            epochs=args.taste_epochs,
            batch_size=args.batch,
            lr=args.taste_lr,
        )
        return

    if not args.inverse_only:
        print("=" * 50)
        print("Phase 1: Training Forward Model")
        print("=" * 50)
        train_forward(
            data_dirs=args.data,
            epochs=args.forward_epochs,
            batch_size=args.batch,
            lr=args.lr,
        )

    if not args.forward_only:
        print()
        print("=" * 50)
        print("Phase 2: Training Inverse Model")
        print("=" * 50)
        train_inverse(
            data_dirs=args.data,
            epochs=args.inverse_epochs,
            batch_size=args.batch,
            lr=args.inv_lr,
            perceptual_weight=args.perc_weight,
            freeze_backbone=args.freeze_backbone,
        )


if __name__ == "__main__":
    main()
