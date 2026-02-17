"""
Train Forward + Inverse + Taste Models
=======================================
All models train on the same coverage data (data/).
Forward: params → image (L1 + LPIPS perceptual loss)
Inverse: image → params (param-space + perceptual loss through frozen forward model)
Taste: refs vs random (binary classifier on param features)
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
    def __init__(self, data_dir="data", transform=None, raw_transform=None):
        d = Path(data_dir)
        self.images = sorted(d.glob("images/*.png"))
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

    cat_tensors = {
        name: torch.tensor([c[name] for c in cats], dtype=torch.long)
        for name in CATEGORICAL_KEYS
    }

    return (
        torch.stack(imgs), torch.stack(imgs_raw), torch.stack(conts),
        cat_tensors, torch.stack(bools), torch.tensor(layer_counts, dtype=torch.long),
        [torch.stack([lt[i] for lt in layer_targets]) for i in range(MAX_LAYERS)],
        [torch.stack([lp[i] for lp in layer_pres]) for i in range(MAX_LAYERS)],
        [torch.tensor([lg[i] for lg in layer_geos], dtype=torch.long) for i in range(MAX_LAYERS)],
    )


# ── Taste ──

class TasteDataset(Dataset):
    def __init__(self, features, labels):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_taste(data_dir="data", epochs=12, lr=3e-4, save_path="models/taste_model.pt"):
    device = get_device()
    print(f"[Taste] Device: {device}")

    refs = []
    if REFS_PATH.exists():
        for ln in REFS_PATH.read_text().strip().split("\n"):
            if ln:
                refs.append(json.loads(ln))
    if not refs:
        print("[Taste] No refs. Add refs first.")
        return

    # Positives = refs
    pos = [encode_taste_features(r) for r in refs]

    # Negatives = random sample from generated data
    neg_paths = sorted((Path(data_dir) / "params").glob("*.json"))
    neg_count = max(len(pos) * 4, 2000)
    if len(neg_paths) > neg_count:
        neg_paths = random.sample(neg_paths, neg_count)

    neg = []
    for p in neg_paths:
        try:
            params = json.loads(p.read_text())
            params.pop("url", None)
            neg.append(encode_taste_features(params))
        except Exception:
            continue

    if not neg:
        print("[Taste] No data for negatives. Generate first.")
        return

    features = pos + neg
    labels = [1.0] * len(pos) + [0.0] * len(neg)
    dataset = TasteDataset(features, labels)

    val_size = max(1, len(dataset) // 10)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    batch_size = max(64, auto_batch_size(model_vram_mb=40))
    n_workers = auto_workers()
    args = dict(batch_size=batch_size, num_workers=n_workers,
                pin_memory=device.type == "cuda", persistent_workers=n_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **args)
    val_loader = DataLoader(val_ds, shuffle=False, **args)
    print(f"[Taste] {len(pos)} refs vs {len(neg)} negatives")

    model = TasteModel(in_dim=TASTE_FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.BCEWithLogitsLoss()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, seen = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"[Taste] {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += ((torch.sigmoid(logits) > 0.5).float() == y).sum().item()
            seen += y.numel()

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)

        print(f"[Taste] Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
              f"acc={correct/seen*100:.1f}%, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "feature_dim": TASTE_FEATURE_DIM}, save_path)

    print(f"[Taste] Done → {save_path}")


# ── Forward ──

def train_forward(data_dir="data", epochs=20, lr=2e-4, save_path="models/forward_model.pt"):
    batch_size = auto_batch_size(model_vram_mb=100)
    device = get_device()
    print(f"[Forward] Device: {device}")

    full = ParamsDataset(data_dir)
    if not len(full):
        print("No data. Run generate.py first.")
        return

    val_size = max(1, len(full) // 10)
    train_ds, val_ds = random_split(full, [len(full) - val_size, val_size])
    val_ds.dataset = ParamsDataset(data_dir, transform=VAL_TRANSFORM)

    n_workers = auto_workers()
    kw = dict(batch_size=batch_size, collate_fn=collate_fn, num_workers=n_workers,
              pin_memory=device.type == "cuda", persistent_workers=True)
    loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    print(f"[Forward] {len(full)} samples, batch={batch_size}")

    model = ForwardModel().to(device)
    if device.type == "cuda":
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(loader, desc=f"[Fwd] {epoch+1}/{epochs}"):
            (_, imgs_raw, conts, cats, bools, lc, lt, lp, lg) = batch
            imgs_raw, conts, bools, lc = [t.to(device) for t in [imgs_raw, conts, bools, lc]]
            cats = {k: v.to(device) for k, v in cats.items()}
            lt = [t.to(device) for t in lt]
            lp = [t.to(device) for t in lp]
            lg = [t.to(device) for t in lg]

            with torch.amp.autocast(device.type, enabled=use_amp):
                pred = model(conts, cats, bools, lc, lt, lp, lg)
                l1 = F.l1_loss(pred, imgs_raw)
                perc = lpips_fn(pred * 2 - 1, imgs_raw * 2 - 1).mean()
                loss = l1 + 0.5 * perc

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                (_, imgs_raw, conts, cats, bools, lc, lt, lp, lg) = batch
                imgs_raw, conts, bools, lc = [t.to(device) for t in [imgs_raw, conts, bools, lc]]
                cats = {k: v.to(device) for k, v in cats.items()}
                lt = [t.to(device) for t in lt]
                lp = [t.to(device) for t in lp]
                lg = [t.to(device) for t in lg]
                pred = model(conts, cats, bools, lc, lt, lp, lg)
                val_loss += F.l1_loss(pred, imgs_raw).item()
        val_loss /= len(val_loader)

        print(f"[Fwd] Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, val_l1={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            print(f"  Saved (best={val_loss:.4f})")

    print(f"[Forward] Done → {save_path}")


# ── Inverse ──

def train_inverse(
    data_dir="data", epochs=10, lr=1e-4, save_path="models/inverse_model.pt",
    forward_path="models/forward_model.pt", perc_weight=0.5, freeze_backbone=3,
):
    batch_size = auto_batch_size(model_vram_mb=400)
    device = get_device()
    print(f"[Inverse] Device: {device}")

    full = ParamsDataset(data_dir)
    if not len(full):
        print("No data.")
        return

    val_size = max(1, len(full) // 10)
    train_ds, val_ds = random_split(full, [len(full) - val_size, val_size])
    val_ds.dataset = ParamsDataset(data_dir, transform=VAL_TRANSFORM)

    n_workers = auto_workers()
    kw = dict(batch_size=batch_size, collate_fn=collate_fn, num_workers=n_workers,
              pin_memory=device.type == "cuda", persistent_workers=True)
    loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    print(f"[Inverse] {len(full)} samples, batch={batch_size}")

    inverse = InverseModel(pretrained=True).to(device)

    if freeze_backbone > 0:
        for p in inverse.backbone.parameters():
            p.requires_grad = False
        print(f"[Inverse] Backbone frozen for {freeze_backbone} epochs")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, inverse.parameters()), lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    forward_model, lpips_fn = None, None
    if Path(forward_path).exists():
        forward_model = ForwardModel().to(device)
        ckpt = torch.load(forward_path, map_location=device, weights_only=False)
        forward_model.load_state_dict(load_state_dict_compat(ckpt["model_state_dict"]))
        forward_model.eval()
        for p in forward_model.parameters():
            p.requires_grad = False
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        print("[Inverse] Forward model loaded (frozen)")

    w = {"cont": 1.0, "cat": 1.0, "bool": 0.5, "lc": 1.0,
         "layer": 0.5, "lpres": 0.3, "lgeo": 0.3}

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        if epoch == freeze_backbone and freeze_backbone > 0:
            for p in inverse.backbone.parameters():
                p.requires_grad = True
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
        total_loss = 0

        pw = perc_weight * min(1.0, (epoch + 1) / max(1, epochs // 2))

        for batch in tqdm(loader, desc=f"[Inv] {epoch+1}/{epochs}"):
            (imgs, imgs_raw, cont_t, cat_t, bool_t, lc_t, lt_t, lp_t, lg_t) = batch
            imgs, imgs_raw = imgs.to(device), imgs_raw.to(device)
            cont_t, bool_t, lc_t = cont_t.to(device), bool_t.to(device), lc_t.to(device)

            with torch.amp.autocast(device.type, enabled=use_amp):
                (cont_p, cat_p, bool_p, lc_p, l_ps, lpr_ps, lg_ps) = inverse(imgs)

                loss = w["cont"] * F.mse_loss(cont_p, cont_t)
                for k, pred in cat_p.items():
                    loss = loss + w["cat"] * F.cross_entropy(pred, cat_t[k].to(device))
                loss = loss + w["bool"] * F.binary_cross_entropy_with_logits(bool_p, bool_t)
                loss = loss + w["lc"] * F.cross_entropy(lc_p, lc_t)

                for i in range(MAX_LAYERS):
                    mask = (lc_t > i).float().unsqueeze(1)
                    if mask.sum() == 0:
                        continue
                    loss = loss + w["layer"] * (((l_ps[i] - lt_t[i].to(device)) ** 2 * mask).sum() / mask.sum())
                    loss = loss + w["lpres"] * ((F.binary_cross_entropy_with_logits(
                        lpr_ps[i], lp_t[i].to(device), reduction="none") * mask).sum() / mask.sum())
                    gt = lg_t[i].to(device)
                    geo_on = lp_t[i][:, LAYER_OPTIONALS.index("geometry")].bool().to(device) & (lc_t > i).bool()
                    if geo_on.sum() > 0:
                        loss = loss + w["lgeo"] * F.cross_entropy(lg_ps[i][geo_on], gt[geo_on])

                if forward_model is not None and pw > 0:
                    fwd_img = forward_model(
                        cont_p.clamp(0, 1),
                        {k: p.argmax(1) for k, p in cat_p.items()},
                        bool_p, lc_p.argmax(1),
                        [lp.clamp(0, 1) for lp in l_ps],
                        list(lpr_ps),
                        [lg.argmax(1) for lg in lg_ps],
                    )
                    loss = loss + pw * lpips_fn(fwd_img * 2 - 1, imgs_raw * 2 - 1).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        inverse.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                cont_p, *_ = inverse(batch[0].to(device))
                val_loss += F.mse_loss(cont_p, batch[2].to(device)).item()
        val_loss /= len(val_loader)

        print(f"[Inv] Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, val_mse={val_loss:.4f}")
        torch.save({"model_state_dict": inverse.state_dict()}, save_path)

    print(f"[Inverse] Done → {save_path}")


# ── Inverse Phase 2: taste-scored fine-tuning ──

class TasteFilteredDataset(Dataset):
    """Subset of ParamsDataset filtered by taste model scores (top K%)."""

    def __init__(self, data_dir, indices, transform=None, raw_transform=None):
        full = ParamsDataset(data_dir, transform=transform, raw_transform=raw_transform)
        self.parent = full
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]


def _score_data_with_taste(data_dir, taste_path="models/taste_model.pt", top_pct=0.2):
    """Score all params in data_dir with taste model, return indices of top K%."""
    device = get_device()

    if not Path(taste_path).exists():
        print("[Finetune] No taste model found, skipping.")
        return None

    taste = TasteModel(in_dim=TASTE_FEATURE_DIM).to(device)
    ckpt = torch.load(taste_path, map_location=device, weights_only=False)
    taste.load_state_dict(load_state_dict_compat(ckpt["model_state_dict"]))
    taste.eval()

    param_files = sorted((Path(data_dir) / "params").glob("*.json"))
    if not param_files:
        return None

    scores = []
    for pf in param_files:
        try:
            params = json.loads(pf.read_text())
            params.pop("url", None)
            feat = torch.tensor(encode_taste_features(params), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                scores.append(torch.sigmoid(taste(feat)).item())
        except Exception:
            scores.append(0.0)

    # Select top K%
    n_keep = max(1, int(len(scores) * top_pct))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = ranked[:n_keep]
    threshold = scores[top_indices[-1]] if top_indices else 0

    print(f"[Finetune] Scored {len(scores)} samples, keeping top {n_keep} "
          f"(threshold={threshold:.3f})")
    return top_indices


def finetune_inverse(
    data_dir="data", epochs=5, lr=2e-5, save_path="models/inverse_model.pt",
    forward_path="models/forward_model.pt", taste_path="models/taste_model.pt",
    top_pct=0.2, perc_weight=0.5,
):
    """Fine-tune inverse model on taste-filtered top samples."""
    top_indices = _score_data_with_taste(data_dir, taste_path, top_pct)
    if top_indices is None:
        return

    batch_size = auto_batch_size(model_vram_mb=400)
    device = get_device()

    train_ds = TasteFilteredDataset(data_dir, top_indices)
    val_size = max(1, len(train_ds) // 10)
    train_ds_split, val_ds = random_split(train_ds, [len(train_ds) - val_size, val_size])

    n_workers = auto_workers()
    kw = dict(batch_size=batch_size, collate_fn=collate_fn, num_workers=n_workers,
              pin_memory=device.type == "cuda", persistent_workers=True)
    loader = DataLoader(train_ds_split, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    print(f"[Finetune] {len(train_ds)} samples, batch={batch_size}")

    # Load Phase 1 inverse model
    inverse = InverseModel(pretrained=False).to(device)
    if Path(save_path).exists():
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        inverse.load_state_dict(load_state_dict_compat(ckpt["model_state_dict"]))
        print("[Finetune] Loaded Phase 1 inverse model")

    optimizer = torch.optim.AdamW([
        {"params": inverse.backbone.parameters(), "lr": lr * 0.1},
        {"params": [p for n, p in inverse.named_parameters()
                   if not n.startswith("backbone")]},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    forward_model, lpips_fn = None, None
    if Path(forward_path).exists():
        forward_model = ForwardModel().to(device)
        ckpt = torch.load(forward_path, map_location=device, weights_only=False)
        forward_model.load_state_dict(load_state_dict_compat(ckpt["model_state_dict"]))
        forward_model.eval()
        for p in forward_model.parameters():
            p.requires_grad = False
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    w = {"cont": 1.0, "cat": 1.0, "bool": 0.5, "lc": 1.0,
         "layer": 0.5, "lpres": 0.3, "lgeo": 0.3}

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        inverse.train()
        total_loss = 0
        pw = perc_weight

        for batch in tqdm(loader, desc=f"[FT] {epoch+1}/{epochs}"):
            (imgs, imgs_raw, cont_t, cat_t, bool_t, lc_t, lt_t, lp_t, lg_t) = batch
            imgs, imgs_raw = imgs.to(device), imgs_raw.to(device)
            cont_t, bool_t, lc_t = cont_t.to(device), bool_t.to(device), lc_t.to(device)

            with torch.amp.autocast(device.type, enabled=use_amp):
                (cont_p, cat_p, bool_p, lc_p, l_ps, lpr_ps, lg_ps) = inverse(imgs)

                loss = w["cont"] * F.mse_loss(cont_p, cont_t)
                for k, pred in cat_p.items():
                    loss = loss + w["cat"] * F.cross_entropy(pred, cat_t[k].to(device))
                loss = loss + w["bool"] * F.binary_cross_entropy_with_logits(bool_p, bool_t)
                loss = loss + w["lc"] * F.cross_entropy(lc_p, lc_t)

                for i in range(MAX_LAYERS):
                    mask = (lc_t > i).float().unsqueeze(1)
                    if mask.sum() == 0:
                        continue
                    loss = loss + w["layer"] * (((l_ps[i] - lt_t[i].to(device)) ** 2 * mask).sum() / mask.sum())
                    loss = loss + w["lpres"] * ((F.binary_cross_entropy_with_logits(
                        lpr_ps[i], lp_t[i].to(device), reduction="none") * mask).sum() / mask.sum())
                    gt = lg_t[i].to(device)
                    geo_on = lp_t[i][:, LAYER_OPTIONALS.index("geometry")].bool().to(device) & (lc_t > i).bool()
                    if geo_on.sum() > 0:
                        loss = loss + w["lgeo"] * F.cross_entropy(lg_ps[i][geo_on], gt[geo_on])

                if forward_model is not None and pw > 0:
                    fwd_img = forward_model(
                        cont_p.clamp(0, 1),
                        {k: p.argmax(1) for k, p in cat_p.items()},
                        bool_p, lc_p.argmax(1),
                        [lp.clamp(0, 1) for lp in l_ps],
                        list(lpr_ps),
                        [lg.argmax(1) for lg in lg_ps],
                    )
                    loss = loss + pw * lpips_fn(fwd_img * 2 - 1, imgs_raw * 2 - 1).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        inverse.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                cont_p, *_ = inverse(batch[0].to(device))
                val_loss += F.mse_loss(cont_p, batch[2].to(device)).item()
        val_loss /= len(val_loader)

        print(f"[FT] Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, val_mse={val_loss:.4f}")
        torch.save({"model_state_dict": inverse.state_dict()}, save_path)

    print(f"[Finetune] Done → {save_path}")


# ── CLI ──

def main():
    p = argparse.ArgumentParser(description="Train models")
    p.add_argument("--data", default="data", help="Data directory")
    p.add_argument("--forward-epochs", type=int, default=20)
    p.add_argument("--inverse-epochs", type=int, default=10)
    p.add_argument("--taste-epochs", type=int, default=12)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--top-pct", type=float, default=0.2, help="Top %% for taste fine-tuning")
    args = p.parse_args()

    print("=" * 50)
    print("Phase 1: Forward Model")
    print("=" * 50)
    train_forward(data_dir=args.data, epochs=args.forward_epochs)

    print()
    print("=" * 50)
    print("Phase 2: Inverse Model (coverage)")
    print("=" * 50)
    train_inverse(data_dir=args.data, epochs=args.inverse_epochs)

    print()
    print("=" * 50)
    print("Phase 3: Taste Model")
    print("=" * 50)
    train_taste(data_dir=args.data, epochs=args.taste_epochs)

    print()
    print("=" * 50)
    print("Phase 4: Inverse Fine-tune (taste-filtered)")
    print("=" * 50)
    finetune_inverse(
        data_dir=args.data, epochs=args.finetune_epochs, top_pct=args.top_pct,
    )


if __name__ == "__main__":
    main()
