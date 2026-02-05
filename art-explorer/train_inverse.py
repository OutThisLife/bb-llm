"""
Phase 2: Train Inverse Model
============================
Train ResNet18 to predict params from images.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from inverse_model import (
    BOOLEANS,
    CATEGORICAL,
    LAYER_CONTINUOUS,
    MAX_LAYERS,
    InverseModel,
    normalize_continuous,
    normalize_layer,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ParamsDataset(Dataset):
    def __init__(self, data_dir="data", transform=None):
        self.data_dir = Path(data_dir)
        self.images = sorted(self.data_dir.glob("images/*.png"))
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        param_path = self.data_dir / "params" / f"{img_path.stem}.json"

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        with open(param_path) as f:
            params = json.load(f)

        # Continuous targets (normalized)
        cont = torch.tensor(normalize_continuous(params), dtype=torch.float32)

        # Categorical targets (indices)
        cat = {}
        for name, opts in CATEGORICAL.items():
            val = params.get(name, opts[0])
            cat[name] = opts.index(val) if val in opts else 0

        # Boolean targets
        bools = torch.tensor(
            [params.get(b, True) for b in BOOLEANS], dtype=torch.float32
        )

        # Layer targets
        layers = params.get("layers", [])
        layer_count = min(len(layers), MAX_LAYERS)

        # Normalize each layer (pad to MAX_LAYERS)
        layer_targets = []
        for i in range(MAX_LAYERS):
            if i < len(layers):
                layer_targets.append(
                    torch.tensor(normalize_layer(layers[i]), dtype=torch.float32)
                )
            else:
                # Default layer (zeros after normalization would be mid-range)
                layer_targets.append(
                    torch.tensor([0.5] * len(LAYER_CONTINUOUS), dtype=torch.float32)
                )

        return img, cont, cat, bools, layer_count, layer_targets


def collate_fn(batch):
    imgs, conts, cats, bools, layer_counts, layer_targets = zip(*batch)
    imgs = torch.stack(imgs)
    conts = torch.stack(conts)
    bools = torch.stack(bools)
    layer_counts = torch.tensor(layer_counts, dtype=torch.long)

    cat_tensors = {}
    for name in CATEGORICAL:
        cat_tensors[name] = torch.tensor([c[name] for c in cats], dtype=torch.long)

    # Stack layer targets: (batch, MAX_LAYERS, LAYER_CONTINUOUS)
    layer_tensors = []
    for i in range(MAX_LAYERS):
        layer_tensors.append(torch.stack([lt[i] for lt in layer_targets]))

    return imgs, conts, cat_tensors, bools, layer_counts, layer_tensors


def train(
    data_dir="data",
    epochs=10,
    batch_size=16,
    lr=1e-4,
    save_path="models/inverse_model.pt",
):
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    dataset = ParamsDataset(data_dir)
    print(f"Dataset: {len(dataset)} samples")

    if len(dataset) == 0:
        print("No data found. Run generate_data.py first.")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    model = InverseModel(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss weights
    cont_weight = 1.0
    cat_weight = 1.0
    bool_weight = 0.5
    layer_count_weight = 1.0
    layer_weight = 0.5

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        cat_correct = {k: 0 for k in CATEGORICAL}
        layer_count_correct = 0
        cat_total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for (
            imgs,
            cont_targets,
            cat_targets,
            bool_targets,
            layer_count_targets,
            layer_targets,
        ) in pbar:
            imgs = imgs.to(device)
            cont_targets = cont_targets.to(device)
            bool_targets = bool_targets.to(device)
            layer_count_targets = layer_count_targets.to(device)

            cont_pred, cat_pred, bool_pred, layer_count_pred, layer_preds = model(imgs)

            # MSE for continuous
            loss = cont_weight * F.mse_loss(cont_pred, cont_targets)

            # CrossEntropy for categoricals
            for k, pred in cat_pred.items():
                target = cat_targets[k].to(device)
                loss += cat_weight * F.cross_entropy(pred, target)
                cat_correct[k] += (pred.argmax(1) == target).sum().item()

            # BCE for boolean
            loss += bool_weight * F.binary_cross_entropy(bool_pred, bool_targets)

            # CrossEntropy for layer count
            loss += layer_count_weight * F.cross_entropy(
                layer_count_pred, layer_count_targets
            )
            layer_count_correct += (
                (layer_count_pred.argmax(1) == layer_count_targets).sum().item()
            )

            # MSE for layer params (only for layers that exist)
            for i in range(MAX_LAYERS):
                layer_target = layer_targets[i].to(device)
                layer_pred = layer_preds[i]
                # Mask: only compute loss for samples where this layer exists
                mask = (layer_count_targets > i).float().unsqueeze(1)
                if mask.sum() > 0:
                    layer_loss = (
                        (layer_pred - layer_target) ** 2 * mask
                    ).sum() / mask.sum()
                    loss += layer_weight * layer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cat_total += imgs.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = total_loss / len(loader)
        cat_acc = {k: cat_correct[k] / cat_total * 100 for k in CATEGORICAL}
        avg_cat_acc = sum(cat_acc.values()) / len(cat_acc)
        layer_acc = layer_count_correct / cat_total * 100

        print(
            f"Epoch {epoch + 1}: loss={avg_loss:.4f}, cat_acc={avg_cat_acc:.1f}%, layer_count_acc={layer_acc:.1f}%"
        )
        for k, acc in cat_acc.items():
            print(f"  {k}: {acc:.1f}%")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            save_path,
        )

    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description="Train inverse model")
    p.add_argument("--data", default="data", help="Data directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save", default="models/inverse_model.pt")
    args = p.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
