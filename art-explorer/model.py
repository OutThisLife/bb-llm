"""
Forward Model (params → image) + Inverse Model (image → params)
================================================================
ForwardModel: differentiable renderer proxy with attention (~11M params)
InverseModel: DINOv2-S backbone with multi-head output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from utils import (
    CATEGORICAL_KEYS,
    BOOLEAN_KEYS,
    CONTINUOUS_KEYS,
    LAYER_CONTINUOUS_KEYS,
    LAYER_OPTIONALS,
    MAX_LAYERS,
    TASTE_FEATURE_DIM,
)

# Re-export for backward compat
FLAT_CONTINUOUS = CONTINUOUS_KEYS
LAYER_CONTINUOUS = LAYER_CONTINUOUS_KEYS
CATEGORICAL = CATEGORICAL_KEYS
BOOLEANS = BOOLEAN_KEYS

# ============================================================================
# Forward Model (params → image)
# ============================================================================

CAT_EMBED_DIMS = {
    "geometry": 8,
    "scaleProgression": 4,
    "rotationProgression": 4,
    "alphaProgression": 3,
    "positionProgression": 2,
    "origin": 4,
    "ditherType": 4,
    "ditherMatrix": 2,
}

LAYER_GEO_EMBED_DIM = 6


def _forward_input_dim():
    dim = len(CONTINUOUS_KEYS) + len(BOOLEAN_KEYS)
    dim += sum(CAT_EMBED_DIMS.values())
    dim += 1  # layer count
    per_layer = len(LAYER_CONTINUOUS_KEYS) + len(LAYER_OPTIONALS) + LAYER_GEO_EMBED_DIM
    dim += MAX_LAYERS * per_layer
    return dim


class SelfAttention(nn.Module):
    """Spatial self-attention at feature map resolution."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(B, 3, C, H * W).unbind(1)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return x + self.proj(out.transpose(1, 2).reshape(B, C, H, W))


class ForwardModel(nn.Module):
    """Differentiable renderer proxy: params → 3×256×256 image.

    v2: 8×8 spatial bottleneck + self-attention at 32×32. ~11M params.
    """

    def __init__(self):
        super().__init__()

        self.cat_embeds = nn.ModuleDict({
            k: nn.Embedding(len(opts), CAT_EMBED_DIMS[k])
            for k, opts in CATEGORICAL_KEYS.items()
        })

        n_geos = len(CATEGORICAL_KEYS["geometry"])
        self.layer_geo_embeds = nn.ModuleList([
            nn.Embedding(n_geos, LAYER_GEO_EMBED_DIM) for _ in range(MAX_LAYERS)
        ])

        input_dim = _forward_input_dim()

        # MLP → 256×8×8 (4× spatial info vs v1's 4×4)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 8→16→32(+attn)→64→128→256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn = SelfAttention(128)  # at 32×32
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, continuous, cat_indices, booleans, layer_count,
                layer_continuous, layer_presence, layer_geo_indices):
        parts = [continuous, booleans]
        for k in CATEGORICAL_KEYS:
            parts.append(self.cat_embeds[k](cat_indices[k]))
        parts.append((layer_count.float() / MAX_LAYERS).unsqueeze(1))
        for i in range(MAX_LAYERS):
            parts.append(layer_continuous[i])
            parts.append(layer_presence[i])
            parts.append(self.layer_geo_embeds[i](layer_geo_indices[i]))

        x = torch.cat(parts, dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = self.decoder(x)
        x = self.attn(x)
        return self.decoder2(x)


# ============================================================================
# Inverse Model (image → params)
# ============================================================================


class InverseModel(nn.Module):
    """DINOv2-S backbone with multi-head output."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        feat_dim = 384

        self.continuous_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, len(CONTINUOUS_KEYS)),
        )

        self.categorical_heads = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, len(opts)),
            )
            for k, opts in CATEGORICAL_KEYS.items()
        })

        self.bool_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(BOOLEAN_KEYS)),
        )

        self.layer_count_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, MAX_LAYERS + 1),
        )

        self.layer_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, len(LAYER_CONTINUOUS_KEYS)),
            )
            for _ in range(MAX_LAYERS)
        ])

        self.layer_presence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, len(LAYER_OPTIONALS)),
            )
            for _ in range(MAX_LAYERS)
        ])

        n_geos = len(CATEGORICAL_KEYS["geometry"])
        self.layer_geo_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_geos),
            )
            for _ in range(MAX_LAYERS)
        ])

    def forward(self, x):
        features = self.backbone(x)
        continuous = self.continuous_head(features)
        categorical = {k: head(features) for k, head in self.categorical_heads.items()}
        boolean = self.bool_head(features)
        layer_count = self.layer_count_head(features)
        layer_params = [head(features) for head in self.layer_heads]
        layer_presence = [head(features) for head in self.layer_presence_heads]
        layer_geos = [head(features) for head in self.layer_geo_heads]
        return continuous, categorical, boolean, layer_count, layer_params, layer_presence, layer_geos


class TasteModel(nn.Module):
    """Lightweight preference scorer: param features -> interest logit."""

    def __init__(self, in_dim=TASTE_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
