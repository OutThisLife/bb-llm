"""
Inverse Model: image → params
=============================
ResNet18 backbone with multi-head output for continuous, categorical, and boolean params.
Supports up to MAX_LAYERS layers with per-layer params.
"""

import timm
import torch
import torch.nn as nn

MAX_LAYERS = 5

# Flattened continuous params (flat keys, nested objects flattened)
# Note: Scene.rotation/position/scale are fixed, so not predicted
FLAT_CONTINUOUS = [
    "repetitions",
    "alphaFactor",
    "scaleFactor",
    "rotationFactor",
    "stepFactor",
    "xStep",
    "yStep",
    # Noise
    "noiseDensity",
    "noiseOpacity",
    "noiseSize",
]

# Per-layer continuous params (for each of MAX_LAYERS)
LAYER_CONTINUOUS = [
    "position.x",
    "position.y",
    "rotation",
    "scale.x",
    "scale.y",
    "stepFactor",
    "alphaFactor",
    "scaleFactor",
    "rotationFactor",
]

# Per-layer optional params (presence predicted separately)
LAYER_OPTIONALS = ["stepFactor", "alphaFactor", "scaleFactor", "rotationFactor", "geometry"]

# Normalization ranges for continuous params (for training stability)
CONTINUOUS_RANGES = {
    "repetitions": (1, 500),
    "alphaFactor": (0, 1),
    "scaleFactor": (0, 3),
    "rotationFactor": (-1, 1),
    "stepFactor": (0, 2),
    "xStep": (-3, 3),
    "yStep": (-3, 3),
    # Noise
    "noiseDensity": (0, 1),
    "noiseOpacity": (0, 1),
    "noiseSize": (0.1, 10),
}

LAYER_RANGES = {
    "position.x": (-2, 2),
    "position.y": (-2, 2),
    "rotation": (-3.14159, 3.14159),
    "scale.x": (-2, 2),
    "scale.y": (-2, 2),
    "stepFactor": (0, 2),
    "alphaFactor": (0, 1),
    "scaleFactor": (0, 2),
    "rotationFactor": (-1, 1),
}

# Categorical params
CATEGORICAL = {
    "geometry": [
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
    "scaleProgression": [
        "linear",
        "exponential",
        "additive",
        "fibonacci",
        "golden",
        "sine",
    ],
    "rotationProgression": ["linear", "golden-angle", "fibonacci", "sine"],
    "alphaProgression": ["exponential", "linear", "inverse"],
    "positionProgression": ["index", "scale"],
    "origin": [
        "center",
        "top-center",
        "bottom-center",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ],
}

# Booleans
BOOLEANS = ["positionCoupled", "noiseEnabled"]


class InverseModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18", pretrained=pretrained, num_classes=0
        )
        feat_dim = 512

        # Continuous head (base params)
        self.continuous_head = nn.Linear(feat_dim, len(FLAT_CONTINUOUS))

        # Categorical heads
        self.categorical_heads = nn.ModuleDict(
            {k: nn.Linear(feat_dim, len(opts)) for k, opts in CATEGORICAL.items()}
        )

        # Boolean head
        self.bool_head = nn.Linear(feat_dim, len(BOOLEANS))

        # Layer count head (0-5 layers)
        self.layer_count_head = nn.Linear(feat_dim, MAX_LAYERS + 1)

        # Per-layer continuous params (all MAX_LAYERS, masked by count)
        self.layer_heads = nn.ModuleList(
            [nn.Linear(feat_dim, len(LAYER_CONTINUOUS)) for _ in range(MAX_LAYERS)]
        )

        # Per-layer optional presence flags
        self.layer_presence_heads = nn.ModuleList(
            [nn.Linear(feat_dim, len(LAYER_OPTIONALS)) for _ in range(MAX_LAYERS)]
        )

        # Per-layer geometry (categorical)
        n_geos = len(CATEGORICAL["geometry"])
        self.layer_geo_heads = nn.ModuleList(
            [nn.Linear(feat_dim, n_geos) for _ in range(MAX_LAYERS)]
        )

    def forward(self, x):
        features = self.backbone(x)
        continuous = self.continuous_head(features)
        categorical = {k: head(features) for k, head in self.categorical_heads.items()}
        boolean = torch.sigmoid(self.bool_head(features))
        layer_count = self.layer_count_head(features)
        layer_params = [head(features) for head in self.layer_heads]
        layer_presence = [torch.sigmoid(head(features)) for head in self.layer_presence_heads]
        layer_geos = [head(features) for head in self.layer_geo_heads]

        return continuous, categorical, boolean, layer_count, layer_params, layer_presence, layer_geos


def normalize_continuous(params: dict) -> list[float]:
    """Normalize continuous params to [0, 1] for training."""
    values = []

    for name in FLAT_CONTINUOUS:
        v = params.get(name, 0)
        lo, hi = CONTINUOUS_RANGES[name]
        normalized = (v - lo) / (hi - lo) if hi > lo else 0.5
        values.append(max(0, min(1, normalized)))

    return values


def normalize_layer(layer: dict) -> list[float]:
    """Normalize a single layer's params to [0, 1]."""
    values = []

    for name in LAYER_CONTINUOUS:
        if name == "position.x":
            v = layer.get("position", {}).get("x", 0)
        elif name == "position.y":
            v = layer.get("position", {}).get("y", 0)
        elif name == "scale.x":
            v = (
                layer.get("scale", {}).get("x", 1)
                if isinstance(layer.get("scale"), dict)
                else 1
            )
        elif name == "scale.y":
            v = (
                layer.get("scale", {}).get("y", 1)
                if isinstance(layer.get("scale"), dict)
                else 1
            )
        elif name == "rotation":
            v = layer.get("rotation", 0)
        else:
            # Optional continuous (stepFactor, alphaFactor, etc.)
            # Mid-range default when absent (loss masked by presence)
            lo, hi = LAYER_RANGES[name]
            v = layer.get(name, (lo + hi) / 2)

        lo, hi = LAYER_RANGES[name]
        normalized = (v - lo) / (hi - lo) if hi > lo else 0.5
        values.append(max(0, min(1, normalized)))

    return values


def get_layer_presence(layer: dict) -> list[bool]:
    """Which optional params are present in this layer."""
    return [name in layer for name in LAYER_OPTIONALS]


def get_layer_geometry_idx(layer: dict) -> int:
    """Get geometry index for layer (0 if absent)."""
    geo = layer.get("geometry")
    if geo and geo in CATEGORICAL["geometry"]:
        return CATEGORICAL["geometry"].index(geo)
    return 0


def denormalize_continuous(values: list[float]) -> dict:
    """Denormalize [0, 1] values back to original ranges."""
    result = {}

    for i, name in enumerate(FLAT_CONTINUOUS):
        lo, hi = CONTINUOUS_RANGES[name]
        v = lo + values[i] * (hi - lo)
        if name == "repetitions":
            result[name] = round(v)
        else:
            result[name] = round(v, 4)

    return result


def denormalize_layer(values: list[float], presence=None, geo_idx=None) -> dict:
    """Denormalize layer params back to original ranges."""
    layer = {"position": {}, "scale": {}}
    for i, name in enumerate(LAYER_CONTINUOUS):
        lo, hi = LAYER_RANGES[name]
        v = lo + values[i] * (hi - lo)
        if name == "position.x":
            layer["position"]["x"] = round(v, 4)
        elif name == "position.y":
            layer["position"]["y"] = round(v, 4)
        elif name == "scale.x":
            layer["scale"]["x"] = round(v, 4)
        elif name == "scale.y":
            layer["scale"]["y"] = round(v, 4)
        elif name == "rotation":
            layer["rotation"] = round(v, 4)
        elif presence is not None and name in LAYER_OPTIONALS:
            if presence[LAYER_OPTIONALS.index(name)]:
                layer[name] = round(v, 4)
        else:
            layer[name] = round(v, 4)

    # Per-layer geometry
    if geo_idx is not None and presence is not None:
        geo_i = LAYER_OPTIONALS.index("geometry")
        if presence[geo_i]:
            layer["geometry"] = CATEGORICAL["geometry"][geo_idx]

    return layer


def reconstruct_params(
    continuous, categorical, boolean,
    layer_count=None, layer_params=None,
    layer_presence=None, layer_geos=None,
) -> dict:
    """Convert CNN output to flat SceneParams for API."""
    if isinstance(continuous, torch.Tensor):
        continuous = continuous.detach().cpu().tolist()
    params = denormalize_continuous(continuous)

    for name, logits in categorical.items():
        idx = int(logits.argmax().item() if isinstance(logits, torch.Tensor) else logits)
        params[name] = CATEGORICAL[name][idx]

    if isinstance(boolean, torch.Tensor):
        bool_vals = boolean.detach().cpu().tolist()
    else:
        bool_vals = boolean if isinstance(boolean, list) else [boolean]
    for i, name in enumerate(BOOLEANS):
        params[name] = bool_vals[i] > 0.5 if i < len(bool_vals) else False

    params["debug"] = False
    params["color"] = "#FFFDDD"
    params["position"] = {"x": 0, "y": 0}
    params["rotation"] = 0
    params["scale"] = 1

    layers = []
    if layer_count is not None and layer_params is not None:
        n_layers = int(
            layer_count.argmax().item()
            if isinstance(layer_count, torch.Tensor)
            else layer_count
        )
        for i in range(min(n_layers, len(layer_params))):
            lp = layer_params[i]
            if isinstance(lp, torch.Tensor):
                lp = lp.detach().cpu().tolist()

            presence = None
            if layer_presence is not None and i < len(layer_presence):
                lpr = layer_presence[i]
                if isinstance(lpr, torch.Tensor):
                    lpr = lpr.detach().cpu().tolist()
                presence = [v > 0.5 for v in lpr]

            geo_idx = None
            if layer_geos is not None and i < len(layer_geos):
                lg = layer_geos[i]
                geo_idx = int(lg.argmax().item() if isinstance(lg, torch.Tensor) else lg)

            layers.append(denormalize_layer(lp, presence, geo_idx))

    params["layers"] = layers
    return params
