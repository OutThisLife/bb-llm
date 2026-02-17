"""Shared utilities: param schema, random generation, format converters, param encoding."""

import math
import random

# numpy/torch imported lazily in functions that need them

clamp = lambda v, lo, hi: max(lo, min(hi, v))

# ============================================================================
# Color palette
# ============================================================================

COLORS = [
    "#FFFDDD", "#FFFEF0", "#FFFFF0", "#FFF8DC", "#FFEFD5",
    "#FFE4B5", "#F5DEB3", "#EEE8AA", "#E8D9A0", "#D4C896",
]

# ============================================================================
# Parameter schema (single source of truth)
# ============================================================================

SCHEMA = {
    "Scalars.repetitions": {
        "type": "int", "range": (1, 500), "bias_min": 15,
    },
    "Scalars.alphaFactor": {"type": "float", "range": (0, 1)},
    "Scalars.scaleFactor": {"type": "float", "range": (0, 2)},
    "Scalars.rotationFactor": {"type": "float", "range": (-1, 1)},
    "Scalars.stepFactor": {"type": "float", "range": (0.02, 2)},
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
    "Spatial.xStep": {"type": "float", "range": (-2, 2), "center_bias": 0.7},
    "Spatial.yStep": {"type": "float", "range": (-2, 2), "center_bias": 0.7},
    "Spatial.origin": {
        "type": "cat",
        "options": [
            "center", "top-center", "bottom-center",
            "top-left", "top-right", "bottom-left", "bottom-right",
        ],
    },
    "Scene.scale": {"type": "float", "range": (0.85, 1.5)},
    "Scene.rotation": {"type": "float", "fixed": 0},
    "Scene.position": {"type": "obj", "fixed": {"x": 0, "y": 0}},
    "Scene.debug": {"type": "bool", "fixed": False},
    "Scene.transform": {"type": "bool", "fixed": False},
    "Element.geometry": {
        "type": "cat",
        "options": [
            "ring", "bar", "line", "arch", "u",
            "spiral", "wave", "infinity", "square", "roundedRect",
        ],
    },
    "Element.geoWidth": {"type": "float", "range": (0.001, 0.1), "default": 0.041},
    "Element.gradientAngle": {"type": "float", "range": (-3.14159, 3.14159)},
    "Element.color": {"type": "color"},
    "Noise.enabled": {"type": "bool", "prob": 0.35},
    "Noise.density": {"type": "float", "range": (0, 1)},
    "Noise.opacity": {"type": "float", "range": (0, 0.4)},
    "Noise.size": {"type": "float", "range": (0.1, 0.5)},
    "Dither.enabled": {"type": "bool", "prob": 0.35},
    "Dither.type": {
        "type": "cat",
        "options": ["bayer", "noise", "halftone"],
    },
    "Dither.matrix": {"type": "cat", "options": [2, 4, 8]},
    "Dither.colors": {"type": "int", "range": (8, 32)},
    "Dither.strength": {"type": "float", "range": (0.2, 0.8)},
    "Dither.scale": {"type": "float", "range": (1, 8)},
    "Dither.bias": {"type": "float", "range": (0, 0.5)},
    "Dither.grayscale": {"type": "bool", "prob": 0.1},
}

LAYER_SCHEMA = {
    "rotation": {"type": "float", "range": (-3.14159, 3.14159)},
    "position": {"type": "obj", "axes": {"x": (-2, 2), "y": (-2, 2)}},
    "scale": {"type": "obj", "axes": {"x": (0.8, 1.2), "y": (0.8, 1.2)}},
    "stepFactor": {"type": "float", "range": (0, 2), "optional": 0.3},
    "alphaFactor": {"type": "float", "range": (0, 1), "optional": 0.3},
    "scaleFactor": {"type": "float", "range": (0, 2), "optional": 0.3},
    "rotationFactor": {"type": "float", "range": (-1, 1), "optional": 0.3},
    "color": {"type": "color", "optional": 0.2},
    "geoWidth": {"type": "float", "range": (0.001, 0.1), "optional": 0.3},
    "geometry": {
        "type": "cat",
        "options": SCHEMA["Element.geometry"]["options"],
        "optional": 0.4,
    },
}

# ============================================================================
# Random param generation
# ============================================================================


def _random_value(spec):
    t = spec["type"]
    if t == "int":
        lo, hi = spec["range"]
        if "bias_min" in spec and random.random() < 0.8:
            return random.randint(spec["bias_min"], hi)
        return random.randint(lo, hi)
    if t == "float":
        lo, hi = spec["range"]
        if "center_bias" in spec and random.random() < spec["center_bias"]:
            mid = (lo + hi) / 2
            spread = (hi - lo) / 4
            return round(max(lo, min(hi, random.gauss(mid, spread))), 4)
        return round(random.uniform(lo, hi), 4)
    if t == "bool":
        return random.random() < spec.get("prob", 0.5)
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
                val = {"x": clamp(val["x"], -1, 1), "y": clamp(val["y"], -1, 1)}
            p[f"{pre}{name}"] = val
    return p


def _pick_mirror_axis():
    """Randomly choose mirror axis: flip x, flip y, or flip both."""
    return random.choice(["x", "y", "xy"])


def _flip_scale(scale, axis):
    """Negate scale on the given axis."""
    x, y = scale.get("x", 1), scale.get("y", 1)
    if "x" in axis:
        x *= -1
    if "y" in axis:
        y *= -1
    return {"x": x, "y": y}


def _mirror_base(i, axis="x"):
    """Fresh mirror layer: only position/rotation/scale (rest inherits from base)."""
    pre = f"Groups.g{i}.g{i}-"
    return {
        f"{pre}position": {"x": 0, "y": 0},
        f"{pre}rotation": 0,
        f"{pre}scale": _flip_scale({"x": 1, "y": 1}, axis),
    }


def _mirror_layer(src, src_idx, dst_idx, axis="x"):
    """Clone src layer into dst layer index, negate scale on axis."""
    src_pre = f"Groups.g{src_idx}.g{src_idx}-"
    dst_pre = f"Groups.g{dst_idx}.g{dst_idx}-"
    p = {}
    for k, v in src.items():
        if k.startswith(src_pre):
            suffix = k[len(src_pre):]
            p[f"{dst_pre}{suffix}"] = v
    src_scale = src.get(f"{src_pre}scale", {"x": 1, "y": 1})
    p[f"{dst_pre}scale"] = _flip_scale(src_scale, axis)
    return p


_strat_counters = {}


def random_params(n_layers=None, stratified=False):
    """Generate prefixed params. stratified=True cycles through categorical combos."""
    symmetric = random.random() < 0.7

    params = {}
    for name, spec in SCHEMA.items():
        if "fixed" in spec:
            params[name] = spec["fixed"]
        elif stratified and spec["type"] == "cat":
            opts = spec["options"]
            idx = _strat_counters.get(name, 0)
            params[name] = opts[idx % len(opts)]
            _strat_counters[name] = idx + 1
        else:
            params[name] = _random_value(spec)

    if symmetric:
        # Harmonic overrides tuned to ref distributions
        params["Scalars.repetitions"] = random.randint(40, 250)
        params["Scalars.scaleFactor"] = round(random.uniform(0.92, 1.08), 4)
        params["Scalars.stepFactor"] = round(random.uniform(0.01, 0.6), 4)
        params["Scalars.rotationFactor"] = round(random.uniform(-0.15, 0.15), 4)
        params["Element.geometry"] = random.choices(
            ["u", "ring", "arch", "bar", "square", "wave", "infinity", "line"],
            weights=[27, 21, 14, 10, 10, 6, 5, 4],
        )[0]
        params["Scalars.rotationProgression"] = random.choices(
            ["sine", "fibonacci", "golden-angle", "linear"],
            weights=[53, 24, 13, 10],
        )[0]

        axis = _pick_mirror_axis()
        n_pairs = (
            n_layers // 2 + 1
            if n_layers is not None
            else random.choice([1, 2, 3])
        )
        if n_pairs >= 1:
            params.update(_mirror_base(0, axis))
        for pair in range(1, n_pairs):
            orig_idx = pair * 2 - 1
            mirror_idx = pair * 2
            orig = random_layer(orig_idx)
            params.update(orig)
            params.update(_mirror_layer(orig, orig_idx, mirror_idx, axis))
    else:
        n = (
            n_layers
            if n_layers is not None
            else random.choices(range(0, 6), weights=[3, 4, 3, 2, 1, 1])[0]
        )
        for i in range(n):
            params.update(random_layer(i))

    return params


# ============================================================================
# Format converters (prefixed ↔ flat)
# ============================================================================

PREFIX_MAP = {
    "Element.geometry": "geometry",
    "Element.geoWidth": "geoWidth",
    "Element.gradientAngle": "gradientAngle",
    "Element.color": "color",
    "Scalars.repetitions": "repetitions",
    "Scalars.alphaFactor": "alphaFactor",
    "Scalars.scaleFactor": "scaleFactor",
    "Scalars.rotationFactor": "rotationFactor",
    "Scalars.stepFactor": "stepFactor",
    "Scalars.scaleProgression": "scaleProgression",
    "Scalars.rotationProgression": "rotationProgression",
    "Scalars.alphaProgression": "alphaProgression",
    "Scalars.positionProgression": "positionProgression",
    "Scalars.positionCoupled": "positionCoupled",
    "Spatial.origin": "origin",
    "Spatial.xStep": "xStep",
    "Spatial.yStep": "yStep",
    "Scene.debug": "debug",
    "Scene.position": "position",
    "Scene.rotation": "rotation",
    "Scene.scale": "scale",
    "Noise.enabled": "noiseEnabled",
    "Noise.density": "noiseDensity",
    "Noise.opacity": "noiseOpacity",
    "Noise.size": "noiseSize",
    "Dither.enabled": "ditherEnabled",
    "Dither.type": "ditherType",
    "Dither.matrix": "ditherMatrix",
    "Dither.colors": "ditherColors",
    "Dither.strength": "ditherStrength",
    "Dither.scale": "ditherScale",
    "Dither.bias": "ditherBias",
    "Dither.grayscale": "ditherGrayscale",
}

# Flat dither keys → nested dither object keys (for API SceneParams)
DITHER_NEST = {
    "ditherEnabled": "enabled",
    "ditherType": "type",
    "ditherMatrix": "matrix",
    "ditherColors": "colors",
    "ditherStrength": "strength",
    "ditherScale": "scale",
    "ditherBias": "bias",
    "ditherGrayscale": "grayscale",
}

FLAT_MAP = {v: k for k, v in PREFIX_MAP.items()}


def to_scene_params(prefixed: dict) -> dict:
    """Prefixed format → flat SceneParams (model-friendly, no dither nesting)."""
    result = {}
    for pkey, fkey in PREFIX_MAP.items():
        if pkey in prefixed:
            result[fkey] = prefixed[pkey]

    layers = []
    i = 0
    while True:
        pre = f"Groups.g{i}.g{i}-"
        layer_keys = {k: v for k, v in prefixed.items() if k.startswith(pre)}
        if not layer_keys:
            break
        layers.append({k.replace(pre, ""): v for k, v in layer_keys.items()})
        i += 1

    result["layers"] = layers
    return result


def to_prefixed(flat: dict) -> dict:
    """Flat SceneParams → prefixed format."""
    result = {}
    for fkey, pkey in FLAT_MAP.items():
        if fkey in flat and fkey != "layers":
            result[pkey] = flat[fkey]

    # Extract nested dither → prefixed keys
    if "dither" in flat:
        for flat_key, nested_key in DITHER_NEST.items():
            pkey = FLAT_MAP.get(flat_key)
            if pkey and nested_key in flat["dither"]:
                result[pkey] = flat["dither"][nested_key]

    for i, layer in enumerate(flat.get("layers", [])):
        pre = f"Groups.g{i}.g{i}-"
        for k, v in layer.items():
            result[f"{pre}{k}"] = v

    return result


# ============================================================================
# Derived constants (from SCHEMA — single source of truth)
# ============================================================================

MAX_LAYERS = 5

# Continuous params predicted by models (flat keys, nested objects flattened)
CONTINUOUS_KEYS = [
    "repetitions",
    "alphaFactor",
    "scaleFactor",
    "rotationFactor",
    "stepFactor",
    "xStep",
    "yStep",
    "scale",
    "geoWidth",
    "gradientAngle",
    "noiseDensity",
    "noiseOpacity",
    "noiseSize",
    "ditherColors",
    "ditherStrength",
    "ditherScale",
    "ditherBias",
]

CONTINUOUS_RANGES = {
    "repetitions": (1, 500),
    "alphaFactor": (0, 1),
    "scaleFactor": (0, 3),
    "rotationFactor": (-1, 1),
    "stepFactor": (0, 2),
    "xStep": (-3, 3),
    "yStep": (-3, 3),
    "scale": (0.85, 1.5),
    "geoWidth": (0.001, 0.1),
    "gradientAngle": (-3.14159, 3.14159),
    "noiseDensity": (0, 1),
    "noiseOpacity": (0, 0.3),
    "noiseSize": (0.1, 0.55),
    "ditherColors": (8, 32),
    "ditherStrength": (0.2, 0.8),
    "ditherScale": (1, 8),
    "ditherBias": (0, 0.5),
}

# Per-layer continuous params (for each of MAX_LAYERS)
LAYER_CONTINUOUS_KEYS = [
    "position.x",
    "position.y",
    "rotation",
    "scale.x",
    "scale.y",
    "stepFactor",
    "alphaFactor",
    "scaleFactor",
    "rotationFactor",
    "geoWidth",
]

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
    "geoWidth": (0.001, 0.1),
}

# Per-layer optional params (presence predicted separately)
LAYER_OPTIONALS = ["stepFactor", "alphaFactor", "scaleFactor", "rotationFactor", "geoWidth", "geometry"]

# Categorical params with their options
CATEGORICAL_KEYS = {
    "geometry": [
        "ring", "bar", "line", "arch", "u",
        "spiral", "wave", "infinity", "square", "roundedRect",
    ],
    "scaleProgression": [
        "linear", "exponential", "additive", "fibonacci", "golden", "sine",
    ],
    "rotationProgression": ["linear", "golden-angle", "fibonacci", "sine"],
    "alphaProgression": ["exponential", "linear", "inverse"],
    "positionProgression": ["index", "scale"],
    "origin": [
        "center", "top-center", "bottom-center",
        "top-left", "top-right", "bottom-left", "bottom-right",
    ],
    "ditherType": ["bayer", "noise", "halftone"],
    "ditherMatrix": [2, 4, 8],
}

# Booleans
BOOLEAN_KEYS = ["positionCoupled", "noiseEnabled", "ditherEnabled", "ditherGrayscale"]
TASTE_FEATURE_DIM = (
    len(CONTINUOUS_KEYS)
    + len(BOOLEAN_KEYS)
    + sum(len(v) for v in CATEGORICAL_KEYS.values())
    + 1  # layer count
    + MAX_LAYERS
    * (
        len(LAYER_CONTINUOUS_KEYS)
        + len(LAYER_OPTIONALS)
        + len(CATEGORICAL_KEYS["geometry"])
    )
)


# ============================================================================
# Normalization helpers
# ============================================================================


def normalize_continuous(params: dict) -> list[float]:
    """Normalize continuous params to [0, 1] for training."""
    values = []
    for name in CONTINUOUS_KEYS:
        v = params.get(name, 0)
        lo, hi = CONTINUOUS_RANGES[name]
        normalized = (v - lo) / (hi - lo) if hi > lo else 0.5
        values.append(max(0, min(1, normalized)))
    return values


def denormalize_continuous(values: list[float]) -> dict:
    """Denormalize [0, 1] values back to original ranges."""
    result = {}
    for i, name in enumerate(CONTINUOUS_KEYS):
        lo, hi = CONTINUOUS_RANGES[name]
        v = lo + values[i] * (hi - lo)
        if name in ("repetitions", "ditherColors"):
            result[name] = round(v)
        else:
            result[name] = round(v, 4)
    return result


def normalize_layer(layer: dict) -> list[float]:
    """Normalize a single layer's params to [0, 1]."""
    values = []
    for name in LAYER_CONTINUOUS_KEYS:
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
            lo, hi = LAYER_RANGES[name]
            v = layer.get(name, (lo + hi) / 2)
        lo, hi = LAYER_RANGES[name]
        normalized = (v - lo) / (hi - lo) if hi > lo else 0.5
        values.append(max(0, min(1, normalized)))
    return values


def denormalize_layer(values: list[float], presence=None, geo_idx=None) -> dict:
    """Denormalize layer params back to original ranges."""
    layer = {"position": {}, "scale": {}}
    for i, name in enumerate(LAYER_CONTINUOUS_KEYS):
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

    if geo_idx is not None and presence is not None:
        geo_i = LAYER_OPTIONALS.index("geometry")
        if presence[geo_i]:
            layer["geometry"] = CATEGORICAL_KEYS["geometry"][geo_idx]

    return layer


def get_layer_presence(layer: dict) -> list[bool]:
    """Which optional params are present in this layer."""
    return [name in layer for name in LAYER_OPTIONALS]


def get_layer_geometry_idx(layer: dict) -> int:
    """Get geometry index for layer (0 if absent)."""
    geo = layer.get("geometry")
    if geo and geo in CATEGORICAL_KEYS["geometry"]:
        return CATEGORICAL_KEYS["geometry"].index(geo)
    return 0


def _one_hot(index: int, size: int) -> list[float]:
    vec = [0.0] * size
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def encode_taste_features(params: dict) -> list[float]:
    """Flat/prefixed params -> fixed-length taste feature vector."""
    flat = (
        to_scene_params(params)
        if "Scalars.repetitions" in params or "Element.geometry" in params
        else params
    )
    feat = []

    feat.extend(normalize_continuous(flat))
    feat.extend(1.0 if flat.get(k, False) else 0.0 for k in BOOLEAN_KEYS)

    for key, options in CATEGORICAL_KEYS.items():
        val = flat.get(key)
        idx = options.index(val) if val in options else -1
        feat.extend(_one_hot(idx, len(options)))

    layers = flat.get("layers", [])
    feat.append(min(len(layers), MAX_LAYERS) / MAX_LAYERS)

    geo_dim = len(CATEGORICAL_KEYS["geometry"])
    for i in range(MAX_LAYERS):
        if i < len(layers):
            layer = layers[i]
            feat.extend(normalize_layer(layer))
            feat.extend(1.0 if p else 0.0 for p in get_layer_presence(layer))
            feat.extend(_one_hot(get_layer_geometry_idx(layer), geo_dim))
        else:
            feat.extend([0.5] * len(LAYER_CONTINUOUS_KEYS))
            feat.extend([0.0] * len(LAYER_OPTIONALS))
            feat.extend([0.0] * geo_dim)

    return feat


# ============================================================================
# Param vector encoding/decoding (for CMA-ES and forward model)
# ============================================================================


def encode_params(flat_params: dict, n_layers: int):
    """Flat SceneParams → [0,1] normalized vector.

    Layout: [continuous(10) | booleans(2) | layer_0(9) | ... | layer_n(9)]
    """
    import numpy as np

    vec = []

    # Base continuous
    for name in CONTINUOUS_KEYS:
        v = flat_params.get(name, 0)
        lo, hi = CONTINUOUS_RANGES[name]
        vec.append(np.clip((v - lo) / (hi - lo), 0, 1))

    # Booleans as float
    for name in BOOLEAN_KEYS:
        vec.append(1.0 if flat_params.get(name, False) else 0.0)

    # Per-layer continuous
    layers = flat_params.get("layers", [])
    for i in range(n_layers):
        if i < len(layers):
            vec.extend(normalize_layer(layers[i]))
        else:
            vec.extend([0.5] * len(LAYER_CONTINUOUS_KEYS))

    return np.array(vec, dtype=np.float64)


def decode_params(
    vec,
    categoricals: dict,
    layer_geos: list,
    layer_presence: list,
    n_layers: int,
) -> dict:
    """[0,1] vector → flat SceneParams for API.

    Inverse of encode_params. Categoricals, layer geos, and presence flags
    are passed separately (they're discrete, not in the continuous vector).
    """
    # Base continuous
    base_vals = vec[: len(CONTINUOUS_KEYS)].tolist()
    params = denormalize_continuous(base_vals)

    # Categoricals (fixed, not from vector)
    for k, v in categoricals.items():
        params[k] = v

    # Booleans
    bool_offset = len(CONTINUOUS_KEYS)
    for i, name in enumerate(BOOLEAN_KEYS):
        params[name] = bool(vec[bool_offset + i] > 0.5)

    # Fixed fields
    params["debug"] = False
    params["color"] = "#FFFDDD"
    params["position"] = {"x": 0, "y": 0}
    params["rotation"] = 0

    # Layers
    layers = []
    layer_offset = bool_offset + len(BOOLEAN_KEYS)
    n_layer_dims = len(LAYER_CONTINUOUS_KEYS)
    for i in range(n_layers):
        lv = vec[layer_offset + i * n_layer_dims : layer_offset + (i + 1) * n_layer_dims].tolist()
        geo_idx = None
        if i < len(layer_geos) and layer_geos[i] is not None:
            geos = CATEGORICAL_KEYS["geometry"]
            geo_idx = geos.index(layer_geos[i]) if layer_geos[i] in geos else 0
        presence = layer_presence[i] if i < len(layer_presence) else None
        layers.append(denormalize_layer(lv, presence, geo_idx))

    params["layers"] = layers
    return params


def reconstruct_params(
    continuous, categorical, boolean,
    layer_count=None, layer_params=None,
    layer_presence=None, layer_geos=None,
) -> dict:
    """Convert CNN output tensors to flat SceneParams for API."""
    import torch

    if isinstance(continuous, torch.Tensor):
        continuous = continuous.detach().cpu().tolist()
    params = denormalize_continuous(continuous)

    for name, logits in categorical.items():
        idx = int(logits.argmax().item() if isinstance(logits, torch.Tensor) else logits)
        params[name] = CATEGORICAL_KEYS[name][idx]

    if isinstance(boolean, torch.Tensor):
        bool_vals = boolean.detach().cpu().tolist()
    else:
        bool_vals = boolean if isinstance(boolean, list) else [boolean]
    for i, name in enumerate(BOOLEAN_KEYS):
        params[name] = bool_vals[i] > 0.0 if i < len(bool_vals) else False

    params["debug"] = False
    params["color"] = "#FFFDDD"
    params["position"] = {"x": 0, "y": 0}
    params["rotation"] = 0

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
                presence = [v > 0.0 for v in lpr]

            geo_idx = None
            if layer_geos is not None and i < len(layer_geos):
                lg = layer_geos[i]
                geo_idx = int(lg.argmax().item() if isinstance(lg, torch.Tensor) else lg)

            layers.append(denormalize_layer(lp, presence, geo_idx))

    params["layers"] = layers
    return params


# ============================================================================
# Device detection
# ============================================================================


def load_state_dict_compat(sd):
    """Strip torch.compile _orig_mod. prefix if present."""
    if not sd or not next(iter(sd)).startswith("_orig_mod."):
        return sd
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_batch_size(model_vram_mb=200):
    """Pick batch size based on available VRAM. Conservative estimate."""
    import torch

    if not torch.cuda.is_available():
        return 16
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Reserve ~2GB for model + LPIPS + overhead, rest for batches
    # ~50MB per sample at 256×256 with activations
    usable = max(1, vram_gb - 2 - model_vram_mb / 1000)
    return min(128, max(16, int(usable * 1000 / 50)))


def auto_workers():
    """Pick num_workers for DataLoader based on CPU count."""
    import os
    cores = os.cpu_count() or 4
    return min(cores, 12)
