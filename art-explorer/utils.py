"""Shared utilities: param schema, random generation, format converters."""

import math
import random

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
    "Spatial.xStep": {"type": "float", "range": (-2, 2)},
    "Spatial.yStep": {"type": "float", "range": (-2, 2)},
    "Spatial.origin": {
        "type": "cat",
        "options": [
            "center", "top-center", "bottom-center",
            "top-left", "top-right", "bottom-left", "bottom-right",
        ],
    },
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
    "Element.color": {"type": "color"},
    "Noise.enabled": {"type": "bool", "default": False},
    "Noise.density": {"type": "float", "range": (0, 1), "default": 0.11},
    "Noise.opacity": {"type": "float", "range": (0, 1), "default": 0.55},
    "Noise.size": {"type": "float", "range": (0.1, 10), "default": 1.0},
}

LAYER_SCHEMA = {
    "rotation": {"type": "float", "range": (-3.14159, 3.14159)},
    "position": {"type": "obj", "axes": {"x": (-2, 2), "y": (-2, 2)}},
    "scale": {"type": "obj", "axes": {"x": (-2, 2), "y": (-2, 2)}},
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
                val = {"x": clamp(val["x"], -1, 1), "y": clamp(val["y"], -1, 1)}
            p[f"{pre}{name}"] = val
    return p


_strat_counters = {}


def random_params(n_layers=None, stratified=False):
    """Generate prefixed params. stratified=True cycles through categorical combos."""
    params = {}
    for name, spec in SCHEMA.items():
        if "fixed" in spec:
            params[name] = spec["fixed"]
        elif stratified and spec["type"] == "cat":
            # Round-robin through options
            opts = spec["options"]
            idx = _strat_counters.get(name, 0)
            params[name] = opts[idx % len(opts)]
            _strat_counters[name] = idx + 1
        else:
            params[name] = _random_value(spec)

    # Prevent exponential blowout: scaleFactor^repetitions < 1000
    sf = params.get("Scalars.scaleFactor", 1)
    reps = params.get("Scalars.repetitions", 65)
    if sf > 1.01:
        max_reps = int(6.9 / math.log(sf))
        params["Scalars.repetitions"] = min(reps, max(10, max_reps))

    n = (
        n_layers
        if n_layers is not None
        else random.choices(range(1, 6), weights=[4, 3, 2, 1, 1])[0]
    )
    for i in range(n):
        params.update(random_layer(i))

    return params


# ============================================================================
# Format converters (prefixed ↔ flat)
# ============================================================================

PREFIX_MAP = {
    "Element.geometry": "geometry",
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
}

FLAT_MAP = {v: k for k, v in PREFIX_MAP.items()}


def to_scene_params(prefixed: dict) -> dict:
    """Prefixed format → flat SceneParams for API."""
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

    for i, layer in enumerate(flat.get("layers", [])):
        pre = f"Groups.g{i}.g{i}-"
        for k, v in layer.items():
            result[f"{pre}{k}"] = v

    return result
