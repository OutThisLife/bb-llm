"""Shared utilities + format converters."""

clamp = lambda v, lo, hi: max(lo, min(hi, v))

# Single source of truth: prefixed key → flat key
# Reverse mapping is derived automatically
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
    # Noise effect
    "Noise.enabled": "noiseEnabled",
    "Noise.density": "noiseDensity",
    "Noise.opacity": "noiseOpacity",
    "Noise.size": "noiseSize",
}

# Reverse: flat key → prefixed key
FLAT_MAP = {v: k for k, v in PREFIX_MAP.items()}


def to_scene_params(prefixed: dict) -> dict:
    """explore.py format → SceneParams for API."""
    # Base params: strip prefixes
    result = {}
    for pkey, fkey in PREFIX_MAP.items():
        if pkey in prefixed:
            result[fkey] = prefixed[pkey]

    # Layers: Groups.g{i}.g{i}-{key} → layers[i].{key}
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
    """SceneParams → explore.py format (for CMA-ES integration)."""
    # Base params: add prefixes
    result = {}
    for fkey, pkey in FLAT_MAP.items():
        if fkey in flat and fkey != "layers":
            result[pkey] = flat[fkey]

    # Layers: layers[i].{key} → Groups.g{i}.g{i}-{key}
    for i, layer in enumerate(flat.get("layers", [])):
        pre = f"Groups.g{i}.g{i}-"
        for k, v in layer.items():
            result[f"{pre}{k}"] = v

    return result
