"""Shared utilities: param schema, random generation, format converters."""

import random

clamp = lambda v, lo, hi: max(lo, min(hi, v))

COLORS = [
    "#FFFDDD", "#FFFEF0", "#FFFFF0", "#FFF8DC", "#FFEFD5",
    "#FFE4B5", "#F5DEB3", "#EEE8AA", "#E8D9A0", "#D4C896",
    "#dadada", "#adadad",
]


SCHEMA = {
    "Scalars.repetitions": {"type": "int", "range": (1, 500), "bias_min": 80},
    "Scalars.alphaFactor": {"type": "float", "range": (0, 1), "power": 0.4},
    "Scalars.scaleFactor": {"type": "float", "range": (0, 2), "center_bias": 0.8},
    "Scalars.rotationFactor": {"type": "float", "range": (-1, 1), "center_bias": 0.8},
    "Scalars.stepFactor": {"type": "float", "range": (0.02, 2), "power": 2.0},
    "Scalars.positionCoupled": {"type": "bool", "prob": 0.7},
    "Scalars.scaleProgression": {
        "type": "cat",
        "options": ["linear", "exponential", "additive", "fibonacci", "golden", "sine"],
        "weights": [1, 4, 1, 2, 2, 4],
    },
    "Scalars.rotationProgression": {
        "type": "cat",
        "options": ["linear", "golden-angle", "fibonacci", "sine"],
        "weights": [1, 3, 2, 3],
    },
    "Scalars.alphaProgression": {
        "type": "cat",
        "options": ["exponential", "linear", "inverse"],
        "weights": [4, 2, 1],
    },
    "Scalars.positionProgression": {"type": "cat", "options": ["index", "scale"]},
    "Spatial.xStep": {"type": "float", "range": (-2, 2), "center_bias": 0.7},
    "Spatial.yStep": {"type": "float", "range": (-2, 2), "center_bias": 0.7},
    "Spatial.origin": {
        "type": "cat",
        "options": ["center", "top-center", "bottom-center", "top-left", "top-right", "bottom-left", "bottom-right"],
        "weights": [6, 2, 2, 1, 1, 1, 1],
    },
    "Scene.scale": {"type": "float", "range": (0.05, 2.0)},
    "Scene.rotation": {"type": "float", "range": (-3.14159, 3.14159), "center_bias": 0.8},
    "Scene.position": {"type": "obj", "fixed": {"x": 0, "y": -0.5}},
    "Scene.debug": {"type": "bool", "fixed": False},
    "Scene.transform": {"type": "bool", "fixed": False},
    "Element.geometry": {
        "type": "cat",
        "options": ["ring", "bar", "line", "arch", "u", "spiral", "wave", "infinity", "square", "roundedRect"],
        "weights": [5, 1, 1, 2, 1, 1, 1, 5, 1, 5],
    },
    "Element.geoWidth": {"type": "float", "range": (0.001, 0.1), "default": 0.041, "power": 2.5},
    "Element.startAngle": {"type": "float", "range": (-3.14159, 3.14159)},
    "Element.gradientAngle": {"type": "float", "range": (-3.14159, 3.14159)},
    "Element.gradientRange": {"type": "interval", "range": (-1, 2), "default": [0.2, 1.0]},
    "Element.color": {"type": "color"},
    "Noise.enabled": {"type": "bool", "prob": 0.12},
    "Noise.density": {"type": "float", "range": (0, 1)},
    "Noise.opacity": {"type": "float", "range": (0, 0.3)},
    "Noise.size": {"type": "float", "range": (0.1, 0.55)},
    "CRT.enabled": {"type": "bool", "prob": 0.12},
    "CRT.bleed": {"type": "float", "range": (0, 1)},
    "CRT.bloom": {"type": "float", "range": (0, 1)},
    "CRT.brightness": {"type": "float", "range": (0.5, 2)},
    "CRT.mask": {
        "type": "cat",
        "options": ["none", "shadow", "grille", "stretched", "vga"],
        "weights": [1, 1, 5, 1, 1],
    },
    "CRT.maskStrength": {"type": "float", "range": (0, 1)},
    "CRT.scale": {"type": "float", "range": (1, 8)},
    "CRT.scanlines": {"type": "float", "range": (0, 1)},
    "CRT.warp": {"type": "float", "range": (0, 0.1)},
}

LAYER_SCHEMA = {
    "rotation": {"type": "float", "range": (-3.14159, 3.14159)},
    "position": {"type": "obj", "axes": {"x": (-2, 2), "y": (-2, 2)}},
    "scale": {"type": "obj", "axes": {"x": (-2, 2), "y": (-2, 2)}},
    "stepFactor": {"type": "float", "range": (0, 2), "optional": 0.3, "power": 2.0},
    "alphaFactor": {"type": "float", "range": (0, 1), "optional": 0.3, "power": 0.4},
    "scaleFactor": {"type": "float", "range": (0, 2), "optional": 0.3, "center_bias": 0.8},
    "rotationFactor": {"type": "float", "range": (-1, 1), "optional": 0.3, "center_bias": 0.8},
    "color": {"type": "color", "optional": 0.2},
    "geoWidth": {"type": "float", "range": (0.001, 0.1), "optional": 0.3, "power": 2.5},
    "startAngle": {"type": "float", "range": (-3.14159, 3.14159), "optional": 0.2},
    "geometry": {
        "type": "cat",
        "options": SCHEMA["Element.geometry"]["options"],
        "weights": SCHEMA["Element.geometry"]["weights"],
        "optional": 0.4,
    },
}



_strat_counters = {}


def _strat_pick(name, options):
    idx = _strat_counters.get(name, 0)
    _strat_counters[name] = idx + 1
    return options[idx % len(options)]


def _strat_gate(name, prob):
    if prob >= 1:
        return True
    slots = max(1, min(9, int(round(prob * 10))))
    return _strat_pick(name, [True] * slots + [False] * (10 - slots))


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
            mid, spread = (lo + hi) / 2, (hi - lo) / 4
            return round(clamp(random.gauss(mid, spread), lo, hi), 4)
        if "power" in spec:
            return round(lo + (hi - lo) * random.random() ** spec["power"], 4)
        return round(random.uniform(lo, hi), 4)

    if t == "bool":
        return random.random() < spec.get("prob", 0.5)

    if t == "cat":
        return random.choices(spec["options"], weights=spec.get("weights"))[0] if "weights" in spec else random.choice(spec["options"])

    if t == "color":
        return random.choice(COLORS)

    if t == "obj":
        if "axes" in spec:
            return {k: round(random.uniform(*v), 4) for k, v in spec["axes"].items()}
        return None

    if t == "interval":
        lo, hi = spec["range"]
        a, b = random.uniform(lo, hi), random.uniform(lo, hi)
        return [round(min(a, b), 4), round(max(a, b), 4)]

    return None


def _random_layer(i, stratified=False):
    pre = f"Groups.g{i}.g{i}-"
    p = {}

    for name, spec in LAYER_SCHEMA.items():
        prob = spec.get("optional", 1.0)
        include = _strat_gate(f"{pre}{name}:present", prob) if stratified else random.random() < prob
        if not include:
            continue

        if stratified and spec["type"] == "cat":
            val = _strat_pick(f"{pre}{name}:cat", spec["options"])
        elif stratified and spec["type"] == "bool":
            val = _strat_pick(f"{pre}{name}:bool", [True, False])
        else:
            val = _random_value(spec)

        if name == "position" and isinstance(val, dict):
            val = {"x": clamp(val["x"], -1, 1), "y": clamp(val["y"], -1, 1)}

        p[f"{pre}{name}"] = val

    return p


def _flip_scale(scale, axis):
    x, y = scale.get("x", 1), scale.get("y", 1)
    if "x" in axis:
        x *= -1
    if "y" in axis:
        y *= -1
    return {"x": x, "y": y}


def _mirror_base(i, axis="x"):
    pre = f"Groups.g{i}.g{i}-"
    return {
        f"{pre}position": {"x": 0, "y": 0},
        f"{pre}rotation": 0,
        f"{pre}scale": _flip_scale({"x": 1, "y": 1}, axis),
    }


def _mirror_layer(src, src_idx, dst_idx, axis="x"):
    src_pre = f"Groups.g{src_idx}.g{src_idx}-"
    dst_pre = f"Groups.g{dst_idx}.g{dst_idx}-"
    p = {f"{dst_pre}{k[len(src_pre):]}": v for k, v in src.items() if k.startswith(src_pre)}
    p[f"{dst_pre}scale"] = _flip_scale(src.get(f"{src_pre}scale", {"x": 1, "y": 1}), axis)
    return p


def random_params(n_layers=None, stratified=False):
    symmetric = _strat_pick("__symmetric__", [True, False]) if stratified else random.random() < 0.7

    params = {}
    for name, spec in SCHEMA.items():
        if "fixed" in spec:
            params[name] = spec["fixed"]
        elif stratified and spec["type"] == "cat":
            params[name] = _strat_pick(name, spec["options"])
        elif stratified and spec["type"] == "bool":
            params[name] = _strat_pick(name, [True, False])
        else:
            params[name] = _random_value(spec)

    if symmetric:
        axis = _strat_pick("__mirror_axis__", ["x", "y", "xy"]) if stratified else random.choice(["x", "y", "xy"])

        if n_layers is not None:
            use_center, n_pairs = True, n_layers // 2 + 1
        elif stratified:
            use_center = _strat_pick("__sym_center__", [True, False])
            n_pairs = _strat_pick("__n_pairs__", [1, 2, 3] if use_center else [1, 2])
        else:
            use_center, n_pairs = True, random.choice([1, 2, 3])

        if use_center and n_pairs >= 1:
            params.update(_mirror_base(0, axis))

        start = 1 if use_center else 0
        for pair in range(start, n_pairs):
            oi = pair * 2 - 1 if use_center else pair * 2
            mi = pair * 2 if use_center else pair * 2 + 1
            orig = _random_layer(oi, stratified=stratified)
            params.update(orig)
            params.update(_mirror_layer(orig, oi, mi, axis))
    else:
        n = (
            n_layers if n_layers is not None
            else _strat_pick("__layer_count__", [0, 1, 2, 3, 4, 5]) if stratified
            else random.choices(range(6), weights=[3, 4, 3, 2, 1, 1])[0]
        )
        for i in range(n):
            params.update(_random_layer(i, stratified=stratified))

    return params



PREFIX_MAP = {
    "Element.geometry": "geometry",
    "Element.geoWidth": "geoWidth",
    "Element.startAngle": "startAngle",
    "Element.gradientAngle": "gradientAngle",
    "Element.gradientRange": "gradientRange",
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
    "CRT.enabled": "crtEnabled",
    "CRT.bleed": "crtBleed",
    "CRT.bloom": "crtBloom",
    "CRT.brightness": "crtBrightness",
    "CRT.mask": "crtMask",
    "CRT.maskStrength": "crtMaskStrength",
    "CRT.scale": "crtScale",
    "CRT.scanlines": "crtScanlines",
    "CRT.warp": "crtWarp",
}

FLAT_MAP = {v: k for k, v in PREFIX_MAP.items()}


def _ulv(v):
    return v["value"] if isinstance(v, dict) and "disabled" in v else v


def to_scene_params(prefixed: dict) -> dict:
    result = {fkey: _ulv(prefixed[pkey]) for pkey, fkey in PREFIX_MAP.items() if pkey in prefixed}

    layers, i = [], 0
    while True:
        pre = f"Groups.g{i}.g{i}-"
        lk = {k: v for k, v in prefixed.items() if k.startswith(pre)}
        if not lk:
            break
        layers.append({k.replace(pre, ""): _ulv(v) for k, v in lk.items()})
        i += 1

    result["layers"] = layers
    return result


def to_prefixed(flat: dict) -> dict:
    result = {pkey: flat[fkey] for fkey, pkey in FLAT_MAP.items() if fkey in flat and fkey != "layers"}

    for i, layer in enumerate(flat.get("layers", [])):
        pre = f"Groups.g{i}.g{i}-"
        for k, v in layer.items():
            result[f"{pre}{k}"] = v

    return result



MAX_LAYERS = 5

CONTINUOUS_KEYS = [
    "repetitions", "alphaFactor", "scaleFactor", "rotationFactor", "stepFactor",
    "xStep", "yStep", "scale", "rotation", "geoWidth", "startAngle",
    "gradientAngle", "gradientRangeMin", "gradientRangeMax",
    "noiseDensity", "noiseOpacity", "noiseSize",
    "crtBleed", "crtBloom", "crtBrightness", "crtMaskStrength", "crtScale", "crtScanlines", "crtWarp",
]

CONTINUOUS_RANGES = {
    "repetitions": (1, 500), "alphaFactor": (0, 1), "scaleFactor": (0, 2),
    "rotationFactor": (-1, 1), "stepFactor": (0, 2), "xStep": (-2, 2), "yStep": (-2, 2),
    "scale": (0.05, 2.0), "rotation": (-3.14159, 3.14159), "geoWidth": (0.001, 0.1),
    "startAngle": (-3.14159, 3.14159), "gradientAngle": (-3.14159, 3.14159),
    "gradientRangeMin": (-1, 2), "gradientRangeMax": (-1, 2),
    "noiseDensity": (0, 1), "noiseOpacity": (0, 0.3), "noiseSize": (0.1, 0.55),
    "crtBleed": (0, 1), "crtBloom": (0, 1), "crtBrightness": (0.5, 2),
    "crtMaskStrength": (0, 1), "crtScale": (1, 8), "crtScanlines": (0, 1), "crtWarp": (0, 0.1),
}

LAYER_CONTINUOUS_KEYS = [
    "position.x", "position.y", "rotation", "scale.x", "scale.y",
    "stepFactor", "alphaFactor", "scaleFactor", "rotationFactor", "geoWidth", "startAngle",
]

LAYER_RANGES = {
    "position.x": (-2, 2), "position.y": (-2, 2), "rotation": (-3.14159, 3.14159),
    "scale.x": (-2, 2), "scale.y": (-2, 2), "stepFactor": (0, 2),
    "alphaFactor": (0, 1), "scaleFactor": (0, 2), "rotationFactor": (-1, 1),
    "geoWidth": (0.001, 0.1), "startAngle": (-3.14159, 3.14159),
}

LAYER_OPTIONALS = ["stepFactor", "alphaFactor", "scaleFactor", "rotationFactor", "geoWidth", "startAngle", "geometry"]

CATEGORICAL_KEYS = {
    "geometry": ["ring", "bar", "line", "arch", "u", "spiral", "wave", "infinity", "square", "roundedRect"],
    "scaleProgression": ["linear", "exponential", "additive", "fibonacci", "golden", "sine"],
    "rotationProgression": ["linear", "golden-angle", "fibonacci", "sine"],
    "alphaProgression": ["exponential", "linear", "inverse"],
    "positionProgression": ["index", "scale"],
    "origin": ["center", "top-center", "bottom-center", "top-left", "top-right", "bottom-left", "bottom-right"],
    "crtMask": ["none", "shadow", "grille", "stretched", "vga"],
}

BOOLEAN_KEYS = ["positionCoupled", "noiseEnabled", "crtEnabled"]



def normalize_continuous(params: dict) -> list[float]:
    values = []
    for name in CONTINUOUS_KEYS:
        if name == "gradientRangeMin":
            gr = params.get("gradientRange", [0.2, 1.0])
            v = gr[0] if isinstance(gr, (list, tuple)) and len(gr) >= 1 else 0.2
        elif name == "gradientRangeMax":
            gr = params.get("gradientRange", [0.2, 1.0])
            v = gr[1] if isinstance(gr, (list, tuple)) and len(gr) >= 2 else 1.0
        else:
            v = params.get(name, 0)
        lo, hi = CONTINUOUS_RANGES[name]
        values.append(clamp((v - lo) / (hi - lo) if hi > lo else 0.5, 0, 1))
    return values


def denormalize_continuous(values: list[float]) -> dict:
    result = {}
    for i, name in enumerate(CONTINUOUS_KEYS):
        lo, hi = CONTINUOUS_RANGES[name]
        v = lo + values[i] * (hi - lo)
        result[name] = round(v) if name == "repetitions" else round(v, 4)

    result["gradientRange"] = [result.pop("gradientRangeMin", 0.2), result.pop("gradientRangeMax", 1.0)]
    return result


def normalize_layer(layer: dict) -> list[float]:
    values = []
    for name in LAYER_CONTINUOUS_KEYS:
        if name == "position.x":
            v = layer.get("position", {}).get("x", 0)
        elif name == "position.y":
            v = layer.get("position", {}).get("y", 0)
        elif name == "scale.x":
            v = layer.get("scale", {}).get("x", 1) if isinstance(layer.get("scale"), dict) else 1
        elif name == "scale.y":
            v = layer.get("scale", {}).get("y", 1) if isinstance(layer.get("scale"), dict) else 1
        elif name == "rotation":
            v = layer.get("rotation", 0)
        else:
            lo, hi = LAYER_RANGES[name]
            v = layer.get(name, (lo + hi) / 2)
        lo, hi = LAYER_RANGES[name]
        values.append(clamp((v - lo) / (hi - lo) if hi > lo else 0.5, 0, 1))
    return values


def denormalize_layer(values: list[float], presence=None, geo_idx=None) -> dict:
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
        gi = LAYER_OPTIONALS.index("geometry")
        if presence[gi]:
            layer["geometry"] = CATEGORICAL_KEYS["geometry"][geo_idx]

    return layer



def encode_params(flat_params: dict, n_layers: int):
    import numpy as np

    vec = list(normalize_continuous(flat_params))
    for name in BOOLEAN_KEYS:
        vec.append(1.0 if flat_params.get(name, False) else 0.0)

    for i in range(n_layers):
        layers = flat_params.get("layers", [])
        vec.extend(normalize_layer(layers[i]) if i < len(layers) else [0.5] * len(LAYER_CONTINUOUS_KEYS))

    return np.array(vec, dtype=np.float64)


def decode_params(vec, categoricals: dict, layer_geos: list, layer_presence: list, n_layers: int) -> dict:
    params = denormalize_continuous(vec[: len(CONTINUOUS_KEYS)].tolist())
    params.update(categoricals)

    bo = len(CONTINUOUS_KEYS)
    for i, name in enumerate(BOOLEAN_KEYS):
        params[name] = bool(vec[bo + i] > 0.5)

    params["debug"] = False
    params["color"] = "#efeddb"
    params["position"] = {"x": 0, "y": -0.5}

    lo = bo + len(BOOLEAN_KEYS)
    nd = len(LAYER_CONTINUOUS_KEYS)
    layers = []
    for i in range(n_layers):
        lv = vec[lo + i * nd : lo + (i + 1) * nd].tolist()
        gi = None
        if i < len(layer_geos) and layer_geos[i]:
            geos = CATEGORICAL_KEYS["geometry"]
            gi = geos.index(layer_geos[i]) if layer_geos[i] in geos else 0
        pr = layer_presence[i] if i < len(layer_presence) else None
        layers.append(denormalize_layer(lv, pr, gi))

    params["layers"] = layers
    return params



def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
