# Inverse Rendering + VLM Exploration Plan

## Context: Why This Architecture

### The Problem
VLM-guided exploration (CMA-ES + Qwen2-VL scoring) **works** for creative discovery - it found 9/10 outputs that match the reference aesthetic. But:
- Takes ~10hrs to find good regions
- Can't explain *why* something scored high (no param insight)
- Can't efficiently seed from external designs (Figma)

### The Solution: Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ CNN INVERSE MODEL                                               │
│   image → params                                                │
│   - Gives you coordinates in param space                        │
│   - Trained on synthetic (image, params) pairs                  │
│   - Fast inference (~10ms)                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ VLM EXPLORATION                                                 │
│   params → render → score → evolve                              │
│   - Explores neighborhood of CNN-derived seeds                  │
│   - Finds novel high-quality outputs                            │
│   - Has "taste" - judges aesthetics                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ CLOSED LOOP                                                     │
│   - VLM finds 9/10 output                                       │
│   - CNN extracts params → add to seed pool                      │
│   - Next exploration starts from better seeds                   │
└─────────────────────────────────────────────────────────────────┘
```

**CNN gives you the map. VLM explores the territory.**

### Key Format Mapping

Two key formats exist:

| Context | Format | Example |
|---------|--------|---------|
| `explore.py` | Leva-prefixed | `"Scalars.repetitions"`, `"Scene.position"` |
| `SceneParams` (API) | Flat | `"repetitions"`, `"position"` |

**Strategy:** CNN uses flat keys (same as API). Convert only at boundaries:

```python
def to_scene_params(prefixed: dict) -> dict:
    """explore.py format → SceneParams for API"""
    return {
        "geometry": prefixed.get("Element.geometry", "ring"),
        "color": prefixed.get("Element.color", "#FFFDDD"),
        "repetitions": prefixed.get("Scalars.repetitions", 65),
        "alphaFactor": prefixed.get("Scalars.alphaFactor", 0.65),
        "scaleFactor": prefixed.get("Scalars.scaleFactor", 1.05),
        "rotationFactor": prefixed.get("Scalars.rotationFactor", 0),
        "stepFactor": prefixed.get("Scalars.stepFactor", 0.02),
        "scaleProgression": prefixed.get("Scalars.scaleProgression", "exponential"),
        "rotationProgression": prefixed.get("Scalars.rotationProgression", "linear"),
        "alphaProgression": prefixed.get("Scalars.alphaProgression", "exponential"),
        "positionProgression": prefixed.get("Scalars.positionProgression", "index"),
        "positionCoupled": prefixed.get("Scalars.positionCoupled", True),
        "origin": prefixed.get("Spatial.origin", "center"),
        "xStep": prefixed.get("Spatial.xStep", 0),
        "yStep": prefixed.get("Spatial.yStep", 0),
        "debug": False,
        "position": prefixed.get("Scene.position", {"x": 0, "y": 0}),
        "rotation": prefixed.get("Scene.rotation", 0),
        "scale": prefixed.get("Scene.scale", 1),
        "layers": [],
    }

def to_prefixed(flat: dict) -> dict:
    """SceneParams → explore.py format (for CMA-ES integration)"""
    return {
        "Element.geometry": flat.get("geometry", "ring"),
        "Element.color": flat.get("color", "#FFFDDD"),
        "Scalars.repetitions": flat.get("repetitions", 65),
        "Scalars.alphaFactor": flat.get("alphaFactor", 0.65),
        "Scalars.scaleFactor": flat.get("scaleFactor", 1.05),
        "Scalars.rotationFactor": flat.get("rotationFactor", 0),
        "Scalars.stepFactor": flat.get("stepFactor", 0.02),
        "Scalars.scaleProgression": flat.get("scaleProgression", "exponential"),
        "Scalars.rotationProgression": flat.get("rotationProgression", "linear"),
        "Scalars.alphaProgression": flat.get("alphaProgression", "exponential"),
        "Scalars.positionProgression": flat.get("positionProgression", "index"),
        "Scalars.positionCoupled": flat.get("positionCoupled", True),
        "Spatial.origin": flat.get("origin", "center"),
        "Spatial.xStep": flat.get("xStep", 0),
        "Spatial.yStep": flat.get("yStep", 0),
        "Scene.debug": False,
        "Scene.position": flat.get("position", {"x": 0, "y": 0}),
        "Scene.rotation": flat.get("rotation", 0),
        "Scene.scale": flat.get("scale", 1),
    }
```

### Two Types of References

| Reference | Source | Exact params exist? | Success metric |
|-----------|--------|---------------------|----------------|
| `ref.png` | Your renderer | Yes | SSIM > 0.9 |
| `figma-*.png` | Figma designs | Unknown | Best achievable SSIM |

For renderer outputs (ref.png), the CNN should recover exact params.
For external designs (figma), the CNN finds the *closest approximation* your renderer can produce.

---

## Phase 1: Data Generation

**Objective:** Create synthetic (image, params) pairs with ground truth

### API Endpoint

Renderer exposes POST endpoint - no playwright needed in generation loop:
```python
response = requests.post(
    "http://localhost:3000/api/raster",
    json=params,  # SceneParams structure
)
image = Image.open(io.BytesIO(response.content))
```

### Generation Script

```python
# generate_data.py
import requests
import json
from pathlib import Path
from explore import random_params
from utils import to_scene_params

def generate(n=2000, n_layers=0):
    Path("data/images").mkdir(parents=True, exist_ok=True)
    Path("data/params").mkdir(parents=True, exist_ok=True)
    
    for i in range(n):
        prefixed = random_params(n_layers=n_layers)
        flat = to_scene_params(prefixed)  # Convert to API format
        
        # Call renderer API
        resp = requests.post(
            "http://localhost:3000/api/raster",
            json=flat,
            timeout=10
        )
        
        # Save image
        with open(f"data/images/{i:06d}.png", "wb") as f:
            f.write(resp.content)
        
        # Save params (flat keys - what CNN will learn)
        with open(f"data/params/{i:06d}.json", "w") as f:
            json.dump(flat, f)
        
        if i % 100 == 0:
            print(f"{i}/{n}")

if __name__ == "__main__":
    generate(n=2000, n_layers=0)  # Start simple, no layers
```

### Configuration

- **Start:** 2k samples, n_layers=0 (for M3 prototyping)
- **Scale:** 50k samples, n_layers=0-1 (on GPU)
- **Stratify:** Ensure all geometries, progressions appear equally

---

## Phase 2: CNN Inverse Model

**Objective:** Train ResNet18 to predict params from images

### Flattened Schema (Flat Keys)

The CNN uses flat keys (same as API). Object types are expanded:

```python
# Continuous params (flat keys, nested objects flattened)
FLAT_CONTINUOUS = [
    "repetitions",
    "alphaFactor",
    "scaleFactor",
    "rotationFactor",
    "stepFactor",
    "xStep",
    "yStep",
    "rotation",
    "position.x",   # flattened from position: {x, y}
    "position.y",
    "scale",
]
# Total: 11 continuous values

# Categorical params (flat keys)
CATEGORICAL = {
    "geometry": ["ring", "bar", "line", "arch", "u", "spiral", "wave", "infinity", "square", "roundedRect"],
    "scaleProgression": ["linear", "exponential", "additive", "fibonacci", "golden", "sine"],
    "rotationProgression": ["linear", "golden-angle", "fibonacci", "sine"],
    "alphaProgression": ["exponential", "linear", "inverse"],
    "positionProgression": ["index", "scale"],
    "origin": ["center", "top-center", "bottom-center", "top-left", "top-right", "bottom-left", "bottom-right"],
}
# Total: 6 classification heads

# Booleans (flat key)
BOOLEANS = ["positionCoupled"]
```

### Architecture

```python
class InverseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0)
        feat_dim = 512
        
        # Continuous head (11 values)
        self.continuous_head = nn.Linear(feat_dim, len(FLAT_CONTINUOUS))
        
        # Categorical heads
        self.categorical_heads = nn.ModuleDict({
            k: nn.Linear(feat_dim, len(opts)) 
            for k, opts in CATEGORICAL.items()
        })
        
        # Boolean head
        self.bool_head = nn.Linear(feat_dim, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        continuous = self.continuous_head(features)
        categorical = {k: head(features) for k, head in self.categorical_heads.items()}
        boolean = torch.sigmoid(self.bool_head(features))
        return continuous, categorical, boolean
```

### Reconstruction (CNN Output → SceneParams for API)

```python
def reconstruct_params(continuous, categorical, boolean):
    """Convert CNN output to flat SceneParams for API."""
    params = {}
    
    # Continuous (reconstruct nested position object)
    for i, name in enumerate(FLAT_CONTINUOUS):
        val = continuous[i]
        if name == "position.x":
            if "position" not in params:
                params["position"] = {}
            params["position"]["x"] = float(val)
        elif name == "position.y":
            if "position" not in params:
                params["position"] = {}
            params["position"]["y"] = float(val)
        elif name == "repetitions":
            params[name] = int(round(val))  # count, not float
        else:
            params[name] = float(val)
    
    # Categoricals (argmax)
    for name, logits in categorical.items():
        idx = logits.argmax().item()
        params[name] = CATEGORICAL[name][idx]
    
    # Boolean
    params["positionCoupled"] = boolean.item() > 0.5
    
    # Fixed values
    params["debug"] = False
    params["color"] = "#FFFDDD"  # TODO: add color head if needed
    params["layers"] = []  # n_layers=0 for now
    
    return params  # Ready for POST to /api/raster
```

### Training

```python
# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model
model = InverseModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Loss weights
continuous_weight = 1.0
categorical_weight = 1.0

for epoch in range(10):
    for imgs, cont_targets, cat_targets, bool_targets in loader:
        imgs = imgs.to(device)
        cont_targets = cont_targets.to(device)
        
        cont_pred, cat_pred, bool_pred = model(imgs)
        
        # MSE for continuous
        loss = continuous_weight * F.mse_loss(cont_pred, cont_targets)
        
        # CrossEntropy for categoricals
        for k, pred in cat_pred.items():
            target = cat_targets[k].to(device)
            loss += categorical_weight * F.cross_entropy(pred, target)
        
        # BCE for boolean
        loss += F.binary_cross_entropy(bool_pred.squeeze(), bool_targets.float().to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Configuration

- **Batch size:** 16 (M3), 64 (GPU)
- **Epochs:** 10-20
- **n_layers:** 0 (add layer support after base works)
- **Color:** Hardcoded for now (add head later if needed)

---

## Phase 3: Validation on ref.png

**Objective:** Prove inverse model works on renderer outputs

### Steps

1. Load trained model
2. Inference on ref.png
3. Reconstruct params
4. Call renderer API with predicted params
5. Compare with SSIM (and optionally LPIPS for perceptual similarity)

```python
from skimage.metrics import structural_similarity as ssim

# Inference
model.eval()
img = load_and_transform("references/ref.png")
cont, cat, boolean = model(img.unsqueeze(0).to(device))
params = reconstruct_params(cont[0], {k: v[0] for k, v in cat.items()}, boolean[0])

# Render
resp = requests.post("http://localhost:3000/api/raster", json=params)
reconstructed = Image.open(io.BytesIO(resp.content))

# Compare
original = Image.open("references/ref.png")
score = ssim(np.array(original), np.array(reconstructed), channel_axis=2)
print(f"SSIM: {score}")
```

### Success Criteria

| Metric | Target | Interpretation |
|--------|--------|----------------|
| SSIM > 0.85 | Pass | CNN works |
| SSIM 0.7-0.85 | Refine | Use as CMA-ES seed, optimize with pixel loss |
| SSIM < 0.7 | Debug | Check training, data coverage |

---

## Phase 4: Application to Figma Designs

**Objective:** Find closest params for external designs

### Approach

1. Run inference on each figma-*.png
2. Render predicted params
3. Compute SSIM/LPIPS against original
4. Run CMA-ES refinement with pixel loss if needed
5. Document ceiling (renderer limitation vs CNN failure)

### Use LPIPS for Perceptual Similarity

SSIM is pixel-level. For artistic matching, LPIPS (Learned Perceptual Image Patch Similarity) is better:

```python
import lpips
loss_fn = lpips.LPIPS(net='alex')

# Lower is better (it's a distance)
perceptual_dist = loss_fn(reconstructed_tensor, original_tensor)
```

### Success Criteria

| Design | SSIM | LPIPS | Interpretation |
|--------|------|-------|----------------|
| figma-00.png | >0.7 | <0.3 | Renderer can express this |
| figma-XX.png | 0.4-0.7 | 0.3-0.5 | Partial match, document gaps |
| figma-YY.png | <0.4 | >0.5 | Renderer limitation |

### Refinement (if needed)

```python
from cma import CMAEvolutionStrategy
from utils import to_prefixed, to_scene_params

def pixel_loss(flat_params, target_image):
    resp = requests.post("http://localhost:3000/api/raster", json=flat_params)
    rendered = np.array(Image.open(io.BytesIO(resp.content)))
    return -ssim(rendered, target_image, channel_axis=2)  # negative for minimization

# Start from CNN prediction (flat) → convert to prefixed for CMA-ES
prefixed = to_prefixed(predicted_params)  # CNN output is flat
initial = params_to_vec(prefixed)
es = CMAEvolutionStrategy(initial, 0.1, {"bounds": [0, 1]})

for _ in range(50):
    solutions = es.ask()
    # vec → prefixed → flat for API
    scores = [pixel_loss(to_scene_params(vec_to_params(s)), target) for s in solutions]
    es.tell(solutions, scores)
```

---

## Phase 5: VLM Exploration (Integration)

**Objective:** Use CNN-derived params as seeds for grounded VLM exploration

### Key Format Boundary

```
CNN output (flat) ──▶ to_prefixed() ──▶ explore.py (leva-prefixed)
explore.py output ──▶ to_scene_params() ──▶ API (flat)
```

### Modified Exploration Loop

```python
# explore.py modifications
from utils import to_prefixed, to_scene_params

def load_ref():
    """Use CNN to derive seeds from visual references."""
    # Try CNN-derived params first
    if INVERSE_MODEL and Path("references/ref.png").exists():
        img = load_and_transform("references/ref.png")
        cont, cat, boolean = INVERSE_MODEL(img.unsqueeze(0))
        flat = reconstruct_params(cont[0], cat, boolean[0])
        return to_prefixed(flat)  # Convert to explore.py format
    
    # Fallback to manual refs
    refs = load_refs()
    return random.choice(refs) if refs else {}

async def run_optimization(...):
    ...
    for j, sol in enumerate(solutions):
        params = vec_to_params(sol)  # leva-prefixed
        flat = to_scene_params(params)  # convert for API
        
        # Render via API
        resp = requests.post("http://localhost:3000/api/raster", json=flat)
        
        # ... score with VLM ...
        
        # NEW: When VLM finds high-scoring output, extract params
        if score >= 9:
            extracted_flat = inverse_model_inference(screenshot)
            extracted_prefixed = to_prefixed(extracted_flat)
            save_to_refs(extracted_prefixed)  # Grounded, in explore.py format
            print(f"[g{g}:{j}] {score}/10 *EXTRACTED + SAVED*")
```

### Benefits

| Before | After |
|--------|-------|
| Random/manual seeds | CNN-derived seeds from targets |
| 10hrs to find good region | Start inside good region |
| Save screenshot only | Save extracted params (grounded) |
| Blind exploration | Exploration + understanding |

---

## File Structure

```
bb-llm/
├── plan.md              # this file
├── art-explorer/
│   ├── explore.py       # existing CMA-ES + VLM (modify Phase 5)
│   ├── score.py         # existing VLM scoring
│   ├── utils.py         # helpers + to_scene_params() + to_prefixed()
│   ├── generate_data.py # NEW: Phase 1
│   ├── train_inverse.py # NEW: Phase 2
│   ├── validate.py      # NEW: Phase 3-4
│   ├── inverse_model.py # NEW: model definition + reconstruct_params()
│   ├── data/
│   │   ├── images/
│   │   └── params/      # flat SceneParams JSON
│   ├── models/
│   │   └── inverse_model.pt
│   ├── validation/
│   │   ├── ref_reconstructed.png
│   │   └── figma_*_reconstructed.png
│   └── references/
│       ├── ref.png
│       ├── figma-00.png
│       └── ...
```

---

## Dependencies

```txt
# requirements.txt additions
timm>=0.9.0           # pretrained models
torchvision>=0.15.0   # transforms
scikit-image>=0.21.0  # SSIM
lpips>=0.1.4          # perceptual loss
requests>=2.28.0      # API calls
```

---

## Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| 1 | Data generated | 2k+ pairs |
| 2 | Training converges | Loss decreasing, cat acc >60% |
| 3 | ref.png SSIM | >0.85 (exact params exist) |
| 4 | figma-00.png SSIM | >0.7 or document ceiling |
| 5 | VLM finds 9/10s | Faster than 10hr baseline |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CNN doesn't converge | Check data loading, smaller lr, more epochs |
| Categoricals hard to predict | Weight categorical loss higher, try focal loss |
| SSIM ceiling on Figma designs | Document gaps, consider renderer enhancements |
| M3 too slow | Reduce data size for prototyping, scale on cloud GPU |
| Layers complicate param space | **Start with n_layers=0**, add complexity later |
| Color matching needed | Add color head or treat as categorical |
| Key format confusion | Use converters at boundaries: `to_scene_params()`, `to_prefixed()` |