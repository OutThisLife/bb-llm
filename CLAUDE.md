# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Forward + inverse rendering pipeline. A differentiable forward model (params → image) enables end-to-end training and fast gradient-based optimization (~1ms GPU vs ~50ms HTTP). The renderer is an external service (`http://localhost:3000/api/raster`) that produces ground truth images.

**Forward model is the differentiable proxy. Inverse model maps images back to params. Gradient descent + CMA-ES refines.**

## Environment Setup

```bash
conda create -n bb-llm python=3.11 -y
conda activate bb-llm
pip install -r requirements.txt
playwright install chromium
```

## Commands (run from `art-explorer/`)

```bash
make generate N=5000   # random/bred data → data/
make thumb-refs        # cache ref thumbnails → data/refs/images/
make train             # forward → inverse → taste → inverse finetune
make discover          # balanced taste + novelty explore
make loop N=5000       # generate + train + discover

# Ref curation
make save-ref ID=328   # add from data/
make save-ref ID=out:0 # add from output/
make open-ref ID=328   # open in browser
make list-refs         # show all refs
make rm-ref LINE=3     # remove by line number

make clean             # rm data/ output/ models/
```

## Linting & Formatting

```bash
ruff check .                # lint
ruff format .               # format
pyright                     # type check
```

Config in `pyproject.toml`: Python 3.11, line-length 88, double quotes.

## Architecture

```
Forward Model (params → image)     ←── trained on coverage data (refs + random + bred)
        ↕ differentiable
Inverse Model (image → params)     ←── Phase 1: full coverage, Phase 2: taste-filtered top 20%
        ↕
Taste Model (params → score)       ←── refs (positive) vs random (negative)
        ↓
Discover (taste + novelty)         ←── breed → forward model → rank → API render
        ↓
Curation (save_ref)                ←── human picks → refs.jsonl → next round
```

### Training Lifecycle

```
make generate:
  Generate random + ref-bred data → data/

make thumb-refs:
  Cache ref thumbnails → data/refs/images/ (content-hashed, skip existing)

make train (4 phases):
  1. Forward model on all data (L1 + LPIPS)
  2. Inverse model on all data (param-space + perceptual loss)
  3. Taste model: refs vs random data
  4. Fine-tune inverse on taste-filtered top 20% (lower LR)
```

All models train on `data/`. Ref influence comes through breeding (30% of generated samples inherit from refs). Taste model scores data to focus inverse fine-tuning on the interesting region.

### Pipeline Stages

1. **generate.py** - Creates (image, params) training pairs via renderer API. Random + ref-bred samples (30% breeding rate).

2. **model.py** - Three models:
   - **ForwardModel**: params → 3×256×256 image. Self-attention decoder. ~11M params.
   - **InverseModel**: image → params. DINOv2-S/14 backbone with multi-head MLP outputs.
   - **TasteModel**: param features → preference logit. Lightweight MLP.

3. **train.py** - Four-phase training:
   - Phase 1: Forward model (L1 + LPIPS perceptual loss)
   - Phase 2: Inverse model full coverage (param-space + perceptual loss through frozen forward)
   - Phase 3: Taste model (refs vs random, BCE loss)
   - Phase 4: Inverse fine-tune on taste-filtered top 20% (lower LR)

4. **predict.py** - Inverse prediction + explore:
   - Inverse: CNN predict → CMA-ES → gradient descent → CMA-ES polish → API render
   - Explore: breed from refs → taste + novelty scoring → top N rendered

### Two Parameter Formats

| Context | Format | Example |
|---------|--------|---------|
| refs.jsonl / breeding | Leva-prefixed | `"Scalars.repetitions"`, `"Scene.position"` |
| API / CNN / models | Flat (SceneParams) | `"repetitions"`, `"position"` |

Converters in `utils.py`: `to_scene_params()` (prefixed → flat) and `to_prefixed()` (flat → prefixed). Models use flat format internally; convert only at boundaries. Dither params are flat in model format (`ditherEnabled`, `ditherStrength`, etc.) but nested in API SceneParams (`dither: {enabled, ...}`). Generation sends prefixed format to API (bypasses nesting); predict uses `to_prefixed()` to convert flat→prefixed before API calls.

### Key Files

- **utils.py** - Single source of truth: `SCHEMA`, `LAYER_SCHEMA`, derived constants (`CONTINUOUS_KEYS`, `CATEGORICAL_KEYS`, etc.), param encoding/decoding (`encode_params`/`decode_params`), normalization helpers, format converters.
- **model.py** - ForwardModel (attention-based decoder) + InverseModel (DINOv2-S backbone).
- **references/refs.jsonl** - Curated parameter sets that seed future data generation.
- **output/** - Results from predict/explore. Curate with `make save-ref ID=out:N`.

### External Dependencies

- Renderer API must be running at `localhost:3000` for data generation and prediction.
- DINOv2-S/14 pretrained weights via `timm` for the inverse model backbone.
- CLIP-ViT-B/16 via `transformers` (optional, for `--clip` novelty scoring).
- `lpips` AlexNet for perceptual loss (forward + inverse training, gradient/CMA-ES scoring).

## training-tests/

Separate experimental directory for from-scratch GPT training (MiniGPT with BPE tokenizer). Not part of the main rendering pipeline.
