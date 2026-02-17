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
make iterate                # full lifecycle round: generate → train → explore

# Individual stages
make generate N=5000        # random + ref-biased data → data/
make generate-quality       # scored breeding → data/quality/
make generate-preview       # serial with matplotlib preview window
make train                  # forward model (20 epochs) + inverse model (10 epochs)
make train-forward          # forward model only (coverage data)
make train-inverse-quality  # inverse model on quality data only
make train-inverse-all      # inverse model on coverage + quality data
make predict IMG=path.png   # inverse: image → params (gradient + CMA-ES refine)
make explore                # discover novel outputs (LPIPS novelty)
make explore-clip           # discover novel outputs (CLIP novelty)

# Reference management
make list-refs              # show all refs
make save-ref ID=328        # add sample from data/ to refs.jsonl
make save-ref ID=out:0      # add sample from output/
make rm-ref LINE=3          # remove by line number
make score-refs             # VLM rank refs with Qwen2-VL
make score-refs-fresh       # re-score all (ignore cache)

make clean | clean-data | clean-quality | clean-output
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
Forward Model (params → image)     ←── trained on coverage data (random + bred)
        ↕ differentiable
Gradient descent                   ←── backprop through forward model (~300 steps)
        ↕
CMA-ES polish                     ←── black-box refinement (~500 fevals)
        ↕
Inverse Model (image → params)     ←── DINOv2-S backbone, trained on quality data
        ↓
Curation (save_ref)                ←── human picks the good ones → refs.jsonl
        ↓
Breeding (generate)                ←── refs.jsonl seeds next generation
```

### Two-Dataset Lifecycle

```
Round N:
  1. Generate random data → data/          (coverage for forward model)
  2. Train forward model on data/
  3. Score refs → refs-scored.jsonl         (VLM aesthetic ranking)
  4. Generate quality data → data/quality/  (heavily bred from scored refs)
  5. Train inverse model on data/quality/   (DINOv2 backbone, frozen → unfrozen)
  6. Explore → curate → refs grow
  7. Repeat from 1
```

Forward model needs uniform coverage (random data). Inverse model needs domain-relevant data (quality data bred from curated refs).

### Pipeline Stages

1. **generate.py** - Creates (image, params) training pairs via renderer API. Random, ref-biased, or VLM-score-weighted breeding. Supports `--out` for separate output directories.

2. **model.py** - Two models:
   - **ForwardModel**: params → 3×256×256 image. 8×8 spatial bottleneck + self-attention at 32×32 + ConvTranspose2d decoder. ~11M params.
   - **InverseModel**: image → params. DINOv2-S/14 backbone (384-dim features, pretrained) with multi-head MLP outputs for continuous, categorical, boolean, and per-layer params.

3. **train.py** - Two-phase training:
   - Phase 1: Forward model on L1 + LPIPS perceptual loss (coverage data)
   - Phase 2: Inverse model end-to-end (param-space loss + perceptual loss through frozen forward model). DINOv2 backbone frozen for 3 epochs then unfrozen with lower LR.

4. **predict.py** - Three-stage inverse + explore:
   - Inverse: CNN predict → config screening (CMA-ES) → gradient descent through forward model → CMA-ES polish → API render
   - Explore: breed from refs → forward model scores novelty (LPIPS or CLIP) → top N rendered via API

5. **score_refs.py** - Qwen2-VL aesthetic scoring (0-1) on rendered refs. Outputs `refs-scored.jsonl`.

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
- Qwen2-VL-2B-Instruct model (auto-downloaded) for aesthetic scoring.
- DINOv2-S/14 pretrained weights via `timm` for the inverse model backbone.
- CLIP-ViT-B/16 via `transformers` (optional, for `--clip` novelty scoring).
- `lpips` AlexNet for perceptual loss (forward + inverse training, gradient/CMA-ES scoring).

## training-tests/

Separate experimental directory for from-scratch GPT training (MiniGPT with BPE tokenizer). Not part of the main rendering pipeline.
