# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Inverse rendering pipeline that trains a CNN to predict renderer parameters from images, then uses VLM-guided exploration to discover aesthetic outputs. The renderer is an external service (`http://localhost:3000/api/raster`) that takes scene parameters and produces images.

**CNN gives you the map. VLM explores the territory.**

## Environment Setup

```bash
conda create -n bb-llm python=3.11 -y
conda activate bb-llm
pip install -r requirements.txt
playwright install chromium
```

## Commands (run from `art-explorer/`)

```bash
make all                    # generate → train → validate (full pipeline)
make iterate ITERS=5 N=2000 # feedback loop: regenerate/retrain/revalidate

# Individual stages
make generate N=2000        # random + ref-biased data → data/
make generate-preview       # serial with matplotlib preview window
make generate-biased        # VLM-weighted breeding → data-scored/
make train                  # 10 epochs
make train-long             # 30 epochs
make validate               # predict → render → SSIM, good results → refs.jsonl
make validate-image IMG=path/to/img.png

# Reference management
make list-refs              # show all refs
make save-ref ID=328        # add sample to refs.jsonl
make rm-ref LINE=3          # remove by line number
make score-refs             # VLM rank refs with Qwen2-VL
make score-refs-fresh       # re-score all (ignore cache)

make clean-data | clean-models | clean-all
```

## Linting & Formatting

```bash
ruff check .                # lint
ruff format .               # format
pyright                     # type check
```

Config in `pyproject.toml`: Python 3.11, line-length 88, double quotes.

## Architecture

### Pipeline Stages

1. **generate_data.py** - Creates (image, params) training pairs by calling the renderer API with random/bred parameters. Supports ref-biased breeding (60% inherit + 40% random, ±20% perturbation) and VLM-score-weighted breeding.

2. **model.py + train.py** - ResNet18 backbone with multi-head outputs:
   - Continuous head (10 params) → MSE loss
   - Categorical heads (6 types: geometry, progressions, origin) → Cross-entropy
   - Boolean head (2 flags) → BCE
   - Layer count + per-layer heads for up to 5 layers
   - Saves checkpoint to `models/inverse_model.pt`

3. **validate.py** - Runs inference on reference images, renders predictions via API, computes SSIM. Results with SSIM > 0.85 are appended to `refs.jsonl` for future breeding.

4. **score_refs.py** - Qwen2-VL aesthetic scoring (0-1) on rendered refs. Outputs `refs-scored.jsonl` sorted by score, used for biased generation.

### Two Parameter Formats

| Context | Format | Example |
|---------|--------|---------|
| explore.py (CMA-ES) | Leva-prefixed | `"Scalars.repetitions"`, `"Scene.position"` |
| API / CNN | Flat (SceneParams) | `"repetitions"`, `"position"` |

Converters in `utils.py`: `to_scene_params()` (prefixed → flat) and `to_prefixed()` (flat → prefixed). CNN uses flat format internally; convert only at boundaries.

### Key Files

- **utils.py** - `SCHEMA` and `LAYER_SCHEMA` are the single source of truth for all parameter definitions (types, ranges, biases). Also contains `COLORS` palette and format converters.
- **references/refs.jsonl** - Curated parameter sets that seed future data generation.
- **plan.md** - Detailed architecture document and implementation roadmap.

### External Dependencies

- Renderer API must be running at `localhost:3000` for data generation and validation.
- Qwen2-VL-2B-Instruct model (auto-downloaded) for aesthetic scoring.
- `timm` ResNet18 pretrained weights for the inverse model backbone.

## training-tests/

Separate experimental directory for from-scratch GPT training (MiniGPT with BPE tokenizer). Not part of the main inverse rendering pipeline.
