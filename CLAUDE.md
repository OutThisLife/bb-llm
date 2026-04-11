# CLAUDE.md

## Project Overview

One VLM (Qwen3-VL-8B QLoRA) learns the full parametric art space of bb-particles. Trained on (image, params) pairs from a renderer API, it predicts complete scene parameters from images, text descriptions, or creative prompts. Optional CMA-ES refinement for pixel-perfect reconstruction.

## Environment Setup

```bash
conda create -n bb-llm python=3.11 -y
conda activate bb-llm
pip install -r requirements.txt
```

## Commands (run from repo root)

```bash
# Pipeline: generate → caption → train → eval
just generate N=5000       # random (image, params) pairs → data/
just caption               # text descriptions (ollama) → data/captions/
just train                 # SFT on data/
just eval                  # JSON metrics + parameter coverage
just eval-render           # + render LPIPS

# Inference
just predict TARGET=x.png  # image → params → render
just predict-refine TARGET=x.png  # + CMA-ES polish
just text TEXT="spiral..."  # text → params → render
just discover              # creative generation

# Curation
just judge                 # VLM scoring + keep winners
just save-ref ID=328
just list-refs

# RL
just rl                    # GRPO style mode (auto-harvests winners)
just round                 # full cycle: rl -> caption -> train -> eval
```

## Linting & Formatting

```bash
ruff check .
ruff format .
pyright
```

## Architecture

```
generate.py -> data/ (image, params pairs)
caption.py  -> data/captions/ (text descriptions)
               |
train.py    -> models/ (QLoRA adapter)
               |
eval_sft.py -> JSON metrics + render LPIPS + parameter coverage
               |
infer.py    -> image-to-render | text-to-render
               |
judge.py    -> VLM generate + ollama scoring -> save-ref
```

### Iterative RL Loop

```
SFT (train.py) -> eval -> RL (rl.py, composite reward, auto-harvests)
      ^                          |
      +-- caption new samples <--+
```

`just round` runs one full cycle: rl -> caption -> train -> eval.

### Key Files

- **generate.py** - Stratified random (image, params) pairs via renderer API.
- **caption.py** - Text descriptions for text-to-render training (Gemma 3 4B via Ollama).
- **train.py** - QLoRA fine-tune. Supports text+image inputs. Filters degenerate renders.
- **eval_sft.py** - Render-based eval: parse rate, key overlap, LPIPS, parameter coverage.
- **infer.py** - `--target` (image->render), `--text` (text->render). Optional `--refine` for CMA-ES.
- **scoring.py** - Shared ollama judge scoring (used by judge.py and rl.py).
- **judge.py** - VLM generate -> render -> ollama scores vs refs -> keep winners.
- **rl.py** - GRPO RL with composite reward. Zero-std skip. Auto-harvests winners to data/.
- **harvest.py** - Manual retroactive harvest at different thresholds.
- **utils.py** - Schema, random generation, format converters.

### Composite RL Reward

`rl.py` uses multi-signal scoring per mode:

| Mode | lpips_ref | lpips_target | aesthetic |
|------|-----------|-------------|-----------|
| style | 1.0 (0.8 w/ judge) | 0.0 | 0.0 (0.2 w/ judge) |
| inverse | 0.0 | 1.0 | 0.0 |
| explore | 1.0 (0.3 w/ judge) | 0.0 | 0.0 (0.7 w/ judge) |

GRPO stability: zero-std groups are detected and skipped. Logged in `rewards.csv`.

### Two Parameter Formats

| Context | Format | Example |
|---------|--------|---------|
| Renderer API / Leva | Prefixed | `"Scalars.repetitions"` |
| VLM / flat JSON | SceneParams | `"repetitions"` |

Converters in `utils.py`: `to_scene_params()` and `to_prefixed()`.

### External Dependencies

- Renderer API at `localhost:3000` (bb-particles).
- Ollama at `localhost:11434` (gemma3:4b for captions, gemma3:27b for judge scoring).
- `lpips` AlexNet for CMA-ES refinement.

## GPU Safety (4090)

- Per-sample backward in GRPO (no batch accumulation that could OOM)
- LPIPS on CPU (avoids stacking AlexNet on same GPU as 7B VLM)
- Explicit `torch.cuda.synchronize()` after optimizer step
- 4-bit NF4 quantization for base model

## Failure Triage

| Symptom | Cause | Fix |
|---------|-------|-----|
| All render fails | Renderer down | Start bb-particles `pnpm dev` |
| All judge scores 0 | Ollama down | `ollama serve` + pull model |
| Zero-std collapse | Uniform rewards | Check refs exist, try different mode |
| Illegal memory access | GPU OOM | Reduce `--group`, check no other GPU users |
| Harvest finds nothing | Threshold too high | Lower `--threshold` (default -0.3) |
