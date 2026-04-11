# Art Explorer

A VLM learns the full parametric space of a generative art engine. One model, three modes: image-to-params, text-to-params, and creative discovery. Trained via SFT on rendered pairs, refined with GRPO using the renderer as the RL environment.

## The Problem

[bb-particles](https://github.com/outthislife/bb-particles) is a Three.js + Leva generative art engine with ~90 continuous, categorical, and structural parameters (34 base + 11 per-layer x 5 layers). Geometry types, progression curves, layer compositions, symmetry axes, CRT/noise effects. No human can intuit what combination produces a given visual.

We want a single model that masters this entire space:

- Reconstruct an image's parameters (image-to-params)
- Generate parameters from text descriptions (text-to-params)
- Discover novel aesthetically pleasing outputs (discover mode)

## Model

[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), quantized to 4-bit NF4 via [bitsandbytes](https://huggingface.co/docs/bitsandbytes), with a [LoRA](https://huggingface.co/docs/peft/developer_guides/lora) adapter (rank=16, alpha=32) on all linear layers. This is the [QLoRA](https://huggingface.co/papers/2305.14314) pattern. The frozen base provides vision and language understanding; the ~50MB adapter learns the renderer's parameter space.

The vision encoder processes rendered images. The language model outputs structured JSON matching the renderer's schema. Both image and text conditioning share the same adapter weights and target format.

## Training Pipeline

### 1. Data Generation

Stratified random sampling over the full parameter space. Each sample is rendered via the engine's API, producing (image, params) pairs. Stratification ensures coverage of rare geometry types, edge-case ranges, and underrepresented layer configurations rather than purely random draws.

Separately, a captioning model (Gemma 3 4B via Ollama) describes each rendered image in domain-specific language: geometry names, progression types, density levels. This creates (text, params) pairs from the same images. Without captions, the model has zero text-to-params training signal.

### 2. Supervised Fine-Tuning

Standard next-token cross-entropy loss on the assistant response:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log p_\theta(y_t \mid y_{<t})$$

Prompt and system tokens are masked (labels = -100) so the model only learns to predict the JSON output. Two input modes from one dataset:

- Image input: `[system] [image] "Predict scene parameters" -> {JSON}`
- Text input: `[system] "Predict parameters for: golden spirals..." -> {JSON}`

Same target JSON, different conditioning. The model learns both mappings simultaneously.

SFT teaches the format: valid JSON, correct key names, plausible value ranges. Pure imitation, no notion of output quality.

### 3. Reinforcement Learning (GRPO)

[Group Relative Policy Optimization](https://huggingface.co/papers/2402.03300) refines the model beyond imitation by using the renderer as a live environment.

The RL loop, each step:

1. Sample G completions (default 4) from the current policy at temperature 0.8
2. POST each JSON to the renderer API, get a PNG back
3. Score each rendered image with a composite reward:
   - `lpips_ref`: negative [LPIPS](https://richzhang.github.io/PerceptualSimilarity/) distance to reference artworks
   - `lpips_target`: negative LPIPS to a target image (inverse mode)
   - `aesthetic`: [Ollama](https://ollama.ai) VLM judge score (optional, for explore mode)
4. Compute group-relative advantage, no value model needed:

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

5. Per-sample policy gradient with gradient clipping

GRPO over PPO because there's no value network to train and no reward model to fit. The renderer is the ground truth. Each group of G renders provides its own baseline.

When all G samples score identically, the advantage formula degenerates (division by ~0). Following observations from [Understanding R1-Zero-Like Training](https://huggingface.co/papers/2503.20783), these groups are detected and the gradient update is skipped.

4090 constraints: per-sample backward (no batch accumulation), LPIPS on CPU (avoids OOM from stacking AlexNet alongside the 8B VLM), explicit CUDA sync after optimizer steps. 4-bit quantization keeps the base model at ~5GB VRAM.

### 4. Iterative Feedback Loop

RL winners above a reward threshold are automatically harvested into the SFT training pool:

```
SFT -> eval -> RL (renders against live engine) -> harvest winners -> caption -> SFT
```

Each cycle expands the training set with on-policy examples. SFT data starts random, but the model generates from its own distribution during RL. Harvesting RL winners bridges that distribution gap.

### CMA-ES Refinement (optional)

For pixel-perfect image reconstruction, [CatCMA](https://github.com/CyberAgentAILab/cmaes) (CMA-ES with categorical variables) refines VLM-predicted parameters by directly optimizing LPIPS distance to the target. The VLM provides a strong initialization; CMA-ES polishes the continuous parameters within 1500 function evaluations.

## Parameter Space

| Category     | Parameters                                                                | Type                           |
| ------------ | ------------------------------------------------------------------------- | ------------------------------ |
| Scalars      | repetitions, alphaFactor, scaleFactor, rotationFactor, stepFactor         | Continuous                     |
| Progressions | scale, rotation, alpha, position                                          | Categorical (6-7 options each) |
| Spatial      | xStep, yStep, origin                                                      | Continuous + categorical       |
| Element      | geometry, geoWidth, startAngle, gradientAngle, gradientRange, color       | Mixed                          |
| Layers (x5)  | position, rotation, scale, per-layer overrides, geometry                  | Structured                     |
| Effects      | Noise (density, opacity, size), CRT (bleed, bloom, mask, scanlines, warp) | Conditional                    |

The VLM outputs flat JSON (`SceneParams` format). Converters handle the prefixed format the renderer API expects (`Scalars.repetitions` <-> `repetitions`).

## Evaluation

- Parse rate: fraction of outputs that are valid JSON
- Key overlap: predicted keys vs target keys
- Range validity: values within schema bounds
- LPIPS: perceptual distance between rendered prediction and ground truth
- Parameter coverage: categorical option coverage + continuous range bucket coverage

## Quick Start

```bash
conda activate bb-llm

# With existing data + trained adapter:
just caption           # caption uncaptioned images
just train             # retrain SFT with full dataset
just rl                # GRPO with composite reward
just eval              # metrics + coverage report

# Full pipeline from scratch:
just generate N=5000   # render random samples
just caption           # describe them
just train             # SFT
just rl                # RL refinement
just round             # full cycle: rl -> caption -> train -> eval
```

## References

- [QLoRA: Efficient Finetuning of Quantized Language Models](https://huggingface.co/papers/2305.14314)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300) (GRPO)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://huggingface.co/papers/2503.20783) (zero-std analysis)
- [LPIPS: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://richzhang.github.io/PerceptualSimilarity/)
- [CatCMA: Stochastic Optimization for Mixed-Category Problems](https://github.com/CyberAgentAILab/cmaes)
