"""
VLM Fine-tuning: QLoRA on the full param space.
=============================================================
QLoRA (4-bit NF4) on (image, params) + optional (text, params) pairs.
One model → image-to-render, text-to-render, creative discovery.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

KEY_ORDER = [
    "repetitions", "geometry", "origin",
    "scaleProgression", "rotationProgression",
    "alphaProgression", "positionProgression", "positionCoupled",
    "alphaFactor", "scaleFactor", "rotationFactor", "stepFactor",
    "xStep", "yStep", "scale", "rotation",
    "geoWidth", "startAngle", "gradientAngle", "gradientRange",
    "noiseEnabled", "noiseDensity", "noiseOpacity", "noiseSize",
    "crtEnabled", "crtBleed", "crtBloom", "crtBrightness", "crtMask",
    "crtMaskStrength", "crtScale", "crtScanlines", "crtWarp",
]

OMIT_KEYS = {"debug", "url"}
NOISE_DETAIL = {"noiseDensity", "noiseOpacity", "noiseSize"}
CRT_DETAIL = {
    "crtBleed", "crtBloom", "crtBrightness", "crtMask",
    "crtMaskStrength", "crtScale", "crtScanlines", "crtWarp",
}
LAYER_OMIT = {"url"}


def _round(v, d=3):
    if isinstance(v, float):
        return round(v, d)
    if isinstance(v, list):
        return [_round(x, d) for x in v]
    if isinstance(v, dict):
        return {k: _round(val, d) for k, val in v.items()}
    return v


def clean_params(flat: dict) -> dict:
    noise = flat.get("noiseEnabled", False)
    crt = flat.get("crtEnabled", False)

    result = {}
    for key in KEY_ORDER:
        if key in OMIT_KEYS:
            continue
        if key in NOISE_DETAIL and not noise:
            continue
        if key in CRT_DETAIL and not crt:
            continue
        if key in flat:
            result[key] = _round(flat[key])

    # Include color and position (needed for full reconstruction)
    if "color" in flat:
        result["color"] = flat["color"]
    if "position" in flat:
        result["position"] = _round(flat["position"])

    layers = flat.get("layers", [])
    if layers:
        result["layers"] = [
            {k: _round(v) for k, v in layer.items() if k not in LAYER_OMIT}
            for layer in layers
        ]
    return result


SYSTEM_PROMPT = "You are a scene parameter predictor. Output valid JSON matching the renderer's param schema."
USER_PROMPT_IMAGE = "Predict the full scene parameters for this rendered image."
USER_PROMPT_TEXT = "Predict scene parameters for: "


def _iter_images(img_dir: Path):
    return sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in img_dir.glob(ext))


def _load_captions(data_dir: Path):
    path = data_dir / "captions.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def _collect_samples(data_dir: Path):
    img_dir, param_dir = data_dir / "images", data_dir / "params"
    if not img_dir.is_dir():
        return [], 0

    captions = _load_captions(data_dir)
    skipped = 0
    samples = []
    for img_path in _iter_images(img_dir):
        param_path = param_dir / f"{img_path.stem}.json"
        if not param_path.exists():
            continue

        px = np.array(Image.open(img_path)).mean()
        if px < 5 or px > 250:
            skipped += 1
            continue

        target = json.dumps(clean_params(json.loads(param_path.read_text())), separators=(",", ":"))
        samples.append({"image_path": str(img_path), "caption": "", "target": target})

        cap = captions.get(img_path.stem, {})
        for key in ("structured", "prompt", "brief"):
            text = cap.get(key, "").strip()
            if text:
                samples.append({"image_path": "", "caption": text, "target": target})

    return samples, skipped


def build_dataset(data_dir: Path, val_split: float = 0.1):
    dirs = [data_dir]
    refs = data_dir / "refs"
    if refs.is_dir():
        dirs.append(refs)

    samples, skipped = [], 0
    for d in dirs:
        s, sk = _collect_samples(d)
        samples.extend(s)
        skipped += sk
        if s:
            print(f"  {d}: {len(s)} samples")

    if skipped:
        print(f"  Filtered {skipped} degenerate renders")

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_split))
    return samples[n_val:], samples[:n_val]


def _build_messages(sample: dict) -> list[dict]:
    if sample["image_path"]:
        user_content = [
            {"type": "image", "image": sample["image_path"]},
            {"type": "text", "text": USER_PROMPT_IMAGE},
        ]
    else:
        user_content = [{"type": "text", "text": USER_PROMPT_TEXT + sample["caption"]}]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample["target"]},
    ]


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def _find_token_sequence(seq, pattern):
    for i in range(len(seq) - len(pattern), -1, -1):
        if seq[i : i + len(pattern)] == pattern:
            return i
    return -1


def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 16,
    lora_r: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 500,
):
    print(f"Building dataset from {data_dir}...")
    train_samples, val_samples = build_dataset(data_dir)
    n_img = sum(1 for s in train_samples if s["image_path"])
    n_txt = sum(1 for s in train_samples if s["caption"])
    print(f"Train: {len(train_samples)} ({n_img} image, {n_txt} text), Val: {len(val_samples)}")

    train_data = Dataset.from_list(train_samples)
    val_data = Dataset.from_list(val_samples)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {MODEL_ID} (4-bit)...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
    ))
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if "visual" in name and param.requires_grad:
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        torch_empty_cache_steps=4,
    )

    response_marker = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )

    def collate_fn(examples):
        all_messages = [_build_messages(ex) for ex in examples]
        texts = [processor.apply_chat_template(msgs, tokenize=False) for msgs in all_messages]

        flat_images = []
        for msgs in all_messages:
            for msg in msgs:
                if isinstance(msg["content"], list):
                    for part in msg["content"]:
                        if part.get("type") == "image":
                            flat_images.append(Image.open(part["image"]).convert("RGB"))

        batch = processor(
            text=texts, images=flat_images if flat_images else None,
            padding=True, truncation=True, return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        for i in range(len(labels)):
            pos = _find_token_sequence(labels[i].tolist(), response_marker)
            if pos >= 0:
                labels[i, : pos + len(response_marker)] = -100
            labels[i][batch["attention_mask"][i] == 0] = -100
        batch["labels"] = labels
        return batch

    Trainer(
        model=model, args=training_args,
        train_dataset=train_data, eval_dataset=val_data, data_collator=collate_fn,
    ).train()

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Saved adapter to {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Fine-tune VLM on full param space")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=500)
    args = p.parse_args()

    train(
        data_dir=args.data, output_dir=args.output, epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, save_steps=args.save_steps,
    )


if __name__ == "__main__":
    main()
