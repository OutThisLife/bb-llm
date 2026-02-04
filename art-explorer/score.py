"""
Auto-score generative art using Qwen2-VL.
Goals (figma) + known good output (ref) as context.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


MODEL = "Qwen/Qwen2-VL-2B-Instruct"
REFS_DIR = Path("references")
_cache = {}


def load_model():
    if "model" not in _cache:
        print(f"Loading {MODEL}...")
        _cache["model"] = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            ),
            device_map="auto",
        )
        _cache["processor"] = AutoProcessor.from_pretrained(MODEL)
    return _cache["model"], _cache["processor"]


def get_goals(max_goals=2) -> list[Path]:
    all_goals = sorted(REFS_DIR.glob("figma-*.png"))
    return random.sample(all_goals, min(len(all_goals), max_goals)) if all_goals else []


def get_good_example() -> Path | None:
    p = REFS_DIR / "ref.png"
    return p if p.exists() else None


PROMPT = """Score 1-10 how well NEW matches GOALS. Be BRUTAL - most are 1-3.

GOALS: elegant spirals, 100+ thin fading lines, perfect center symmetry, depth.

1-2: chunky/blocky, <50 lines, off-center, random scatter, ugly
3-4: some lines but wrong look, messy, not centered
5-6: decent repetition, somewhat centered, approaching aesthetic
7-8: RARE - beautiful pattern, very close to goals
9-10: EXTREMELY RARE - indistinguishable from GOALS

DEFAULT TO LOW SCORES. 8+ should be exceptional.

Reply with ONLY a single digit."""


def score_image(image_path: str, debug=False) -> tuple[int, str]:
    torch.cuda.empty_cache()  # prevent fragmentation

    # Sanity check: reject blank/empty images
    from PIL import Image
    import numpy as np

    img = np.array(Image.open(image_path).convert("L"))
    mean_brightness = img.mean()
    if mean_brightness > 250 or mean_brightness < 5:  # nearly all white or black
        return 1, "blank"

    goals = get_goals()
    good = get_good_example()

    if debug:
        print(f"  [debug] goals={[g.name for g in goals]}, good={good}")

    if not goals and not good:
        raise FileNotFoundError(f"No references in {REFS_DIR}/")

    model, processor = load_model()

    content = []

    if goals:
        content.append({"type": "text", "text": "GOALS (target aesthetic):"})
        for g in goals:
            content.append({"type": "image", "image": f"file://{g.absolute()}"})

    if good:
        content.append(
            {"type": "text", "text": "\nGOOD EXAMPLE (nice output from my tool):"}
        )
        content.append({"type": "image", "image": f"file://{good.absolute()}"})

    content.append({"type": "text", "text": "\nNEW (to evaluate):"})
    content.append({"type": "image", "image": f"file://{Path(image_path).absolute()}"})
    content.append({"type": "text", "text": f"\n{PROMPT}"})

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    response = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()

    if debug:
        print(f"  [debug] raw: {repr(response)}")

    try:
        score = int("".join(c for c in response if c.isdigit())[:2] or "5")
        score = max(1, min(10, score))
    except:
        score = 5

    del inputs, output_ids
    torch.cuda.empty_cache()

    return score, response


def batch_score(data_dir: str):
    data_path = Path(data_dir)
    ratings_file = data_path / "ratings.jsonl"

    if not ratings_file.exists():
        print(f"No ratings file: {ratings_file}")
        return

    ratings = [
        json.loads(ln) for ln in ratings_file.read_text().strip().split("\n") if ln
    ]
    print(f"Found {len(ratings)} ratings")

    updated = 0
    for i, r in enumerate(ratings):
        if "ai_score" in r:
            continue

        screenshot = data_path / r["screenshot"]
        if not screenshot.exists():
            continue

        print(f"[{i+1}/{len(ratings)}] {screenshot.name}...", end=" ")
        score, _ = score_image(str(screenshot))
        r["ai_score"] = score
        updated += 1
        print(f"{score}/10")

    with open(ratings_file, "w") as f:
        for r in ratings:
            f.write(json.dumps(r) + "\n")

    print(f"\nUpdated {updated}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?")
    p.add_argument("--batch")
    args = p.parse_args()

    print(f"Goals: {[g.name for g in get_goals()]}, Good: {get_good_example()}")

    if args.batch:
        batch_score(args.batch)
    elif args.image:
        score, resp = score_image(args.image)
        print(f"Score: {score}/10 ({resp})")
    else:
        print("Usage: python score.py <image> | --batch <dir>")


if __name__ == "__main__":
    main()
