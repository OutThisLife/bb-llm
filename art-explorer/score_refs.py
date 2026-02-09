"""Score refs.jsonl entries using Qwen2-VL for aesthetic ranking."""

import argparse
import json
import re
import tempfile
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

REFS_PATH = Path("references/refs.jsonl")
SCORED_PATH = Path("references/refs-scored.jsonl")
ENDPOINT = "http://localhost:3000/api/raster"

PROMPT = """Assuming the role of a human art curator or critic, given the knowledge that these are ML derived, rate this piece 0-1 in terms of how amazed you are that randomness could generate this.

Respond with ONLY a decimal number between 0 and 1, nothing else."""


def load_model():
    """Load Qwen2-VL model and processor."""
    print("Loading Qwen2-VL...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor


def render_params(params: dict) -> Image.Image:
    """Render params via API and return PIL Image."""
    resp = requests.post(ENDPOINT, json=params, timeout=60)
    resp.raise_for_status()
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(resp.content)
        return Image.open(f.name).convert("RGB")


def score_image(model, processor, image: Image.Image) -> float:
    """Score a single image using Qwen2-VL."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=16)
    
    # Decode only new tokens
    generated = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    
    # Parse score from response
    match = re.search(r"([01](?:\.\d+)?|\.\d+)", generated)
    if match:
        return min(1.0, max(0.0, float(match.group(1))))
    
    print(f"  Warning: couldn't parse score from '{generated}', defaulting to 0.5")
    return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", action="store_true", help="Re-score all refs, ignore existing scores")
    args = parser.parse_args()
    
    if not REFS_PATH.exists():
        print("refs.jsonl not found")
        return
    
    # Load existing scores
    existing = {}
    if SCORED_PATH.exists() and not args.rescore:
        for line in SCORED_PATH.read_text().strip().split("\n"):
            if line:
                entry = json.loads(line)
                key = json.dumps(entry["params"], sort_keys=True)
                existing[key] = entry["score"]
    
    model, processor = load_model()
    
    refs = [json.loads(ln) for ln in REFS_PATH.read_text().strip().split("\n") if ln]
    results = []
    
    print(f"Scoring {len(refs)} refs...")
    for i, params in enumerate(refs):
        key = json.dumps(params, sort_keys=True)
        
        if key in existing:
            score = existing[key]
            print(f"  [{i+1}/{len(refs)}] cached: {score:.2f}")
        else:
            try:
                image = render_params(params)
                score = score_image(model, processor, image)
                print(f"  [{i+1}/{len(refs)}] scored: {score:.2f}")
            except Exception as e:
                print(f"  [{i+1}/{len(refs)}] error: {e}, defaulting to 0.5")
                score = 0.5
        
        results.append({"params": params, "score": score})
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Save
    with open(SCORED_PATH, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\nSaved to {SCORED_PATH}")
    print("Top 5:")
    for entry in results[:5]:
        print(f"  {entry['score']:.2f}")
    print("Bottom 5:")
    for entry in results[-5:]:
        print(f"  {entry['score']:.2f}")


if __name__ == "__main__":
    main()
