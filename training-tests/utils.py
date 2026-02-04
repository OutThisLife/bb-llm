"""Data loading utilities for bb-llm."""

import json
from pathlib import Path


def load_jsonl(path: str = "data.jsonl") -> list[dict]:
    """Load JSONL file as list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found. Run: make data")

    pairs = []
    for line in p.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            pairs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pairs


def jsonl_to_text(path: str = "data.jsonl") -> str:
    """Load JSONL and convert to plain text for language modeling."""
    pairs = load_jsonl(path)
    if not pairs:
        raise ValueError(f"{path} is empty or invalid")
    return "\n\n".join(f"Q: {p['q']}\nA: {p['a']}" for p in pairs)


def jsonl_to_sft(path: str = "data.jsonl", tokenizer=None) -> list[dict]:
    """Load JSONL and format for SFT training.

    If tokenizer provided, uses chat template for consistency with inference.
    Otherwise returns raw Q/A format.
    """
    pairs = load_jsonl(path)

    if tokenizer is None:
        return [{"text": f"Q: {p['q']}\nA: {p['a']}"} for p in pairs]

    # Format with chat template to match inference
    return [
        {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f"Q: {p['q']}"},
                    {"role": "assistant", "content": p["a"]},
                ],
                tokenize=False,
            )
        }
        for p in pairs
    ]
