"""Generate synthetic Q&A using local HuggingFace model. First run downloads ~4GB."""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_cache = {}

PROMPT = """Generate {n} diverse Q&A pairs about {topic}.

Return JSONL only, one object per line:
{{"q": "question here", "a": "answer here (1-3 sentences)"}}

Constraints:
- Vary question types: what/how/why/explain/compare
- Mix difficulty levels
- No markdown, no extra text
- Answer should be factual and concise

Generate {n} JSON objects:"""

PROMPT_SIMPLE = """Generate {n} simple Q&A pairs about {topic}.

Return JSONL only, one object per line:
{{"q": "What is X?", "a": "X is ... (1 sentence)"}}

Rules:
- Only "What is X?" style questions
- Answers must be exactly ONE short sentence
- Use simple words
- No jargon or complex explanations

Generate {n} JSON objects:"""


def load(name: str):
    if name not in _cache:
        print(f"Loading {name}...")
        _cache[name] = (
            AutoTokenizer.from_pretrained(name),
            AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                ),
                device_map="auto",
            ),
        )
    return _cache[name]


def generate(prompt: str, model_name: str) -> str:
    tok, model = load(model_name)
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )

    return tok.decode(out[0][ids.input_ids.shape[1] :], skip_special_tokens=True)


def parse_jsonl(text: str) -> list[dict]:
    """Parse JSONL output, with fallback to Q:/A: format."""
    pairs = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try JSON first
        try:
            obj = json.loads(line)
            if "q" in obj and "a" in obj:
                pairs.append({"q": obj["q"].strip(), "a": obj["a"].strip()})
                continue
        except json.JSONDecodeError:
            pass

        # Fallback: Q:/A: format (model didn't follow instructions)
        if line.startswith("Q:"):
            pairs.append({"q": line[2:].strip(), "a": ""})
        elif line.startswith("A:") and pairs and not pairs[-1]["a"]:
            pairs[-1]["a"] = line[2:].strip()

    return [p for p in pairs if p["q"] and p["a"]]


def norm_q(q: str) -> str:
    """Normalize question for deduplication."""
    q = q.lower().strip()
    q = re.sub(r"[^a-z0-9 ]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def ok_pair(q: str, a: str) -> bool:
    """Filter low-quality pairs."""
    q, a = q.strip(), a.strip()

    # Length checks
    if len(q) < 10 or len(q) > 200:
        return False
    if len(a) < 20 or len(a) > 500:
        return False

    # Content checks
    a_lower = a.lower()
    if "as an ai" in a_lower or "i don't have" in a_lower:
        return False
    if "[insert" in a_lower or "placeholder" in a_lower:
        return False

    # Answer shouldn't just repeat the question
    q_core = re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()
    if q_core and q_core in a_lower:
        return False

    return True


def load_jsonl(path: str) -> list[dict]:
    """Load existing JSONL file."""
    if not Path(path).exists():
        return []
    pairs = []
    for line in Path(path).read_text().strip().split("\n"):
        if line.strip():
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topic", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--calls", type=int, default=3)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output", default="data.jsonl")
    p.add_argument("--replace", action="store_true", help="replace instead of append")
    p.add_argument(
        "--simple", action="store_true", help="use simpler 'What is X?' format"
    )
    args = p.parse_args()

    prompt_template = PROMPT_SIMPLE if args.simple else PROMPT

    # Load existing data if appending
    existing = [] if args.replace else load_jsonl(args.output)
    seen = {norm_q(p["q"]) for p in existing}

    pairs = []
    for i in range(args.calls):
        print(f"\nBatch {i + 1}/{args.calls}...")
        try:
            raw = generate(
                prompt_template.format(topic=args.topic, n=args.n), args.model
            )
            new = parse_jsonl(raw)
            pairs.extend(new)
            print(f"  Parsed {len(new)} pairs")
        except Exception as e:
            print(f"  Error: {e}")

    # Filter + dedupe (against both new and existing)
    unique = []
    filtered = 0
    for p in pairs:
        if not ok_pair(p["q"], p["a"]):
            filtered += 1
            continue
        key = norm_q(p["q"])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    # Write JSONL (canonical format)
    all_pairs = existing + unique
    with open(args.output, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nFiltered {filtered} low-quality pairs")
    print(f"Total: {len(all_pairs)} pairs ({len(unique)} new) → {args.output}")


if __name__ == "__main__":
    main()
