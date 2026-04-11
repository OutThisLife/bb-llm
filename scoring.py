"""Shared scoring: ollama-based aesthetic judge + grid visualization."""

import base64
import io
import re

import requests
from PIL import Image, ImageDraw

OLLAMA = "http://localhost:11434"
JUDGE_MODEL = "gemma3:27b"

SCORE_PROMPT = (
    "This image shows {} reference artworks on top and 1 candidate on the bottom. "
    "Score the candidate on each criterion (0-10):\n"
    "- COMPOSITION: balance, focal point, use of space\n"
    "- BEAUTY: color harmony, elegance, visual appeal\n"
    "- DEPTH: layering, dimensionality, complexity\n"
    "- SYMMETRY: structural harmony, pattern coherence\n"
    "- MEMORABILITY: uniqueness, striking quality\n"
    "Reply in this exact format:\n"
    "C:_ B:_ D:_ S:_ M:_ TOTAL:_"
)


def img_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_grid(refs, candidate, cell=256):
    w, h = max(len(refs), 1) * cell, cell * 2 + 30
    grid = Image.new("RGB", (w, h), (30, 30, 30))
    for i, ref in enumerate(refs):
        grid.paste(ref.resize((cell, cell)), (i * cell, 0))
    ImageDraw.Draw(grid).text((4, cell + 2), "REFERENCES ^    CANDIDATE v", fill=(180, 180, 180))
    grid.paste(candidate.resize((cell, cell)), ((w - cell) // 2, cell + 30))
    return grid


def score_bar(score, width=20):
    filled = min(int(round(score / 5)), width)
    return f"{'█' * filled}{'░' * (width - filled)}"


def ollama_score(candidate, refs, model=JUDGE_MODEL, endpoint=OLLAMA):
    """Score candidate vs refs. Returns (score_0_100, raw_text, tok_in, tok_out)."""
    try:
        resp = requests.post(
            f"{endpoint}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": SCORE_PROMPT.format(len(refs)),
                              "images": [img_b64(make_grid(refs, candidate))]}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 80},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["message"]["content"].strip()
        tok_in = data.get("prompt_eval_count", 0)
        tok_out = data.get("eval_count", 0)

        scores = re.findall(r"[CBDSM]:\s*(\d+(?:\.\d+)?)", text)
        if len(scores) >= 5:
            return round(sum(float(s) for s in scores[:5]) / 5 * 10, 1), text, tok_in, tok_out

        total_m = re.search(r"TOTAL:\s*(\d+(?:\.\d+)?)", text)
        if total_m:
            return round(float(total_m.group(1)), 1), text, tok_in, tok_out

        for token in text.split():
            try:
                return round(float(token.replace("/100", "").replace("%", "")), 1), text, tok_in, tok_out
            except ValueError:
                continue
        return 0.0, text, tok_in, tok_out
    except Exception as e:
        return 0.0, str(e), 0, 0
