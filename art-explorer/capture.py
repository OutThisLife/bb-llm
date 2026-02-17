"""
High-res capture via Playwright.
Encodes params through the raster API, then screenshots at 4K.
"""

import argparse
import json
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

REFS_PATH = Path("references/refs.jsonl")
CAPTURES_DIR = Path("captures")
BASE_URL = "http://localhost:3000"
API_URL = f"{BASE_URL}/api/raster"
SIZE = 3840  # 4K


def load_ref(line: int) -> dict:
    lines = REFS_PATH.read_text().strip().split("\n")
    return json.loads(lines[line - 1])


def get_encoded(params: dict) -> str:
    """POST params to raster API, extract encoded string from header."""
    r = requests.post(API_URL, json=params, timeout=30)
    r.raise_for_status()
    return r.headers["X-Encoded-Params"]


def capture(encoded: str, out_path: Path, size: int = SIZE):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": size, "height": size})
        page.goto(f"{BASE_URL}/render?c={encoded}", wait_until="domcontentloaded")
        page.wait_for_function("window.__RENDER_READY__ === true", timeout=15000)
        page.screenshot(path=str(out_path), omit_background=True, type="png")
        browser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", type=int, default=180, help="Line number in refs.jsonl")
    ap.add_argument("--size", type=int, default=SIZE, help="Viewport size (square)")
    ap.add_argument("--out", type=str, default=None, help="Output filename")
    args = ap.parse_args()

    params = load_ref(args.line)
    encoded = get_encoded(params)

    CAPTURES_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else CAPTURES_DIR / f"ref_{args.line}_{args.size}.png"
    capture(encoded, out, args.size)
    print(f"Saved {out} ({args.size}x{args.size})")


if __name__ == "__main__":
    main()
