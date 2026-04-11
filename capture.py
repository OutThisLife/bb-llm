"""
High-res capture via Playwright.
Encodes params through the raster API, then screenshots at 4K.
"""

import argparse
import json
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

from cli import cli
from save_ref import resolve_param_file

CAPTURES_DIR = Path("captures")
BASE_URL = "http://localhost:3000"
API_URL = f"{BASE_URL}/api/raster"
SIZE = 3840


def get_encoded(params: dict) -> str:
    """POST params to raster API, extract encoded string from header."""
    r = requests.post(API_URL, json=params, timeout=30)
    r.raise_for_status()
    return r.headers["X-Encoded-Params"]


BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-setuid-sandbox",
    "--disable-gpu-sandbox",
    "--enable-features=WebGL",
    "--ignore-gpu-blocklist",
    "--disable-gpu-compositing",
    "--force-color-profile=srgb",
]


def capture(encoded: str, out_path: Path, size: int = SIZE):
    dpr = max(1, size / 1024)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=BROWSER_ARGS)
        page = browser.new_page(
            viewport={"width": 1024, "height": 1024},
            device_scale_factor=dpr,
        )
        page.goto(f"{BASE_URL}/render?dpr={dpr}", wait_until="load", timeout=30000)
        page.wait_for_function("window.__RENDER_READY__ === true", timeout=60000)
        page.evaluate("(enc) => window.__updateParams?.(enc)", encoded)
        page.wait_for_function("window.__RENDER_READY__ === true", timeout=60000)
        page.screenshot(path=str(out_path), type="png", timeout=120000)
        browser.close()


@cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("id", help="Sample ID (328, out:0) — same as save-ref")
    p.add_argument("--size", type=int, default=SIZE, help="Viewport size (square)")
    p.add_argument("--out", type=str, default=None, help="Output filename")
    args = p.parse_args()

    path = resolve_param_file(args.id)
    if not path:
        return
    params = json.loads(path.read_text())

    encoded = get_encoded(params)
    CAPTURES_DIR.mkdir(exist_ok=True)
    label = args.id.replace(":", "_")
    out = Path(args.out) if args.out else CAPTURES_DIR / f"{label}_{args.size}.png"
    capture(encoded, out, args.size)
    print(f"Saved {out} ({args.size}x{args.size})")


if __name__ == "__main__":
    main()
