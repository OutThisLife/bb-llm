"""Save/remove refs from refs.jsonl."""

import argparse
import json
from pathlib import Path

from utils import to_prefixed

REFS_PATH = Path("references/refs.jsonl")
DATA_DIR = Path("data/params")
OUTPUT_DIR = Path("output/params")


def _resolve_param_file(sample_id: str):
    """Resolve sample ID to param file path. Returns Path or None."""
    if sample_id.startswith("out:"):
        sample_id, data_dir = sample_id[4:], OUTPUT_DIR
    else:
        data_dir = DATA_DIR

    sample_id = sample_id.rsplit(".", 1)[0]
    param_file = data_dir / f"{sample_id}.json"
    if not param_file.exists():
        sample_id = sample_id.lstrip("0") or "0"
        param_file = data_dir / f"{int(sample_id):06d}.json"
    if not param_file.exists():
        print(f"Not found: {param_file}")
        return None
    return param_file


def save_ref(sample_id: str):
    """Append sample's params to refs.jsonl."""
    path = _resolve_param_file(sample_id)
    if not path:
        return

    source = "output" if OUTPUT_DIR in path.parents else "data"

    with open(path) as f:
        flat = json.load(f)

    flat.pop("url", None)
    prefixed = to_prefixed(flat)

    REFS_PATH.parent.mkdir(exist_ok=True)
    new_json = json.dumps(prefixed, sort_keys=True)
    if REFS_PATH.exists():
        for line in REFS_PATH.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                if json.dumps(json.loads(line), sort_keys=True) == new_json:
                    print(f"Duplicate: {path.stem} already in refs.jsonl")
                    return
            except json.JSONDecodeError:
                continue

    if REFS_PATH.exists():
        content = REFS_PATH.read_text()
        if content and not content.endswith("\n"):
            with open(REFS_PATH, "a") as f:
                f.write("\n")

    with open(REFS_PATH, "a") as f:
        f.write(json.dumps(prefixed) + "\n")

    total = len(REFS_PATH.read_text().strip().split("\n"))
    print(f"Saved {path.stem} ({source}) → refs.jsonl ({total} total)")


def rm_ref(line_arg: str):
    """Remove ref by line number (1-indexed)."""
    line_num = int(line_arg.rsplit(".", 1)[0])
    if not REFS_PATH.exists():
        print("refs.jsonl not found")
        return
    lines = REFS_PATH.read_text().strip().split("\n")
    if line_num < 1 or line_num > len(lines):
        print(f"Invalid line {line_num} (1-{len(lines)})")
        return
    lines.pop(line_num - 1)
    REFS_PATH.write_text("\n".join(lines) + "\n" if lines else "")
    print(f"Removed line {line_num} ({len(lines)} remaining)")


def list_refs():
    """List refs with line numbers."""
    if not REFS_PATH.exists():
        print("refs.jsonl not found")
        return
    lines = REFS_PATH.read_text().strip().split("\n")
    for i, line in enumerate(lines, 1):
        data = json.loads(line)
        geo = data.get("Element.geometry", "?")
        rep = data.get("Scalars.repetitions", "?")
        print(f"{i:3d}. geo={geo}, rep={rep}")


def open_ref(sample_id: str):
    """Open sample in browser."""
    import webbrowser

    path = _resolve_param_file(sample_id)
    if not path:
        return
    with open(path) as f:
        flat = json.load(f)
    url = flat.get("url")
    if not url:
        print(f"No url in {path}")
        return
    webbrowser.open(url)
    print(f"Opened {path.stem}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("add", help="Add sample to refs").add_argument("id")
    sub.add_parser("rm", help="Remove by line number").add_argument("line")
    sub.add_parser("open", help="Open in browser").add_argument("id")
    sub.add_parser("list", help="List refs")

    args = p.parse_args()
    if args.cmd == "add":
        save_ref(args.id)
    elif args.cmd == "rm":
        rm_ref(args.line)
    elif args.cmd == "open":
        open_ref(args.id)
    elif args.cmd == "list":
        list_refs()
    else:
        p.print_help()
