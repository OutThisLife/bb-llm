"""Save/remove refs from refs.jsonl."""

import argparse
import json
from pathlib import Path

from utils import to_prefixed

REFS_PATH = Path("references/refs.jsonl")
DATA_DIR = Path("data/params")
QUALITY_DIR = Path("data/quality/params")
DATA_SCORED_DIR = Path("data-scored/params")
OUTPUT_DIR = Path("output/params")


def save_ref(sample_id: str):
    """Append sample's params to refs.jsonl.

    Supports:
      - ID (e.g., 328) → from data/params/
      - q:ID / quality:ID (e.g., q:328) → from data/quality/params/
      - bias:ID (e.g., bias:328) → from data-scored/params/
      - out:ID (e.g., out:0) → from output/params/
      - out:NAME (e.g., out:ref_000) → from output/params/ by name
    """
    if sample_id.startswith("out:"):
        sample_id = sample_id[4:]
        data_dir = OUTPUT_DIR
        source = "output"
    elif sample_id.startswith("quality:"):
        sample_id = sample_id[8:]
        data_dir = QUALITY_DIR
        source = "quality"
    elif sample_id.startswith("q:"):
        sample_id = sample_id[2:]
        data_dir = QUALITY_DIR
        source = "quality"
    elif sample_id.startswith("bias:"):
        sample_id = sample_id[5:]
        data_dir = DATA_SCORED_DIR
        source = "data-scored"
    else:
        data_dir = DATA_DIR
        source = "data"

    # Strip extension if present
    sample_id = sample_id.rsplit(".", 1)[0]

    # Try exact name first (for output files like "ref_000" or "explore_003")
    param_file = data_dir / f"{sample_id}.json"
    if not param_file.exists():
        # Try zero-padded numeric
        sample_id = sample_id.lstrip("0") or "0"
        padded = f"{int(sample_id):06d}"
        param_file = data_dir / f"{padded}.json"
    else:
        padded = sample_id
    if not param_file.exists():
        print(f"Not found: {param_file}")
        return

    with open(param_file) as f:
        flat = json.load(f)

    flat.pop("url", None)
    prefixed = to_prefixed(flat)

    # Check for duplicates
    REFS_PATH.parent.mkdir(exist_ok=True)
    new_json = json.dumps(prefixed, sort_keys=True)
    if REFS_PATH.exists():
        for line in REFS_PATH.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                if json.dumps(json.loads(line), sort_keys=True) == new_json:
                    print(f"Duplicate: {padded} already in refs.jsonl")
                    return
            except json.JSONDecodeError:
                continue  # skip malformed lines

    # Ensure file ends with newline before appending
    if REFS_PATH.exists():
        content = REFS_PATH.read_text()
        if content and not content.endswith("\n"):
            with open(REFS_PATH, "a") as f:
                f.write("\n")

    with open(REFS_PATH, "a") as f:
        f.write(json.dumps(prefixed) + "\n")

    total = len(REFS_PATH.read_text().strip().split("\n"))
    print(f"Saved {padded} ({source}) → refs.jsonl ({total} total)")


def rm_ref(line_arg: str):
    """Remove ref by line number (1-indexed)."""
    # Strip extension if present
    line_arg = line_arg.rsplit(".", 1)[0]
    line_num = int(line_arg)

    if not REFS_PATH.exists():
        print("refs.jsonl not found")
        return

    lines = REFS_PATH.read_text().strip().split("\n")
    if line_num < 1 or line_num > len(lines):
        print(f"Invalid line {line_num} (1-{len(lines)})")
        return

    removed = lines.pop(line_num - 1)
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    add = sub.add_parser("add", help="Add sample to refs")
    add.add_argument("id", help="Sample ID (e.g., 328)")

    rm = sub.add_parser("rm", help="Remove ref by line number")
    rm.add_argument("line", help="Line number (1-indexed)")

    sub.add_parser("list", help="List refs")

    args = p.parse_args()

    if args.cmd == "add":
        save_ref(args.id)
    elif args.cmd == "rm":
        rm_ref(args.line)
    elif args.cmd == "list":
        list_refs()
    else:
        p.print_help()
