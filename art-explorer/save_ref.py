"""Save/remove refs from refs.jsonl."""

import argparse
import json
from pathlib import Path

from utils import to_prefixed

REFS_PATH = Path("references/refs.jsonl")
DATA_DIR = Path("data/params")


def save_ref(sample_id: str):
    """Append sample's params to refs.jsonl."""
    # Strip extension if present (e.g., 000328.png → 000328)
    sample_id = sample_id.rsplit(".", 1)[0]
    sample_id = sample_id.lstrip("0") or "0"
    padded = f"{int(sample_id):06d}"
    
    param_file = DATA_DIR / f"{padded}.json"
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
    
    with open(REFS_PATH, "a") as f:
        f.write(json.dumps(prefixed) + "\n")
    
    total = len(REFS_PATH.read_text().strip().split("\n"))
    print(f"Saved {padded} → refs.jsonl ({total} total)")


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
