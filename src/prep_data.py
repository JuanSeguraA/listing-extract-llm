"""Download the WDC PAVE dataset (siavashsaki/wdc-pave-ave mirror) as local JSONL files."""

import json
from collections import Counter
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPO = "siavashsaki/wdc-pave-ave"
SPLITS = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    for split_name, filename in SPLITS.items():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{filename}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        out_path = DATA_DIR / filename
        out_path.write_bytes(resp.content)

        rows = [json.loads(line) for line in resp.text.splitlines()]
        print(f"{split_name}: {len(rows)} rows -> {out_path}")
        print(f"  categories: {dict(Counter(r['category'] for r in rows))}")


if __name__ == "__main__":
    main()
