from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from fastdllm.config import DEFAULT_DATASET, DEFAULT_DATASET_CONFIG


def download_fineweb_subset(
    output_path: str,
    *,
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    split: str = "train",
    target_bytes: int = 300_000_000,
    text_key: str = "text",
) -> dict[str, int | str]:
    """Stream FineWeb and write approximately target_bytes of UTF-8 text as JSONL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    total_bytes = 0
    rows = 0

    with out.open("w", encoding="utf-8") as f, tqdm(total=target_bytes, unit="B", unit_scale=True) as bar:
        for item in ds:
            text = item.get(text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            record = {"text": text.strip()}
            line = json.dumps(record, ensure_ascii=False) + "\n"
            encoded = len(line.encode("utf-8"))
            f.write(line)
            total_bytes += encoded
            rows += 1
            bar.update(encoded)
            if total_bytes >= target_bytes:
                break

    meta = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "output_path": str(out),
        "rows": rows,
        "bytes": total_bytes,
    }
    with out.with_suffix(out.suffix + ".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an approximate 300MB FineWeb JSONL subset.")
    parser.add_argument("--out", default="data/fineweb_300mb.jsonl")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-bytes", type=int, default=300_000_000)
    parser.add_argument("--text-key", default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = download_fineweb_subset(
        args.out,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        target_bytes=args.target_bytes,
        text_key=args.text_key,
    )
    print(json.dumps(meta, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
