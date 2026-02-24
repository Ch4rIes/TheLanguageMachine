"""
Encode a raw text file to a uint16 binary file using a BPE tokenizer.

Usage:
    python encode_dataset.py --tokenizer tokenizers/tinystories_10k.json \
                              --input data/tinystories_train.txt \
                              --output data/tinystories_train.bin

The output is a flat array of uint16 token IDs (numpy memmap compatible).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from any directory by adding core to path
ASSIGNMENT_DIR = Path(__file__).parent.parent.parent.parent / "core"
if str(ASSIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_DIR))

from language_machine.generate import encode  # noqa: E402


def load_tokenizer(path: str):
    with open(path) as f:
        data = json.load(f)
    vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in data["merges"]]
    return vocab, merges


def encode_file(input_path: str, output_path: str, tokenizer_path: str, chunk_size: int = 200_000) -> None:
    vocab, merges = load_tokenizer(tokenizer_path)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_size = input_path.stat().st_size
    processed = 0
    total_tokens = 0

    print(f"Encoding {input_path} → {output_path}")
    print(f"  Tokenizer: {tokenizer_path} ({len(vocab)} tokens)")

    with open(input_path, encoding="utf-8", errors="replace") as f, \
         open(output_path, "wb") as out_f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            ids = encode(chunk, vocab, merges)
            np.array(ids, dtype=np.uint16).tofile(out_f)
            total_tokens += len(ids)
            processed += len(chunk.encode("utf-8", errors="replace"))
            pct = min(100 * processed / max(total_size, 1), 100.0)
            print(f"  {pct:.1f}%  ({total_tokens:,} tokens so far)", flush=True)

    print(f"Done. {total_tokens:,} tokens saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode text dataset to uint16 binary")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer JSON")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output", required=True, help="Output .bin file")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Characters per chunk")
    args = parser.parse_args()

    encode_file(args.input, args.output, args.tokenizer, args.chunk_size)
