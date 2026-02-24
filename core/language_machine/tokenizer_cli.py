"""
CLI for BPE tokenization.

Usage:
    python -m language_machine.tokenizer_cli train -i data.txt -v 1000 -o tokenizer.json
    python -m language_machine.tokenizer_cli encode -t tokenizer.json --text "Hello world"
    python -m language_machine.tokenizer_cli decode -t tokenizer.json 72 101 108 108 111
    python -m language_machine.tokenizer_cli info -t tokenizer.json
"""

import argparse
import json
import sys
from pathlib import Path

from language_machine.generate import decode, encode


def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], path: str) -> None:
    """Save vocab and merges to JSON file."""
    data = {
        "vocab": {str(k): list(v) for k, v in vocab.items()},
        "merges": [[list(a), list(b)] for a, b in merges],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_tokenizer(path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Load vocab and merges from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in data["merges"]]

    return vocab, merges


def cmd_train(args):
    # Import tokenizer from tokenizer
    sys.path.insert(0, str(Path(__file__).parent.parent / "tokenizer" / "src"))
    from tokenizer import BPETokenizer

    special_tokens = (
        [s.encode("utf-8") for s in args.special_tokens.split(",")] if args.special_tokens else [b"<|endoftext|>"]
    )

    print(f"Training tokenizer on {args.input} (vocab_size={args.vocab_size})")
    tokenizer = BPETokenizer()
    vocab, merges = tokenizer.train(args.input, args.vocab_size, special_tokens)

    save_tokenizer(vocab, merges, args.output)
    print(f"Saved to {args.output} ({len(vocab)} tokens, {len(merges)} merges)")


def cmd_encode(args):
    vocab, merges = load_tokenizer(args.tokenizer)
    token_ids = encode(args.text, vocab, merges)
    print(" ".join(map(str, token_ids)))


def cmd_decode(args):
    vocab, _ = load_tokenizer(args.tokenizer)
    token_ids = [int(t) for t in args.tokens]
    print(decode(token_ids, vocab))


def cmd_info(args):
    vocab, merges = load_tokenizer(args.tokenizer)
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges: {len(merges)}")

    # Show special tokens (256 onwards, before merges)
    num_special = len(vocab) - 256 - len(merges)
    for i in range(256, 256 + num_special):
        if i in vocab:
            print(f"  {i}: {vocab[i].decode('utf-8', errors='replace')}")


def main():
    parser = argparse.ArgumentParser(description="BPE Tokenizer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p = subparsers.add_parser("train")
    p.add_argument("-i", "--input", required=True, help="Input text file")
    p.add_argument("-v", "--vocab-size", type=int, required=True)
    p.add_argument("-o", "--output", required=True, help="Output tokenizer JSON")
    p.add_argument("-s", "--special-tokens", help="Comma-separated special tokens")
    p.set_defaults(func=cmd_train)

    # Encode
    p = subparsers.add_parser("encode")
    p.add_argument("-t", "--tokenizer", required=True)
    p.add_argument("--text", required=True)
    p.set_defaults(func=cmd_encode)

    # Decode
    p = subparsers.add_parser("decode")
    p.add_argument("-t", "--tokenizer", required=True)
    p.add_argument("tokens", nargs="+")
    p.set_defaults(func=cmd_decode)

    # Info
    p = subparsers.add_parser("info")
    p.add_argument("-t", "--tokenizer", required=True)
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
