"""Bake the BERTimbau tokenizer and config (never the weights) into a local directory.

Run at image build time so the runtime container needs no network access.
"""

import argparse

from transformers import AutoConfig, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Download tokenizer and config only")
    parser.add_argument("--model-name", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    AutoTokenizer.from_pretrained(args.model_name).save_pretrained(args.out)
    AutoConfig.from_pretrained(args.model_name).save_pretrained(args.out)
    print(f"[fetch_tokenizer] Saved tokenizer and config of {args.model_name} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
