from __future__ import annotations

import argparse
import json

from .config import DEFAULT_BLOCK_SIZE, DEFAULT_MASK_TOKEN, DEFAULT_SUB_BLOCK_SIZE
from .generation import generate_block_diffusion
from .utils import load_model_and_tokenizer, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate with a converted block-diffusion SmolLM2 model.")
    p.add_argument("--model", required=True, help="Path to converted model, e.g. outputs/.../final")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--sub-block-size", type=int, default=DEFAULT_SUB_BLOCK_SIZE)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        device=args.device,
        mask_token=args.mask_token,
    )
    result = generate_block_diffusion(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        sub_block_size=args.sub_block_size,
        threshold=args.threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        mask_token=args.mask_token,
    )
    if args.json:
        print(json.dumps({"text": result.text, "stats": result.stats.__dict__}, indent=2))
    else:
        print(result.text)
        print(json.dumps(result.stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
