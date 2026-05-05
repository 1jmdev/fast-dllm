from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastdllm.config import DEFAULT_BASE_MODEL, DEFAULT_BLOCK_SIZE, DEFAULT_MASK_TOKEN, DEFAULT_SUB_BLOCK_SIZE
from fastdllm.generation import generate_block_diffusion
from fastdllm.utils import load_model_and_tokenizer, resolve_dtype, save_json, set_seed


DEFAULT_PROMPTS = [
    "Write a concise explanation of why the sky is blue.",
    "Give three practical tips for learning Linux command line tools.",
    "Continue this story: The small robot found a map under the floorboards",
]


def _sample_next(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


@torch.inference_mode()
def generate_ar_timed(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    start = time.perf_counter()
    past_key_values = None
    new_tokens: list[int] = []
    eos = tokenizer.eos_token_id
    ttft = None

    cur = input_ids
    for step in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_id = _sample_next(out.logits[:, -1, :], temperature=temperature)
        if step == 0:
            ttft = time.perf_counter() - start
        token = int(next_id.item())
        if eos is not None and token == eos:
            break
        new_tokens.append(token)
        cur = next_id.view(1, 1)

    total = time.perf_counter() - start
    return {
        "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
        "stats": {
            "total_time_s": total,
            "ttft_s": ttft,
            "new_tokens": len(new_tokens),
            "tokens_per_second": (len(new_tokens) / total) if total > 0 else 0.0,
        },
    }


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare original AR SmolLM2 vs converted Fast-dLLM sampler.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dllm-model", required=True, help="Converted model path")
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--sub-block-size", type=int, default=DEFAULT_SUB_BLOCK_SIZE)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    p.add_argument("--out", default="outputs/bench_results.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    return prompts or DEFAULT_PROMPTS


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    prompts = load_prompts(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=resolve_dtype(args.dtype),
    ).to(device)
    base_model.eval()

    dllm_model, dllm_tokenizer = load_model_and_tokenizer(
        args.dllm_model,
        dtype=args.dtype,
        device=device,
        mask_token=args.mask_token,
    )

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ar = generate_ar_timed(
            base_model,
            base_tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dllm = generate_block_diffusion(
            dllm_model,
            dllm_tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            sub_block_size=args.sub_block_size,
            threshold=args.threshold,
            temperature=args.temperature,
            mask_token=args.mask_token,
        )
        rows.append(
            {
                "prompt": prompt,
                "ar_original": ar,
                "dllm_converted": {"text": dllm.text, "stats": asdict(dllm.stats)},
            }
        )

    ar_tps = [r["ar_original"]["stats"]["tokens_per_second"] for r in rows]
    ar_ttft = [r["ar_original"]["stats"]["ttft_s"] for r in rows]
    dllm_tps = [r["dllm_converted"]["stats"]["tokens_per_second"] for r in rows]
    dllm_ttft = [r["dllm_converted"]["stats"]["ttft_s"] for r in rows]
    summary = {
        "base_model": args.base_model,
        "dllm_model": args.dllm_model,
        "prompts": len(rows),
        "ar_original_mean_tps": _mean(ar_tps),
        "ar_original_mean_ttft_s": _mean(ar_ttft),
        "dllm_converted_mean_tps": _mean(dllm_tps),
        "dllm_converted_mean_ttft_s": _mean(dllm_ttft),
        "speedup_tps": (_mean(dllm_tps) / _mean(ar_tps)) if _mean(ar_tps) else None,
        "ttft_ratio_dllm_over_ar": (_mean(dllm_ttft) / _mean(ar_ttft)) if _mean(ar_ttft) else None,
    }
    payload = {"summary": summary, "rows": rows}
    save_json(args.out, payload)

    print(json.dumps(summary, indent=2))
    print(f"Wrote detailed results to {Path(args.out)}")


if __name__ == "__main__":
    main()
