from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .config import DEFAULT_MASK_TOKEN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str) -> torch.dtype | str:
    dtype = dtype.lower()
    if dtype == "auto":
        return "auto"
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def ensure_tokenizer_tokens(
    tokenizer: PreTrainedTokenizerBase,
    mask_token: str = DEFAULT_MASK_TOKEN,
) -> tuple[PreTrainedTokenizerBase, int, bool]:
    added = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    if mask_id == tokenizer.unk_token_id or mask_id is None:
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        added = True
    return tokenizer, int(mask_id), added


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    dtype: str = "auto",
    device: str | None = None,
    mask_token: str = DEFAULT_MASK_TOKEN,
    attn_implementation: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    tokenizer, mask_id, token_added = ensure_tokenizer_tokens(tokenizer, mask_token=mask_token)

    kwargs: dict[str, Any] = {"dtype": resolve_dtype(dtype)}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if token_added or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.mask_token_id = mask_id
    model.config.dllm_mask_token = mask_token

    target_device = torch.device(device) if device else get_device()
    model.to(target_device)
    model.eval()
    return model, tokenizer


def save_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


class Stopwatch:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        return False
