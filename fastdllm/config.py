from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_MASK_TOKEN = "[MASK]"
DEFAULT_BLOCK_SIZE = 32
DEFAULT_SUB_BLOCK_SIZE = 8


@dataclass(frozen=True)
class DllmConfig:
    base_model: str = DEFAULT_BASE_MODEL
    mask_token: str = DEFAULT_MASK_TOKEN
    block_size: int = DEFAULT_BLOCK_SIZE
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE
    threshold: float = 0.9
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.95
