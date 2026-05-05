from __future__ import annotations

import torch


def _blocked_value(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


def bool_to_additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert an allowed-attention boolean matrix to a 4D additive mask."""
    additive = torch.zeros_like(allowed, dtype=dtype)
    additive = additive.masked_fill(~allowed, _blocked_value(dtype))
    return additive.unsqueeze(0).unsqueeze(0)


def make_training_attention_mask(
    seq_len: int,
    block_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the 2L x 2L Fast-dLLM v2 training mask.

    Layout is [x_t, x_0]. For noised tokens x_t, attention is bidirectional only
    inside the same block and causal to clean previous blocks. Clean tokens x_0
    use block-causal attention over the clean sequence.
    """
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")

    total = 2 * seq_len
    allowed = torch.zeros((total, total), device=device, dtype=torch.bool)
    positions = torch.arange(seq_len, device=device)
    blocks = positions // block_size

    # x_t -> x_t: block diagonal, bidirectional inside block.
    same_block = blocks[:, None] == blocks[None, :]
    allowed[:seq_len, :seq_len] = same_block

    # x_t -> x_0: previous clean blocks only.
    previous_block = blocks[None, :] < blocks[:, None]
    allowed[:seq_len, seq_len:] = previous_block

    # x_0 -> x_0: block causal, same or previous clean blocks.
    clean_block_causal = blocks[None, :] <= blocks[:, None]
    allowed[seq_len:, seq_len:] = clean_block_causal

    return bool_to_additive_mask(allowed, dtype=dtype)


def make_generation_attention_mask(
    prefix_len: int,
    block_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the inference mask for [clean_prefix, noised_current_block].

    Prefix queries use ordinary causal attention. Current-block queries can attend
    to every prefix token and every token in the current block.
    """
    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive; prepend a BOS token for empty prompts")
    total = prefix_len + block_size
    allowed = torch.zeros((total, total), device=device, dtype=torch.bool)

    idx = torch.arange(total, device=device)
    prefix_q = idx < prefix_len
    current_q = ~prefix_q

    # Prefix remains left-to-right causal.
    causal = idx[None, :] <= idx[:, None]
    allowed[prefix_q] = causal[prefix_q]

    # Current block attends to all prefix tokens plus the full current block.
    allowed[current_q, :total] = True
    return bool_to_additive_mask(allowed, dtype=dtype)


def expand_mask_for_batch(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mask.shape[0] == batch_size:
        return mask
    return mask.expand(batch_size, -1, -1, -1).contiguous()
