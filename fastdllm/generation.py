from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .attention import make_generation_attention_mask
from .config import DEFAULT_BLOCK_SIZE, DEFAULT_MASK_TOKEN, DEFAULT_SUB_BLOCK_SIZE


@dataclass
class GenerationStats:
    total_time_s: float
    ttft_s: float | None
    new_tokens: int
    forward_passes: int
    tokens_per_second: float
    block_size: int
    sub_block_size: int
    threshold: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    stats: GenerationStats


def _sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token ids and selected-token probabilities for [B, N, V] logits."""
    if temperature <= 0.0:
        probs = F.softmax(logits.float(), dim=-1)
        token_ids = probs.argmax(dim=-1)
        selected_p = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return token_ids, selected_p

    logits = logits.float() / temperature
    probs = F.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_sorted = torch.multinomial(sorted_probs.view(-1, sorted_probs.shape[-1]), 1)
        sampled_sorted = sampled_sorted.view(*sorted_probs.shape[:-1])
        token_ids = sorted_indices.gather(-1, sampled_sorted.unsqueeze(-1)).squeeze(-1)
    else:
        token_ids = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(*probs.shape[:-1])
    selected_p = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_ids, selected_p


def _mask_id(tokenizer: PreTrainedTokenizerBase, mask_token: str) -> int:
    mask = getattr(tokenizer, "mask_token_id", None)
    if mask is not None:
        return int(mask)
    token_id = tokenizer.convert_tokens_to_ids(mask_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Mask token {mask_token!r} is not present in the tokenizer")
    return int(token_id)


def _prepare_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    if ids.shape[1] == 0:
        bos = tokenizer.bos_token_id or tokenizer.eos_token_id
        if bos is None:
            raise ValueError("Tokenizer has no BOS/EOS token for an empty prompt")
        ids = torch.tensor([[bos]], dtype=torch.long, device=device)
    return ids


@torch.inference_mode()
def generate_block_diffusion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    block_size: int = DEFAULT_BLOCK_SIZE,
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
    threshold: float = 0.9,
    temperature: float = 0.0,
    top_p: float = 0.95,
    mask_token: str = DEFAULT_MASK_TOKEN,
    eos_token_id: int | None = None,
    max_steps_per_block: int | None = None,
) -> GenerationResult:
    """Generate with block-wise masked refinement.

    This is a practical Fast-dLLM v2 style sampler for decoder-only HF models that
    accept 4D additive attention masks. It prioritizes correctness and portability;
    benchmark scripts report actual speed on the user's hardware.
    """
    if block_size <= 0 or sub_block_size <= 0:
        raise ValueError("block_size and sub_block_size must be positive")
    if block_size % sub_block_size != 0:
        raise ValueError("block_size must be divisible by sub_block_size")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    device = next(model.parameters()).device
    mask_token_id = _mask_id(tokenizer, mask_token)
    eos = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    max_steps = max_steps_per_block or (block_size * 4)

    prefix = _prepare_prompt_ids(tokenizer, prompt, device)
    generated: list[int] = []
    first_commit_time: float | None = None
    forward_passes = 0
    start = time.perf_counter()

    while len(generated) < max_new_tokens:
        block = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x = torch.cat([prefix, block], dim=1)
        prefix_len = prefix.shape[1]
        decoded = torch.zeros((block_size,), dtype=torch.bool, device=device)
        block_steps = 0

        while not decoded.all():
            progressed_in_round = False
            for sb_start in range(0, block_size, sub_block_size):
                sb_end = sb_start + sub_block_size
                while not decoded[sb_start:sb_end].all():
                    if block_steps >= max_steps:
                        remaining = ~decoded[sb_start:sb_end]
                        if not remaining.any():
                            break
                    attention_mask = make_generation_attention_mask(
                        prefix_len,
                        block_size,
                        device=device,
                        dtype=torch.float32,
                    )
                    out = model(input_ids=x, attention_mask=attention_mask, use_cache=False)
                    forward_passes += 1
                    block_steps += 1

                    positions = torch.arange(sb_start, sb_end, device=device)
                    positions = positions[~decoded[sb_start:sb_end]]
                    if positions.numel() == 0:
                        break

                    # Token j is predicted from hidden state at absolute position prefix_len+j-1.
                    logit_positions = prefix_len + positions - 1
                    valid = logit_positions >= 0
                    positions = positions[valid]
                    logit_positions = logit_positions[valid]
                    if positions.numel() == 0:
                        break

                    logits = out.logits[:, logit_positions, :]
                    token_ids, probs = _sample_logits(logits, temperature=temperature, top_p=top_p)
                    token_ids = token_ids[0]
                    probs = probs[0]

                    commit = probs >= threshold
                    if not commit.any():
                        commit[probs.argmax()] = True
                    if block_steps >= max_steps:
                        commit[:] = True

                    commit_positions = positions[commit]
                    commit_tokens = token_ids[commit]
                    x[0, prefix_len + commit_positions] = commit_tokens
                    decoded[commit_positions] = True
                    progressed_in_round = True
                    if first_commit_time is None and commit_tokens.numel() > 0:
                        first_commit_time = time.perf_counter() - start

                    if decoded[sb_start:sb_end].all():
                        break
            if not progressed_in_round:
                break

        block_ids = x[0, prefix_len : prefix_len + block_size].tolist()
        stop = False
        for token in block_ids:
            if len(generated) >= max_new_tokens:
                break
            if eos is not None and token == eos:
                stop = True
                break
            generated.append(int(token))
        prefix = torch.cat([prefix, torch.tensor([block_ids], dtype=torch.long, device=device)], dim=1)
        if stop:
            break

    total = time.perf_counter() - start
    text = tokenizer.decode(generated, skip_special_tokens=True)
    stats = GenerationStats(
        total_time_s=total,
        ttft_s=first_commit_time,
        new_tokens=len(generated),
        forward_passes=forward_passes,
        tokens_per_second=(len(generated) / total) if total > 0 else 0.0,
        block_size=block_size,
        sub_block_size=sub_block_size,
        threshold=threshold,
    )
    return GenerationResult(text=text, token_ids=generated, stats=stats)
