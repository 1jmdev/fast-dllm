from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .attention import expand_mask_for_batch, make_training_attention_mask


@dataclass
class BlockDiffusionCollator:
    mask_token_id: int
    block_size: int = 32
    context_length: int = 512
    mask_min_p: float = 0.15
    mask_max_p: float = 0.85
    attention_dtype: torch.dtype = torch.float32
    _base_attention_mask: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.context_length % self.block_size != 0:
            raise ValueError("context_length must be divisible by block_size")
        if not (0.0 < self.mask_min_p < self.mask_max_p < 1.0):
            raise ValueError("mask_min_p and mask_max_p must satisfy 0 < min < max < 1")

    def _sample_mask(self, batch: int, device: torch.device) -> torch.Tensor:
        shape = (batch, self.context_length)
        p = torch.empty((batch, 1), device=device).uniform_(self.mask_min_p, self.mask_max_p)
        mask = torch.rand(shape, device=device) < p

        # Guarantee useful supervision in every block.
        blocks = self.context_length // self.block_size
        mask_view = mask.view(batch, blocks, self.block_size)
        for b in range(batch):
            for blk in range(blocks):
                if not mask_view[b, blk].any():
                    mask_view[b, blk, torch.randint(0, self.block_size, (1,), device=device)] = True
                if mask_view[b, blk].all():
                    mask_view[b, blk, torch.randint(0, self.block_size, (1,), device=device)] = False
        return mask

    def _make_view(self, clean: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noised = clean.masked_fill(mask, self.mask_token_id)
        input_ids = torch.cat([noised, clean], dim=1)
        labels = torch.full_like(input_ids, -100)

        # AutoModelForCausalLM shifts labels internally, so place token i's label
        # at i + 1 to train the logit emitted at noised position i.
        shifted_labels = torch.full_like(clean, -100)
        shifted_labels[mask] = clean[mask]
        labels[:, 1 : self.context_length + 1] = shifted_labels
        return input_ids, labels

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        clean = torch.stack([ex["input_ids"] for ex in examples], dim=0).long()
        device = clean.device
        mask = self._sample_mask(clean.shape[0], device)
        comp_mask = ~mask

        input_a, labels_a = self._make_view(clean, mask)
        input_b, labels_b = self._make_view(clean, comp_mask)
        input_ids = torch.cat([input_a, input_b], dim=0)
        labels = torch.cat([labels_a, labels_b], dim=0)

        if self._base_attention_mask is None or self._base_attention_mask.device != device:
            self._base_attention_mask = make_training_attention_mask(
                self.context_length,
                self.block_size,
                device=device,
                dtype=self.attention_dtype,
            )
        attention_mask = expand_mask_for_batch(self._base_attention_mask, input_ids.shape[0])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
