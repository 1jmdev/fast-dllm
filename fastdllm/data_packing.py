from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


class JsonlTokenBlockDataset(IterableDataset):
    """Stream JSONL text and yield fixed-length token blocks."""

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        *,
        context_length: int = 512,
        text_key: str = "text",
        add_eos: bool = True,
        repeat: bool = True,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.text_key = text_key
        self.add_eos = add_eos
        self.repeat = repeat
        if not self.path.exists():
            raise FileNotFoundError(f"Missing dataset file: {self.path}")

    def _lines(self) -> Iterator[str]:
        while True:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
            if not self.repeat:
                return

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        eos = self.tokenizer.eos_token_id
        for line in self._lines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get(self.text_key, "")
            if not isinstance(text, str) or not text.strip():
                continue
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=min(int(getattr(self.tokenizer, "model_max_length", 8192)), 8192),
            ).input_ids
            if self.add_eos and eos is not None:
                ids.append(int(eos))
            buffer.extend(ids)
            while len(buffer) >= self.context_length:
                block = buffer[: self.context_length]
                del buffer[: self.context_length]
                yield {"input_ids": torch.tensor(block, dtype=torch.long)}
