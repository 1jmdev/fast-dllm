# fastdllm-smollm2

Fast-dLLM v2 style block-diffusion adaptation for `HuggingFaceTB/SmolLM2-135M`.

This repo is a small, runnable conversion pipeline for experimenting with block-diffusion language modeling on SmolLM2-135M. It downloads a local FineWeb subset, adds a learned `[MASK]` token, trains with a Fast-dLLM-style masked block objective, and generates with block-wise masked refinement.

It does not ship model weights or dataset shards. The base model and FineWeb data are downloaded at runtime.

References:

- Fast-dLLM v2 paper: <https://arxiv.org/pdf/2509.26328>
- NVIDIA Fast-dLLM repo: <https://github.com/NVlabs/Fast-dLLM>

## Install

```bash
pip install -e .
```

For development tools:

```bash
pip install -e '.[dev]'
```

Dependencies live in `pyproject.toml` only.

## Download FineWeb

Download a 300MB streaming subset from `HuggingFaceFW/fineweb`:

```bash
python -m fastdllm.download_fineweb \
  --out data/fineweb_300mb.jsonl \
  --dataset-name HuggingFaceFW/fineweb \
  --dataset-config sample-10BT \
  --target-bytes 300000000
```

This writes:

```text
data/fineweb_300mb.jsonl
data/fineweb_300mb.jsonl.meta.json
```

## Train

Training saves checkpoints under `outputs/smollm2-135m-fastdllm/` and the final converted model under:

```text
outputs/smollm2-135m-fastdllm/final
```

### H100 Config

Use this for a larger single-H100 run:

```bash
accelerate launch -m fastdllm.train \
  --base-model HuggingFaceTB/SmolLM2-135M \
  --train-jsonl data/fineweb_300mb.jsonl \
  --output-dir outputs/smollm2-135m-fastdllm \
  --context-length 2048 \
  --block-size 64 \
  --per-device-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --max-steps 5000 \
  --learning-rate 2e-5 \
  --warmup-steps 500 \
  --dtype bf16 \
  --attn-implementation flash_attention_2
```

If FlashAttention 2 is not installed, remove `--attn-implementation flash_attention_2` or use `--attn-implementation sdpa`.

### 8GB GPU Config

Use this for an optimized single-GPU run:

```bash
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  -m fastdllm.train \
  --base-model HuggingFaceTB/SmolLM2-135M \
  --train-jsonl data/fineweb_300mb.jsonl \
  --output-dir outputs/smollm2-135m-fastdllm \
  --context-length 512 \
  --block-size 32 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-steps 1000 \
  --learning-rate 2e-5 \
  --warmup-steps 100 \
  --dtype bf16 \
  --attn-implementation sdpa \
  --torch-compile \
  --dataloader-num-workers 2
```

If you run out of memory, use `--per-device-batch-size 1 --gradient-accumulation-steps 16`; only add `--gradient-checkpointing` if needed. If your GPU does not support BF16, use `--dtype fp16`.

The 300MB dataset and short runs are for reproducible experimentation, not paper-level reproduction. Increase `--target-bytes`, `--max-steps`, and context length for more serious training.

## Generate

Generate from the converted model:

```bash
python -m fastdllm.cli_generate \
  --model outputs/smollm2-135m-fastdllm/final \
  --prompt "Explain block diffusion language models in one paragraph." \
  --max-new-tokens 128 \
  --block-size 32 \
  --sub-block-size 8 \
  --threshold 0.9
```

Python API:

```python
from fastdllm.generation import generate_block_diffusion
from fastdllm.utils import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer("outputs/smollm2-135m-fastdllm/final")
result = generate_block_diffusion(
    model,
    tokenizer,
    "Explain block diffusion language models.",
    max_new_tokens=128,
    block_size=32,
    sub_block_size=8,
    threshold=0.9,
)

print(result.text)
print(result.stats)
```

## Project Layout

```text
fastdllm/
  attention.py          # 4D additive attention masks
  cli_generate.py       # CLI generation
  collator.py           # complementary masking and shifted labels
  config.py             # defaults
  data_packing.py       # JSONL token packing dataset
  download_fineweb.py   # FineWeb subset downloader
  generation.py         # block-wise refinement sampler
  train.py              # Accelerate training loop
  utils.py              # model/tokenizer helpers
pyproject.toml          # package metadata and dependencies
```
