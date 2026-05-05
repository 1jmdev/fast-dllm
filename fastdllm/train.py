from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .collator import BlockDiffusionCollator
from .config import DEFAULT_BASE_MODEL, DEFAULT_BLOCK_SIZE, DEFAULT_MASK_TOKEN
from .data_packing import JsonlTokenBlockDataset
from .utils import ensure_tokenizer_tokens, resolve_dtype, set_seed


def unwrap_for_save(accelerator: Accelerator, model):
    unwrapped = accelerator.unwrap_model(model)
    return getattr(unwrapped, "_orig_mod", unwrapped)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SmolLM2 with a Fast-dLLM v2 style objective.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--train-jsonl", default="data/fineweb_300mb.jsonl")
    p.add_argument("--output-dir", default="outputs/smollm2-135m-fastdllm")
    p.add_argument("--mask-token", default=DEFAULT_MASK_TOKEN)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    p.add_argument("--attn-implementation", default=None, choices=[None, "eager", "sdpa", "flash_attention_2"])
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_checkpoint(accelerator: Accelerator, model, tokenizer, output_dir: str, step: int) -> None:
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = unwrap_for_save(accelerator, model)
    unwrapped.save_pretrained(
        ckpt_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(ckpt_dir)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer, mask_id, added = ensure_tokenizer_tokens(tokenizer, args.mask_token)

    model_kwargs = {"dtype": resolve_dtype(args.dtype)}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if added or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.mask_token_id = mask_id
    model.config.dllm_mask_token = args.mask_token
    model.config.dllm_block_size = args.block_size
    model.config.dllm_context_length = args.context_length
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.torch_compile:
        if torch.cuda.is_available():
            model = torch.compile(model, options={"triton.cudagraphs": False})
        else:
            model = torch.compile(model, mode=args.compile_mode)

    dataset = JsonlTokenBlockDataset(
        args.train_jsonl,
        tokenizer,
        context_length=args.context_length,
        repeat=True,
    )
    collator = BlockDiffusionCollator(
        mask_token_id=mask_id,
        block_size=args.block_size,
        context_length=args.context_length,
        attention_dtype=resolve_dtype(args.dtype) if args.dtype != "auto" else torch.float32,
    )
    loader_kwargs = {
        "batch_size": args.per_device_batch_size,
        "collate_fn": collator,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.dataloader_num_workers > 0,
    }
    if args.dataloader_num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    optimizer_kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if torch.cuda.is_available():
        optimizer_kwargs["fused"] = True
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    model.train()
    model_dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())

    progress = tqdm(range(args.max_steps), disable=not accelerator.is_local_main_process)
    step = 0
    running = 0.0
    pending_loss = 0.0
    pending_count = 0
    while step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                if "attention_mask" in batch and batch["attention_mask"].is_floating_point():
                    batch["attention_mask"] = batch["attention_mask"].to(dtype=model_dtype)
                out = model(**batch, use_cache=False)
                loss = out.loss
                detached_loss = loss.detach().float().item()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pending_loss += detached_loss
            pending_count += 1

            if accelerator.sync_gradients:
                step += 1
                step_loss = pending_loss / max(1, pending_count)
                pending_loss = 0.0
                pending_count = 0

                running += step_loss
                avg_loss = running / max(1, step)
                progress.set_description(f"loss={avg_loss:.4f} step_loss={step_loss:.4f}")
                progress.update(1)
                if step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    save_checkpoint(accelerator, model, tokenizer, args.output_dir, step)
                if step >= args.max_steps:
                    break

    accelerator.wait_for_everyone()
    final_dir = Path(args.output_dir) / "final"
    unwrapped = unwrap_for_save(accelerator, model)
    unwrapped.save_pretrained(
        final_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_dir)
        print(f"Saved final model to {final_dir}")


if __name__ == "__main__":
    main()
