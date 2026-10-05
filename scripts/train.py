"""Fine-tune a small model for text-to-SQL with a hand-written PyTorch loop.

This is deliberately NOT transformers.Trainer — the training step is written out
so every moving part is visible and explainable:

  data      -> custom SFTDataset + DataLoader with a padding collate_fn
  model     -> base model (optionally 4-bit) + LoRA adapters via PEFT
  step      -> forward -> loss -> backward, with gradient accumulation
  precision -> bf16 autocast (no GradScaler needed for bf16)
  schedule  -> linear warmup then cosine decay, with gradient clipping
  saving    -> checkpoint the adapter + optimizer + scheduler + step; resume

Read the loop in main() top to bottom — the comments explain *why* each piece is
there, which is what gets probed in an interview.

Typical runs (see README for the full sequence):
  # 0. sanity: overfit one batch to ~0 loss (proves the loop learns at all)
  python scripts/train.py --spider-root data/spider_data --overfit-one-batch --include-samples

  # 1. real run on the 1.5B (RTX 4090)
  python scripts/train.py --spider-root data/spider_data \
      --model Qwen/Qwen2.5-Coder-1.5B-Instruct --include-samples \
      --epochs 3 --batch-size 8 --grad-accum 4 --out runs/qwen1.5b-lora

  # 2. bigger headline run, QLoRA 4-bit
  python scripts/train.py --spider-root data/spider_data \
      --model Qwen/Qwen2.5-Coder-14B-Instruct \
      --load-in-4bit --grad-checkpointing --include-samples \
      --epochs 3 --batch-size 4 --grad-accum 8 --out runs/qwen14b-qlora
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.spider import load_split
from src.data.sft_dataset import SFTDataset, make_collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer(args):
    """Load the base model, attach LoRA adapters, and return (model, tokenizer).

    Two paths:
      - 4-bit (QLoRA): base weights quantized to NF4, then prepared for k-bit
        training. This is what lets a 14B/32B fit on a single 24GB card.
      - bf16 (plain LoRA): base weights in bf16. Fine for the 1.5B/7B.
    In both cases only the small LoRA adapters are trainable; the base is frozen.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,   # extra ~0.4 bits/param saved
        )
        model_kwargs["device_map"] = {"": 0}  # single GPU

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Gradient checkpointing trades compute for memory by recomputing
    # activations in the backward pass instead of storing them. Essential for
    # the big QLoRA runs; harmless (just slower) for the small ones.
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # incompatible with checkpointing

    if args.load_in_4bit:
        # Casts layernorms to fp32, enables input grads, etc. — the standard
        # prep so LoRA trains stably on top of quantized weights.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_checkpointing
        )
    elif args.grad_checkpointing:
        # Non-4bit + checkpointing still needs the frozen base's inputs to carry
        # grad so the checkpointed blocks have something to differentiate.
        model.enable_input_require_grads()

    if args.resume_from:
        # Resume: load the saved adapter as trainable rather than making a new one.
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # The attention + MLP projections. Covers Qwen/Llama-family names.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)

    # 4-bit already lands on GPU via device_map; the plain bf16 path does not,
    # so move it explicitly or inputs (on cuda) won't match the weights (on cpu).
    if not args.load_in_4bit:
        model = model.to("cuda")

    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()  # e.g. "trainable: 0.7% of all params"
    return model, tokenizer


def save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)       # LoRA adapter only — a few MB, not GBs
    tokenizer.save_pretrained(out_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        },
        os.path.join(out_dir, "training_state.pt"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider-root", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--out", default="runs/finetune")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8, help="per-step micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="micro-batches per optimizer step (effective batch = "
                         "batch-size * grad-accum)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="drop examples longer than this many tokens; raise if "
                         "many drop with --include-samples")
    ap.add_argument("--include-samples", action="store_true",
                    help="add sample rows to schemas (match this at eval time!)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers; 0 is fine (dataset is in memory) "
                         "and avoids Windows spawn overhead")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200, help="optimizer steps")
    ap.add_argument("--limit", type=int, default=None, help="cap train examples")
    ap.add_argument("--resume-from", default=None, help="checkpoint dir to resume")
    ap.add_argument("--overfit-one-batch", action="store_true",
                    help="sanity mode: loop one batch until loss ~0, then exit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"

    # ---- data ----
    print("Loading training data ...")
    examples = load_split(
        args.spider_root, split="train",
        include_samples=args.include_samples, limit=args.limit,
    )
    model, tokenizer = build_model_and_tokenizer(args)

    dataset = SFTDataset(examples, tokenizer, max_length=args.max_length)
    print(f"{len(dataset)} usable examples "
          f"({dataset.dropped} dropped for exceeding max_length={args.max_length})")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
        pin_memory=True,
        drop_last=True,
    )

    # ---- optimizer + schedule ----
    # Only LoRA params require grad; the base is frozen. Optimizing just these is
    # why memory and compute stay small.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step, start_epoch = 0, 0
    if args.resume_from:
        state = torch.load(os.path.join(args.resume_from, "training_state.pt"))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = state["global_step"]
        start_epoch = state["epoch"]
        print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

    # ---- sanity mode: overfit a single batch ----
    # If the loop is correct, one batch driven repeatedly should reach ~0 loss in
    # a few dozen steps. If it plateaus high, something is wrong (masking, LR,
    # frozen params) — find it here, not after a multi-hour run.
    if args.overfit_one_batch:
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        model.train()
        for g in optimizer.param_groups:
            g['lr'] = args.lr
        for step in range(60):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch)          # passing labels -> model returns loss
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad()
            print(f"[overfit] step {step:2d}  loss {out.loss.item():.4f}")
        print("If loss fell toward ~0, the loop learns. Now run for real.")
        return

    # ---- training loop ----
    print(f"Training: {total_steps} optimizer steps "
          f"(effective batch {args.batch_size * args.grad_accum})")
    model.train()
    optimizer.zero_grad()
    running = 0.0

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch)
                # Divide by grad_accum so accumulated grads average (not sum) to
                # the same scale as a single large batch.
                loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()

            # Step only on accumulation boundaries.
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = running / (args.log_every * args.grad_accum)
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}",
                                     step=global_step)
                    running = 0.0

                if global_step % args.save_every == 0:
                    ckpt = os.path.join(args.out, f"step-{global_step}")
                    save_checkpoint(model, tokenizer, optimizer, scheduler,
                                    global_step, epoch, ckpt)

        # Save at the end of every epoch too.
        save_checkpoint(model, tokenizer, optimizer, scheduler,
                        global_step, epoch + 1, os.path.join(args.out, f"epoch-{epoch}"))

    # Final checkpoint — point eval at this directory.
    final = os.path.join(args.out, "final")
    save_checkpoint(model, tokenizer, optimizer, scheduler, global_step,
                    args.epochs, final)
    print(f"\nDone. Fine-tuned adapter saved to: {final}")
    print(f"Evaluate it with:\n  python scripts/run_eval.py --spider-root "
          f"{args.spider_root} --model {args.model} --adapter-path {final} "
          f"--include-samples --result-out runs/finetuned_dev.json")


if __name__ == "__main__":
    main()
