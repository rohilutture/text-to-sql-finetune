"""Evaluate a model on Spider dev — prompt-only baseline OR fine-tuned.

Same script for both; the only difference is whether you pass --adapter-path.

  # baseline (your 'before'):
  python scripts/run_eval.py --spider-root data/spider_data \
      --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --include-samples --result-out runs/baseline_dev.json

  # fine-tuned (your 'after'):
  python scripts/run_eval.py --spider-root data/spider_data \
      --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --adapter-path runs/qwen1.5b-lora/final \
      --include-samples --result-out runs/finetuned_dev.json

IMPORTANT: keep --include-samples the SAME here as you used in training. The
model must see schemas serialized the way it was trained on, or you'll measure a
train/eval mismatch instead of the model's real quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.spider import load_split
from src.eval.harness import evaluate, print_report
from src.models.baseline import BaselineModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider-root", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter-path", default=None,
                    help="path to a fine-tuned LoRA checkpoint; omit for baseline")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-samples", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--predictions-out", default=None)
    ap.add_argument("--result-out", default=None)
    args = ap.parse_args()

    tag = "fine-tuned" if args.adapter_path else "prompt-only baseline"
    print(f"Loading {args.split} split from {args.spider_root} ...")
    examples = load_split(
        args.spider_root, split=args.split,
        include_samples=args.include_samples, limit=args.limit,
    )
    print(f"{len(examples)} examples, {len(set(e.db_id for e in examples))} databases")

    print(f"Loading model ({tag}) ...")
    model = BaselineModel(
        model_name=args.model,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
    )

    def generate(items):
        return model.generate_sql_batch(items, batch_size=args.batch_size)

    if args.predictions_out:
        os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)

    result = evaluate(examples, generate, timeout_s=args.timeout_s,
                      predictions_out=args.predictions_out)
    print_report(tag, result)

    if args.result_out:
        os.makedirs(os.path.dirname(args.result_out) or ".", exist_ok=True)
        with open(args.result_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved result to {args.result_out}")


if __name__ == "__main__":
    main()
