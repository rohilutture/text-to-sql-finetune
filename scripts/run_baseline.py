"""Run the prompt-only baseline over Spider dev and save your 'before' number.

Examples:
  # 20-example smoke test (do this first to confirm the pipeline runs):
  python scripts/run_baseline.py --spider-root data/spider_data --limit 20

  # full dev run, with sample rows in the schema (usually a few points higher):
  python scripts/run_baseline.py --spider-root data/spider_data --include-samples \
      --predictions-out runs/baseline_dev.jsonl --result-out runs/baseline_dev.json

The result JSON is what you'll pass to compare.py alongside your fine-tuned run.
(run_eval.py does the same thing and also handles the fine-tuned model.)
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
    ap.add_argument("--spider-root", required=True, help="Path to unzipped Spider")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--limit", type=int, default=None, help="Cap examples (smoke test)")
    ap.add_argument("--include-samples", action="store_true",
                    help="Add sample rows to each schema (helps accuracy)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--predictions-out", default=None)
    ap.add_argument("--result-out", default=None)
    args = ap.parse_args()

    print(f"Loading {args.split} split from {args.spider_root} ...")
    examples = load_split(
        args.spider_root, split=args.split,
        include_samples=args.include_samples, limit=args.limit,
    )
    print(f"{len(examples)} examples, {len(set(e.db_id for e in examples))} databases")

    print(f"Loading model {args.model} ...")
    model = BaselineModel(model_name=args.model, load_in_4bit=args.load_in_4bit)

    def generate(items):
        return model.generate_sql_batch(items, batch_size=args.batch_size)

    if args.predictions_out:
        os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)

    result = evaluate(
        examples, generate, timeout_s=args.timeout_s,
        predictions_out=args.predictions_out,
    )
    print_report("Prompt-only baseline", result)

    if args.result_out:
        os.makedirs(os.path.dirname(args.result_out) or ".", exist_ok=True)
        with open(args.result_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved result to {args.result_out}")


if __name__ == "__main__":
    main()
