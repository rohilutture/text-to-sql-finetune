"""Print the before/after execution-accuracy table from two eval results.

  python scripts/compare.py \
      --baseline runs/baseline_dev.json \
      --finetuned runs/finetuned_dev.json

This markdown table is the centerpiece of your writeup and resume bullet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.harness import comparison_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline result JSON")
    ap.add_argument("--finetuned", required=True, help="fine-tuned result JSON")
    args = ap.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.finetuned) as f:
        finetuned = json.load(f)

    print(comparison_table(baseline, finetuned))


if __name__ == "__main__":
    main()
