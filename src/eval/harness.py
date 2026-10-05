"""Run a model over a Spider split and report execution accuracy.

Produces: overall accuracy, a per-difficulty breakdown, and a saved JSONL of
every prediction (with the failure reason) so you can eyeball what the model
gets wrong. `comparison_table` renders the headline before/after table once you
have two runs.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from ..data.spider import Example
from .execution import execution_match
from .hardness import TIERS, classify_hardness


def evaluate(
    examples: list[Example],
    generate: Callable[[list[tuple[str, str]]], list[str]],
    timeout_s: float = 30.0,
    predictions_out: Optional[str] = None,
) -> dict:
    """Score a model on `examples`.

    generate: a function taking [(schema, question), ...] and returning SQL
    strings in the same order (e.g. BaselineModel.generate_sql_batch).
    predictions_out: optional path to write per-example results as JSONL.

    Returns a dict with overall accuracy and per-tier accuracy.
    """
    items = [(ex.schema_text, ex.question) for ex in examples]
    preds = generate(items)
    assert len(preds) == len(examples), "generate() must return one SQL per example"

    per_tier_total = {t: 0 for t in TIERS}
    per_tier_correct = {t: 0 for t in TIERS}
    records = []

    for ex, pred_sql in zip(examples, preds):
        tier = classify_hardness(ex.gold_sql)
        verdict = execution_match(ex.db_path, ex.gold_sql, pred_sql, timeout_s)
        per_tier_total[tier] += 1
        if verdict["correct"]:
            per_tier_correct[tier] += 1
        records.append(
            {
                "db_id": ex.db_id,
                "question": ex.question,
                "gold_sql": ex.gold_sql,
                "pred_sql": pred_sql,
                "tier": tier,
                "correct": verdict["correct"],
                "reason": verdict["reason"],
            }
        )

    if predictions_out:
        with open(predictions_out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(examples)
    correct = sum(r["correct"] for r in records)
    by_tier = {
        t: {
            "n": per_tier_total[t],
            "correct": per_tier_correct[t],
            "accuracy": (per_tier_correct[t] / per_tier_total[t]) if per_tier_total[t] else None,
        }
        for t in TIERS
    }
    return {
        "n": total,
        "correct": correct,
        "execution_accuracy": correct / total if total else 0.0,
        "by_tier": by_tier,
    }


def print_report(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"Execution accuracy: {result['execution_accuracy']:.1%} "
          f"({result['correct']}/{result['n']})")
    print(f"{'tier':<8}{'n':>6}{'correct':>10}{'acc':>8}")
    for t in TIERS:
        b = result["by_tier"][t]
        acc = f"{b['accuracy']:.1%}" if b["accuracy"] is not None else "  -"
        print(f"{t:<8}{b['n']:>6}{b['correct']:>10}{acc:>8}")


def comparison_table(baseline: dict, finetuned: dict) -> str:
    """Render the before/after table — the centerpiece of your writeup."""
    lines = [
        "| Metric | Prompt-only | Fine-tuned | Delta |",
        "|---|---|---|---|",
    ]

    def row(label, b, f):
        if b is None or f is None:
            return f"| {label} | - | - | - |"
        return f"| {label} | {b:.1%} | {f:.1%} | {f - b:+.1%} |"

    lines.append(row("Overall", baseline["execution_accuracy"], finetuned["execution_accuracy"]))
    for t in TIERS:
        lines.append(
            row(
                t.capitalize(),
                baseline["by_tier"][t]["accuracy"],
                finetuned["by_tier"][t]["accuracy"],
            )
        )
    return "\n".join(lines)
