"""Execute SQL against a SQLite database and compare result sets.

This is the heart of *execution accuracy*: a prediction is correct if running it
returns the same rows as the gold query, regardless of how the SQL is written.

The three edge cases that silently corrupt a text-to-SQL eval — and how we handle
them here:

1. Row ordering. Two queries that differ only in row order are equivalent UNLESS
   the gold query has an ORDER BY. We compare as an ordered list only when the
   gold query is order-sensitive; otherwise as a multiset.

2. Column ordering. `SELECT name, age` and `SELECT age, name` are semantically
   equal but produce different tuples. We try column permutations (capped) so a
   correct answer with reordered columns still counts.

3. Cell normalization. Floats that differ in the last decimal place, ints vs.
   floats, and None handling all cause spurious mismatches. We normalize cells
   before comparing.

For the *final headline number* in your report, cross-check against the official
Spider test-suite evaluator (github.com/taoyds/test-suite-sql-eval), which uses
multiple databases per schema to catch queries that coincidentally return the
right rows on one DB. This harness is the fast local one you iterate against.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from itertools import permutations
from typing import Any, Optional


class ExecutionResult:
    """Outcome of running one SQL query."""

    def __init__(self, rows: Optional[list[tuple]], error: Optional[str] = None,
                 timed_out: bool = False):
        self.rows = rows
        self.error = error
        self.timed_out = timed_out

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out


def execute_sql(db_path: str, sql: str, timeout_s: float = 30.0) -> ExecutionResult:
    """Run `sql` against the SQLite file at `db_path` with a wall-clock timeout.

    SQLite has no native statement timeout in the stdlib driver, so we run the
    query on a worker thread and call conn.interrupt() from a timer if it runs
    long. Returns an ExecutionResult; execution errors are captured, not raised,
    because an invalid prediction is simply a wrong answer.
    """
    # check_same_thread=False: we run the query on a worker thread (for the
    # timeout) while conn.interrupt() may fire from this thread. Both are safe.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # Spider DBs occasionally contain bytes that aren't valid UTF-8.
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")

    result: dict[str, Any] = {}

    def _run():
        try:
            cur = conn.cursor()
            cur.execute(sql)
            result["rows"] = cur.fetchall()
        except Exception as e:  # noqa: BLE001 - a bad prediction is a wrong answer
            result["error"] = f"{type(e).__name__}: {e}"

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout_s)

    if worker.is_alive():
        conn.interrupt()
        worker.join(1.0)
        conn.close()
        return ExecutionResult(rows=None, timed_out=True)

    conn.close()
    if "error" in result:
        return ExecutionResult(rows=None, error=result["error"])
    return ExecutionResult(rows=result.get("rows", []))


def _normalize_cell(value: Any, float_places: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, float_places)
    if isinstance(value, bool):  # sqlite returns ints, but be safe
        return int(value)
    return value


def _normalize_rows(rows: list[tuple], float_places: int = 4) -> list[tuple]:
    return [tuple(_normalize_cell(c, float_places) for c in row) for row in rows]


def result_sets_match(
    gold_rows: list[tuple],
    pred_rows: list[tuple],
    order_matters: bool,
    max_perm_cols: int = 6,
    float_places: int = 4,
) -> bool:
    """Compare two result sets under the rules described in the module docstring.

    order_matters: pass True when the gold query has an ORDER BY (see
    `order_sensitive`). Then rows are compared as an ordered list; otherwise as
    a multiset.

    Column permutation: if column counts match and are <= max_perm_cols, we try
    every column permutation of the prediction so that a correct answer with
    reordered SELECT columns still counts. Above the cap we only try the identity
    permutation to avoid factorial blow-up (rare in practice).
    """
    if gold_rows is None or pred_rows is None:
        return False
    if len(gold_rows) != len(pred_rows):
        return False
    if len(gold_rows) == 0:
        return True

    n_cols_gold = len(gold_rows[0])
    n_cols_pred = len(pred_rows[0])
    if n_cols_gold != n_cols_pred:
        return False

    gold = _normalize_rows(gold_rows, float_places)
    pred = _normalize_rows(pred_rows, float_places)

    if n_cols_gold <= max_perm_cols:
        col_perms = permutations(range(n_cols_gold))
    else:
        col_perms = [tuple(range(n_cols_gold))]

    gold_multiset = None if order_matters else Counter(gold)
    for perm in col_perms:
        permuted = [tuple(row[i] for i in perm) for row in pred]
        if order_matters:
            if gold == permuted:
                return True
        else:
            if gold_multiset == Counter(permuted):
                return True
    return False


def order_sensitive(sql: str) -> bool:
    """Heuristic: the gold query cares about row order iff it has an ORDER BY.

    Simple substring check on the lowercased SQL. This is the standard
    simplification used by Spider-style evaluators and is correct for the vast
    majority of cases.
    """
    return "order by" in " ".join(sql.lower().split())


def execution_match(
    db_path: str,
    gold_sql: str,
    pred_sql: str,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Full comparison for one (gold, pred) pair. Returns a dict with the verdict
    and enough detail to debug failures later."""
    gold_res = execute_sql(db_path, gold_sql, timeout_s)
    if not gold_res.ok:
        # A gold query that won't run is a dataset/DB problem, not a model miss.
        return {
            "correct": False,
            "reason": "gold_failed",
            "gold_error": gold_res.error,
            "gold_timeout": gold_res.timed_out,
        }

    pred_res = execute_sql(db_path, pred_sql, timeout_s)
    if not pred_res.ok:
        return {
            "correct": False,
            "reason": "pred_timeout" if pred_res.timed_out else "pred_error",
            "pred_error": pred_res.error,
        }

    match = result_sets_match(
        gold_res.rows, pred_res.rows, order_matters=order_sensitive(gold_sql)
    )
    return {"correct": match, "reason": "match" if match else "wrong_rows"}
