"""Bucket queries into Spider-style difficulty tiers: easy / medium / hard / extra.

IMPORTANT: this is a transparent HEURISTIC based on counting SQL components in
the gold query string. It approximates Spider's official tiers well enough to
break your accuracy down by difficulty during iteration ("am I only winning on
easy queries?"), but it is NOT the exact official algorithm, which parses the SQL
into a structured form and counts components precisely.

For the numbers that go in your final report/README, run the official Spider
evaluator so your per-tier breakdown matches what interviewers expect. Use this
one for fast local feedback.
"""

from __future__ import annotations

import re

_AGG = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)
_KEYWORDS = {
    "where": re.compile(r"\bwhere\b", re.IGNORECASE),
    "group_by": re.compile(r"\bgroup\s+by\b", re.IGNORECASE),
    "order_by": re.compile(r"\border\s+by\b", re.IGNORECASE),
    "having": re.compile(r"\bhaving\b", re.IGNORECASE),
    "limit": re.compile(r"\blimit\b", re.IGNORECASE),
    "join": re.compile(r"\bjoin\b", re.IGNORECASE),
    "or": re.compile(r"\bor\b", re.IGNORECASE),
    "like": re.compile(r"\blike\b", re.IGNORECASE),
}
_SETOPS = re.compile(r"\b(intersect|union|except)\b", re.IGNORECASE)


def _count_components(sql: str) -> int:
    score = len(_AGG.findall(sql))
    for pat in _KEYWORDS.values():
        score += len(pat.findall(sql))
    return score


def _has_nested_query(sql: str) -> bool:
    # A subquery shows up as a SELECT that isn't the first one.
    return len(re.findall(r"\bselect\b", sql, flags=re.IGNORECASE)) > 1


def classify_hardness(gold_sql: str) -> str:
    """Return 'easy' | 'medium' | 'hard' | 'extra' for a gold SQL string."""
    components = _count_components(gold_sql)
    nested = _has_nested_query(gold_sql)
    setops = bool(_SETOPS.search(gold_sql))

    # Set operations and nesting are the classic markers of the hardest tier.
    if setops or (nested and components >= 3):
        return "extra"
    if nested or components >= 4:
        return "hard"
    if components >= 2:
        return "medium"
    return "easy"


TIERS = ["easy", "medium", "hard", "extra"]
