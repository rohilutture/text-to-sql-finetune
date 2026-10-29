"""Load Spider examples and pair each with its database path and schema text.

Expected layout after you unzip Spider (see scripts/download_spider.md):

  <spider_root>/
    train_spider.json
    dev.json
    tables.json
    database/<db_id>/<db_id>.sqlite

Each example we yield is an Example dataclass:
  db_id, question, gold_sql, db_path, schema_text
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from .schema import load_tables, serialize_schema


@dataclass
class Example:
    db_id: str
    question: str
    gold_sql: str
    db_path: str
    schema_text: str


def _db_path(spider_root: str, db_id: str) -> str:
    return os.path.join(spider_root, "database", db_id, f"{db_id}.sqlite")


def load_split(
    spider_root: str,
    split: str = "dev",
    include_samples: bool = False,
    n_samples: int = 3,
    limit: Optional[int] = None,
) -> list[Example]:
    """Load a Spider split ('dev' or 'train') as a list of Example objects.

    include_samples appends sample rows to each schema (reads the DB files).
    limit caps the number of examples for quick smoke tests.
    """
    split_file = "dev.json" if split == "dev" else "train_spider.json"
    with open(os.path.join(spider_root, split_file), "r", encoding="utf-8") as f:
        raw = json.load(f)
    if limit is not None:
        raw = raw[:limit]

    tables = load_tables(os.path.join(spider_root, "tables.json"))

    # Serialize each db's schema once, not per-example (dev reuses ~20 DBs).
    schema_cache: dict[str, str] = {}
    examples: list[Example] = []
    for row in raw:
        db_id = row["db_id"]
        db_path = _db_path(spider_root, db_id)
        if db_id not in schema_cache:
            schema_cache[db_id] = serialize_schema(
                tables[db_id],
                db_path=db_path,
                include_samples=include_samples,
                n_samples=n_samples,
            )
        examples.append(
            Example(
                db_id=db_id,
                question=row["question"],
                gold_sql=row["query"],
                db_path=db_path,
                schema_text=schema_cache[db_id],
            )
        )
    return examples
