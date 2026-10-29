"""Turn a Spider `tables.json` entry into the text schema you feed the model.

Schema serialization is the quiet design choice that dominates text-to-SQL
accuracy. The rule that keeps your before/after comparison honest: the baseline
and the fine-tuned model must see the *identical* serialization. So this lives in
one place, and both paths import it.

`tables.json` format (per database):
  db_id                   : str
  table_names_original    : ["stadium", "singer", ...]
  column_names_original   : [[-1, "*"], [0, "Stadium_ID"], [0, "Location"], ...]
                            each entry is [table_index, column_name]; the first
                            is always [-1, "*"]
  column_types            : ["text", "number", ...]  (aligned to columns)
  primary_keys            : [col_index, ...]
  foreign_keys            : [[col_index, col_index], ...]
"""

from __future__ import annotations

import json
import sqlite3
from typing import Optional


def load_tables(tables_json_path: str) -> dict[str, dict]:
    """Load tables.json and index it by db_id."""
    with open(tables_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {e["db_id"]: e for e in entries}


def _sample_rows(db_path: str, table: str, n: int) -> list[tuple]:
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cur = conn.cursor()
        cur.execute(f'SELECT * FROM "{table}" LIMIT {n}')
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def serialize_schema(
    entry: dict,
    db_path: Optional[str] = None,
    include_samples: bool = False,
    n_samples: int = 3,
) -> str:
    """Render one database's schema as text.

    include_samples=True appends a few example rows per table, which measurably
    helps the model pick the right columns and value formats. It needs the DB
    file (db_path), so it's off by default; flip it on as an easy accuracy lever.
    """
    tables = entry["table_names_original"]
    columns = entry["column_names_original"]
    col_types = entry["column_types"]
    pk_set = set(entry.get("primary_keys", []))

    # Group columns by their table index (skipping the leading "*").
    cols_by_table: dict[int, list[tuple[int, str, str, bool]]] = {
        i: [] for i in range(len(tables))
    }
    for col_idx, (tbl_idx, col_name) in enumerate(columns):
        if tbl_idx < 0:
            continue
        cols_by_table[tbl_idx].append(
            (col_idx, col_name, col_types[col_idx], col_idx in pk_set)
        )

    lines: list[str] = [f"# Database: {entry['db_id']}", ""]
    for tbl_idx, table in enumerate(tables):
        lines.append(f"Table: {table}")
        for _, col_name, col_type, is_pk in cols_by_table[tbl_idx]:
            tag = " [PK]" if is_pk else ""
            lines.append(f"  - {col_name} ({col_type}){tag}")
        if include_samples and db_path:
            rows = _sample_rows(db_path, table, n_samples)
            if rows:
                lines.append(f"  sample rows: {rows}")
        lines.append("")

    fks = entry.get("foreign_keys", [])
    if fks:
        lines.append("Foreign keys:")
        for a, b in fks:
            ta, ca = columns[a]
            tb, cb = columns[b]
            lines.append(f"  - {tables[ta]}.{ca} -> {tables[tb]}.{cb}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
