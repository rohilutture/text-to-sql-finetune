"""The prompt template and SQL extraction, shared across baseline and fine-tune.

Keeping this in one module is what makes the before/after comparison fair: the
prompt-only baseline and the fine-tuned model are prompted the same way, so any
accuracy delta comes from the weights, not from a prompt you happened to tweak.

When you build the fine-tuning data, format each training example with
`build_messages` too, so train-time and eval-time prompting match.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite SQL. "
    "Given a database schema and a question, output a single valid SQLite query "
    "that answers it. Output only the SQL, with no explanation and no markdown."
)

USER_TEMPLATE = """Schema:
{schema}

Question: {question}

SQL:"""


def build_messages(schema: str, question: str) -> list[dict]:
    """Chat-format messages for an instruct model's chat template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema, question=question)},
    ]


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model completion.

    Handles the common failure modes: markdown code fences, a leading "SQL:"
    label, and trailing prose after the statement. Falls back to the raw text so
    a malformed-but-runnable query still gets a chance to execute (and a truly
    broken one just counts as wrong, which is the honest outcome).
    """
    m = _FENCE.search(text)
    if m:
        text = m.group(1)

    text = text.strip()
    # Drop a leading "SQL:" the model sometimes echoes.
    text = re.sub(r"^\s*sql\s*:\s*", "", text, flags=re.IGNORECASE)

    # Keep only up to the first statement terminator if there's trailing chatter.
    if ";" in text:
        text = text.split(";")[0]

    return text.strip()
