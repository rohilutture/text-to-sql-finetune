"""Turn Spider examples into training tensors for supervised fine-tuning (SFT).

The one subtle thing here: **loss masking**. For each example we build the full
token sequence = prompt (system + user + the "assistant:" header) followed by the
target SQL. But we only want the model to be *graded* on the SQL it should
produce — not on the schema and question we handed it for free. So the label for
every prompt token is set to -100, which PyTorch's cross-entropy ignores. Get
this wrong and you either train the model to regurgitate schemas (masking too
little) or train on nothing (masking too much) — and loss looks "fine" in both
cases, which is why it's the classic silent bug.

We find the masked region as the common token prefix between the prompt-only
encoding and the full encoding. That's robust even if the tokenizer merges the
boundary between the assistant header and the first SQL token.
"""

from __future__ import annotations

from typing import Optional

from ..prompts import build_messages

IGNORE_INDEX = -100  # cross-entropy ignores label positions equal to this


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build_example_ids(
    tokenizer,
    schema: str,
    question: str,
    target_sql: str,
    max_length: int = 2048,
) -> Optional[dict]:
    """Tokenize one example into {input_ids, labels}.

    Returns None if the full sequence exceeds max_length — we drop it rather than
    truncate, because truncating would either cut off the SQL we're training on
    or drop tables from the schema (both corrupt the example).
    """
    prompt_messages = build_messages(schema, question)
    # Render to STRINGS first, then tokenize ourselves. `apply_chat_template`
    # with tokenize=True returns different types across transformers versions
    # (list[int] vs BatchEncoding), which breaks downstream. Going through the
    # string and calling the tokenizer directly always yields a clean list[int].
    # add_special_tokens=False because the template string already contains the
    # model's special tokens (e.g. <|im_start|>) as text.
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )
    full_messages = prompt_messages + [{"role": "assistant", "content": target_sql}]
    full_text = tokenizer.apply_chat_template(
        full_messages, add_generation_prompt=False, tokenize=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_length:
        return None

    mask_len = _common_prefix_len(prompt_ids, full_ids)
    labels = [IGNORE_INDEX] * mask_len + full_ids[mask_len:]
    # Sanity: labels and inputs must line up 1:1.
    assert len(labels) == len(full_ids)
    return {"input_ids": full_ids, "labels": labels}


class SFTDataset:
    """A torch Dataset of pre-tokenized, loss-masked examples.

    We tokenize everything up front (Spider train is ~7k examples — trivial to
    hold in memory) and report how many were dropped for length.
    """

    def __init__(self, examples, tokenizer, max_length: int = 2048):
        # `examples` is a list of src.data.spider.Example
        self.records: list[dict] = []
        dropped = 0
        for ex in examples:
            built = build_example_ids(
                tokenizer, ex.schema_text, ex.question, ex.gold_sql, max_length
            )
            if built is None:
                dropped += 1
                continue
            self.records.append(built)
        self.dropped = dropped

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def _pad_batch(batch: list[dict], pad_id: int) -> dict:
    """Right-pad a batch to its longest sequence. Pure-python (unit-testable):
    returns lists; the collate wrapper turns them into tensors."""
    max_len = max(len(r["input_ids"]) for r in batch)
    input_ids, labels, attention_mask = [], [], []
    for r in batch:
        pad = max_len - len(r["input_ids"])
        input_ids.append(r["input_ids"] + [pad_id] * pad)
        labels.append(r["labels"] + [IGNORE_INDEX] * pad)  # padded labels ignored
        attention_mask.append([1] * len(r["input_ids"]) + [0] * pad)  # ignore pads
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class Collator:
    """Picklable collate callable for DataLoader.

    A plain closure can't be pickled, and Windows DataLoader workers use `spawn`
    (which pickles the collate_fn to send to each worker) — so a closure crashes
    there. A top-level class with __call__ pickles fine on every platform.
    """

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        padded = _pad_batch(batch, self.pad_id)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in padded.items()}


def make_collate_fn(pad_id: int):
    """Return a picklable collate_fn for DataLoader (works with num_workers>0)."""
    return Collator(pad_id)
