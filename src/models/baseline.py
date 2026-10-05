"""Prompt-only baseline OR fine-tuned model — same class, same interface.

This is your "before" when adapter_path is None, and your "after" when you point
adapter_path at a trained LoRA checkpoint. The eval harness calls
`generate_sql_batch` and doesn't care which one it's scoring — that's what keeps
the comparison apples-to-apples: identical prompt, identical decoding, only the
weights differ.

Runs the 1.5B comfortably on the RTX 4090. For a bigger headline run, pass
load_in_4bit=True (QLoRA-style 4-bit inference) or just let it use bf16.
"""

from __future__ import annotations

from typing import Optional


class BaselineModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        adapter_path: Optional[str] = None,   # <- set this to load a fine-tuned LoRA
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        max_new_tokens: int = 256,
    ):
        # Imported lazily so the rest of the harness (data, eval) imports and
        # unit-tests fine on a machine without torch installed.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..prompts import build_messages, extract_sql
        self._build_messages = build_messages
        self._extract_sql = extract_sql

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding so batched generation keeps the newest tokens aligned.
        self.tokenizer.padding_side = "left"

        model_kwargs = {"torch_dtype": getattr(torch, dtype)}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Load the fine-tuned LoRA adapter on top of the frozen base, if given.
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            # Merging folds the adapter into the base weights for faster
            # inference. Skip it for 4-bit (can't merge into quantized weights).
            if not load_in_4bit:
                model = model.merge_and_unload()

        if not load_in_4bit:
            model = model.to(device)
        model.eval()
        self.model = model
        self.device = device

    def _render(self, schema: str, question: str) -> str:
        messages = self._build_messages(schema, question)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_sql_batch(
        self, items: list[tuple[str, str]], batch_size: int = 8
    ) -> list[str]:
        """items: list of (schema_text, question). Returns extracted SQL strings."""
        import torch

        outputs: list[str] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            prompts = [self._render(s, q) for s, q in chunk]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=4096,
            ).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # greedy: deterministic, reproducible
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # Slice off the prompt tokens; decode only the new completion.
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend(self._extract_sql(d) for d in decoded)
        return outputs
