import sys, math
sys.path.insert(0, ".")
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.data.spider import load_split
from src.data.sft_dataset import SFTDataset, make_collate_fn

name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora).to("cuda")
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.train()

ex = load_split("data/spider_data", split="train", include_samples=True, limit=64)
ds = SFTDataset(ex, tok, max_length=2048)
dl = DataLoader(ds, batch_size=8, shuffle=False,
                collate_fn=make_collate_fn(tok.pad_token_id))
batch = {k: v.to("cuda") for k, v in next(iter(dl)).items()}

nvalid = int((batch["labels"] != -100).sum())
print("valid (unmasked) label tokens in batch:", nvalid, "/", batch["labels"].numel())

with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(**batch)
print("loss:", float(out.loss), "| loss.requires_grad:", out.loss.requires_grad)

out.loss.backward()

n = none = zero = 0
tot = 0.0
lora_b_grad = 0.0
for nm, p in model.named_parameters():
    if not p.requires_grad:
        continue
    n += 1
    if p.grad is None:
        none += 1
    else:
        g = float(p.grad.detach().float().norm())
        tot += g * g
        if g == 0.0:
            zero += 1
        if "lora_B" in nm:
            lora_b_grad += g
print(f"trainable tensors: {n} | grad is None: {none} | grad exactly 0: {zero}")
print(f"global grad norm: {math.sqrt(tot):.6f}")
print(f"sum of lora_B grad norms: {lora_b_grad:.6f}")

# Now actually take ONE optimizer step and see if loss moves.
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
opt.step(); opt.zero_grad()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out2 = model(**batch)
print(f"loss before step: {float(out.loss):.6f} | loss after 1 step: {float(out2.loss):.6f}")
