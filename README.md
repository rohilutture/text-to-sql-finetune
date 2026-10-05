# text-to-sql-finetune

Fine-tune a small open model to translate natural-language questions into SQL,
**measure** it against a prompt-only baseline, and **ship** it. The interview
story: "I built the prompt-based version before; here I fine-tuned my own,
measured the gap on execution accuracy, and deployed it end to end."

## Why the eval harness comes first

The whole project rests on one number: the execution-accuracy delta between the
prompt-only baseline and the fine-tuned model. If you build evaluation last (as
most projects do), you finish training and then can't trust the harness that's
supposed to prove it worked. So this repo starts with the eval harness and the
baseline. Every training run afterward gets an instant, honest read.

## Quickstart (Linux + RTX 4090/5090)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
# torch build matching your driver (Ada/Blackwell -> cu128):
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# sanity check the GPU (want cuda True and a capability tuple, no kernel error):
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Then get Spider (see `scripts/download_spider.md`) into `data/spider_data`, and:

```bash
# smoke test (20 examples) — confirms the pipeline runs end to end:
python scripts/run_baseline.py --spider-root data/spider_data --limit 20

# full dev baseline — your "before" number:
python scripts/run_baseline.py --spider-root data/spider_data --include-samples \
    --predictions-out runs/baseline_dev.jsonl --result-out runs/baseline_dev.json
```

Read the failures (`head runs/baseline_dev.jsonl`) — *how* the prompt-only model
fails tells you what fine-tuning needs to fix.

## Fine-tune

Run in order. **Do the sanity step first** — a minute now saves a wasted run.

```bash
# 0. overfit ONE batch; loss should fall toward ~0. If it plateaus, there's a
#    bug (masking / LR / frozen params) — stop and fix before a real run.
python scripts/train.py --spider-root data/spider_data --overfit-one-batch --include-samples

# 1. real run on the 1.5B:
python scripts/train.py --spider-root data/spider_data \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct --include-samples \
    --epochs 3 --batch-size 8 --grad-accum 4 --out runs/qwen1.5b-lora

# 2. evaluate the fine-tuned adapter (keep --include-samples identical to training!):
python scripts/run_eval.py --spider-root data/spider_data \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --adapter-path runs/qwen1.5b-lora/final --include-samples \
    --predictions-out runs/finetuned_dev.jsonl --result-out runs/finetuned_dev.json

# 3. before/after table:
python scripts/compare.py --baseline runs/baseline_dev.json --finetuned runs/finetuned_dev.json
```

Bigger headline run (QLoRA 4-bit + gradient checkpointing; fits a 14B on 24GB):

```bash
python scripts/train.py --spider-root data/spider_data \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --load-in-4bit --grad-checkpointing --include-samples \
    --epochs 3 --batch-size 4 --grad-accum 8 --out runs/qwen14b-qlora
# then run_eval.py with --load-in-4bit and --adapter-path runs/qwen14b-qlora/final
```

Key knobs: effective batch = `batch-size * grad-accum`; `--lora-r`/`--lora-alpha`
size the adapter; `--save-every`/`--resume-from` handle checkpointing. Full list:
`python scripts/train.py --help`.

## Results (Spider dev, 1034 examples)

Fine-tuning Qwen2.5-Coder-1.5B with LoRA, vs. the same model prompt-only:

| Tier | Prompt-only | Fine-tuned | Δ |
|---|---|---|---|
| **Overall** | **56.7%** | **74.2%** | **+17.5** |
| Easy | 71.4% | 84.4% | +13.0 |
| Medium | 55.0% | 79.2% | +24.2 |
| Hard | 57.7% | 68.6% | +10.9 |
| Extra | 28.0% | 44.8% | +16.8 |

Failure analysis explains the gain: of the baseline's 444 failures, half were
*invalid SQL* (hallucinated columns / unnecessary joins from weak schema-linking).
Fine-tuning cut invalid-SQL errors by 59% (222 → 92) and wrong-row errors by 21%
(222 → 175) — the predicted mechanism, measured.

## Serve it (Phase 3)

A FastAPI service wraps the fine-tuned model with request logging and a metrics
endpoint. Run it locally (GPU recommended):

```bash
pip install -r serve/requirements.txt
export ADAPTER_PATH=runs/qwen1.5b-lora/final
export SPIDER_ROOT=data/spider_data   # enables db_id lookup + /query execution
export INCLUDE_SAMPLES=1               # MUST match how you trained
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

Endpoints: `GET /health`, `POST /generate`, `POST /query` (generate + execute),
`GET /metrics` (latency p50/p95/p99, token averages, error rate). Try it:

```bash
python serve/client_demo.py --question "How many singers are there?" --db-id concert_singer --execute
python serve/client_demo.py --metrics
```

Every request is appended to `logs/requests.jsonl` (timestamp, latency, token
counts, question, SQL, errors) so you can watch for quality drift.

### Docker

```bash
docker build -t text-to-sql .
# GPU (needs nvidia-container-toolkit; on Windows, Docker Desktop + WSL2):
docker run --gpus all -p 8000:8000 \
    -v $HOME/.cache/huggingface:/app/.hf_cache \
    text-to-sql
```

The `-v` mount reuses your Hugging Face cache so the base model isn't
re-downloaded on each run. Deploy the same image to a HF Spaces GPU tier or a
small cloud GPU instance.

## Layout

```
src/
  prompts.py            prompt template + SQL extraction (shared)
  data/
    schema.py           tables.json -> schema text
    spider.py           load examples, pair with DB + schema
    sft_dataset.py      tokenize + loss-mask + pad for training
  eval/
    execution.py        execute + result-set comparison
    hardness.py         difficulty tiers (heuristic)
    harness.py          run, aggregate, comparison table
  models/
    baseline.py         HF inference — baseline OR fine-tuned adapter
scripts/
  download_spider.md    get the dataset
  run_baseline.py       baseline eval
  train.py              the hand-written training loop
  run_eval.py           eval baseline or fine-tuned (--adapter-path)
  compare.py            render the before/after table
serve/
  app.py                FastAPI service (/generate, /query, /metrics, /health)
  monitoring.py         request logging + latency/token metrics
  client_demo.py        tiny client to smoke-test the running service
  requirements.txt      serving-only deps (fastapi, uvicorn, pydantic)
Dockerfile              CUDA image: bakes in the adapter, serves on :8000
```

## Remaining polish (optional)

- Cross-check the headline number against the official Spider test-suite
  evaluator (github.com/taoyds/test-suite-sql-eval) for the writeup.
- Scale the headline result with a 14B QLoRA run (`--load-in-4bit`).