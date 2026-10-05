# Getting Spider

Spider needs the actual SQLite **databases** (not just the question/SQL pairs),
because this project scores *execution accuracy* — it runs each query. The
popular `xlangai/spider` HF dataset ships only the Q/SQL pairs, so use a re-host
that includes the databases.

## Easiest path (Linux, in your venv)

```bash
pip install -U huggingface_hub
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='HAL-9001/spider-databases', filename='spider_data.zip', repo_type='dataset', local_dir='data'))"
unzip -q data/spider_data.zip -d data
```

That gives you `data/spider_data/` containing `dev.json`, `train_spider.json`,
`tables.json`, and `database/<db_id>/<db_id>.sqlite`. Your `--spider-root` is the
folder that **directly** holds those — i.e. `data/spider_data`.

Official source for your writeup citation: https://yale-lily.github.io/spider
(this re-host is a mirror of it.)

## Verify before running anything

```bash
python - <<'PY'
import json, os
root = "data/spider_data"   # <-- your --spider-root
dev = json.load(open(os.path.join(root, "dev.json")))
db_ids = {e["db_id"] for e in dev}
missing = [d for d in db_ids
           if not os.path.exists(os.path.join(root, "database", d, f"{d}.sqlite"))]
print(f"dev examples: {len(dev)}  |  dev DBs: {len(db_ids)}  |  missing .sqlite: {len(missing)}")
if missing:
    print("MISSING:", missing[:10])
PY
```

Expect roughly `dev examples: 1034 | dev DBs: 20 | missing .sqlite: 0`. If missing
is 0, you're ready.

> Some redistributed Spider dev DBs have minor quirks. If a few gold queries fail
> to execute, the harness marks them `gold_failed` — a dataset artifact, not a
> model miss. You'll see them in the predictions JSONL.
