"""FastAPI service wrapping the fine-tuned text-to-SQL model.

Endpoints:
  GET  /health   -> liveness + which model/adapter is loaded
  POST /generate -> {question, schema? , db_id?} -> {sql, latency_ms, tokens}
  POST /query    -> like /generate, but also EXECUTES the SQL against the db_id's
                    SQLite file and returns the rows (handy for a live demo)
  GET  /metrics  -> latency percentiles, token averages, error rate

Schema handling — two ways to tell the service about the database:
  - pass `schema` directly (any text schema), or
  - pass `db_id` and the service serializes it from the Spider tables.json at
    SPIDER_ROOT (same serialization used in training, so quality matches).

Config via environment variables:
  MODEL_NAME     base model repo (default Qwen/Qwen2.5-Coder-1.5B-Instruct)
  ADAPTER_PATH   path to the fine-tuned LoRA adapter (default runs/qwen1.5b-lora/final)
  SPIDER_ROOT    Spider dir, enables db_id lookup + /query execution
  INCLUDE_SAMPLES "1" to add sample rows to serialized schemas (match training!)
  LOAD_IN_4BIT   "1" to load the base in 4-bit (for big models / tight VRAM)
  LOG_PATH       request log (default logs/requests.jsonl)

Run:  uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from serve.monitoring import Monitor

# ---- config ----
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "runs/qwen1.5b-lora/final")
SPIDER_ROOT = os.environ.get("SPIDER_ROOT")  # optional
INCLUDE_SAMPLES = os.environ.get("INCLUDE_SAMPLES", "1") == "1"
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"
LOG_PATH = os.environ.get("LOG_PATH", "logs/requests.jsonl")

app = FastAPI(title="text-to-sql", version="1.0")
monitor = Monitor(log_path=LOG_PATH)

# Lazily populated at startup so import stays cheap (and testable without torch).
_model = None
_tokenizer = None
_schema_cache: dict[str, str] = {}
_tables = None


class GenerateRequest(BaseModel):
    question: str
    schema_text: Optional[str] = None   # raw schema text; OR give db_id
    db_id: Optional[str] = None


class GenerateResponse(BaseModel):
    sql: str
    db_id: Optional[str] = None
    latency_ms: float
    input_tokens: int
    output_tokens: int


def _resolve_schema(req: GenerateRequest) -> str:
    """Return schema text from the request, or serialize it from db_id."""
    if req.schema_text:
        return req.schema_text
    if req.db_id:
        if SPIDER_ROOT is None:
            raise HTTPException(400, "db_id given but SPIDER_ROOT is not set on the server")
        if req.db_id not in _schema_cache:
            from src.data.schema import serialize_schema
            if req.db_id not in _tables:
                raise HTTPException(404, f"unknown db_id: {req.db_id}")
            db_path = os.path.join(SPIDER_ROOT, "database", req.db_id, f"{req.db_id}.sqlite")
            _schema_cache[req.db_id] = serialize_schema(
                _tables[req.db_id], db_path=db_path, include_samples=INCLUDE_SAMPLES
            )
        return _schema_cache[req.db_id]
    raise HTTPException(400, "provide either schema_text or db_id")


@app.on_event("startup")
def _load():
    global _model, _tokenizer, _tables
    # Imported here so the module imports without torch (keeps tests light).
    from src.models.baseline import BaselineModel

    adapter = ADAPTER_PATH if ADAPTER_PATH and os.path.isdir(ADAPTER_PATH) else None
    _model = BaselineModel(
        model_name=MODEL_NAME, adapter_path=adapter, load_in_4bit=LOAD_IN_4BIT,
    )
    _tokenizer = _model.tokenizer
    if SPIDER_ROOT:
        from src.data.schema import load_tables
        globals()["_tables"] = load_tables(os.path.join(SPIDER_ROOT, "tables.json"))


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "loading",
        "model": MODEL_NAME,
        "adapter": ADAPTER_PATH if _model is not None else None,
        "spider_root_configured": SPIDER_ROOT is not None,
    }


@app.get("/metrics")
def metrics():
    return monitor.metrics()


def _generate(question: str, schema_text: str, db_id: Optional[str]):
    """Shared generation path: time it, count tokens, record to the monitor."""
    if _model is None:
        raise HTTPException(503, "model still loading")
    t0 = time.perf_counter()
    try:
        sql = _model.generate_sql_batch([(schema_text, question)], batch_size=1)[0]
    except Exception as e:  # noqa: BLE001
        latency = (time.perf_counter() - t0) * 1000
        monitor.record(question, None, latency, 0, 0, db_id=db_id, error=str(e))
        raise HTTPException(500, f"generation failed: {e}")
    latency = (time.perf_counter() - t0) * 1000

    # Token accounting for cost/throughput monitoring.
    in_tok = len(_tokenizer(schema_text + question)["input_ids"])
    out_tok = len(_tokenizer(sql)["input_ids"])
    monitor.record(question, sql, latency, in_tok, out_tok, db_id=db_id)
    return sql, latency, in_tok, out_tok


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    schema_text = _resolve_schema(req)
    sql, latency, in_tok, out_tok = _generate(req.question, schema_text, req.db_id)
    return GenerateResponse(sql=sql, db_id=req.db_id, latency_ms=round(latency, 1),
                            input_tokens=in_tok, output_tokens=out_tok)


@app.post("/query")
def query(req: GenerateRequest):
    """Generate SQL and also execute it against the db_id's SQLite file."""
    if not req.db_id:
        raise HTTPException(400, "/query needs a db_id to execute against")
    if SPIDER_ROOT is None:
        raise HTTPException(400, "SPIDER_ROOT not set; cannot execute")
    schema_text = _resolve_schema(req)
    sql, latency, in_tok, out_tok = _generate(req.question, schema_text, req.db_id)

    from src.eval.execution import execute_sql
    db_path = os.path.join(SPIDER_ROOT, "database", req.db_id, f"{req.db_id}.sqlite")
    res = execute_sql(db_path, sql, timeout_s=15.0)
    return {
        "sql": sql,
        "db_id": req.db_id,
        "latency_ms": round(latency, 1),
        "rows": res.rows if res.ok else None,
        "execution_error": res.error,
        "timed_out": res.timed_out,
    }
