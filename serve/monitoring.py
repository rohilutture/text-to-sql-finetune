"""Request/response monitoring for the text-to-SQL service.

Every prediction is logged as one JSON line with the fields you need to catch
quality drift in production: timestamp, latency, token counts, the input, and
the output (or error). This is the "basic monitoring" the project calls for —
latency, token counts, and logged inputs/outputs — kept deliberately simple
(append-only JSONL + in-memory aggregates) so it has no external dependencies.

In a bigger system you'd ship these events to Prometheus / a log aggregator;
the shape here is the same, so swapping the sink later is a small change.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Optional


class Monitor:
    def __init__(self, log_path: str = "logs/requests.jsonl", window: int = 500):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        # Rolling window of recent latencies/token counts for /metrics, so the
        # endpoint stays O(1) instead of re-reading the whole log file.
        self._latencies: deque[float] = deque(maxlen=window)
        self._out_tokens: deque[int] = deque(maxlen=window)
        self._total = 0
        self._errors = 0

    def record(
        self,
        question: str,
        sql: Optional[str],
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        db_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "db_id": db_id,
            "question": question,
            "sql": sql,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": error,
        }
        with self._lock:
            self._total += 1
            if error:
                self._errors += 1
            else:
                self._latencies.append(latency_ms)
                self._out_tokens.append(output_tokens)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def _pct(values: list[float], p: float) -> Optional[float]:
        if not values:
            return None
        s = sorted(values)
        k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
        return s[k]

    def metrics(self) -> dict:
        with self._lock:
            lat = list(self._latencies)
            out = list(self._out_tokens)
            total, errors = self._total, self._errors
        return {
            "total_requests": total,
            "errors": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "window_size": len(lat),
            "latency_ms": {
                "p50": self._pct(lat, 50),
                "p95": self._pct(lat, 95),
                "p99": self._pct(lat, 99),
                "max": max(lat) if lat else None,
            },
            "output_tokens_avg": round(sum(out) / len(out), 1) if out else None,
        }
