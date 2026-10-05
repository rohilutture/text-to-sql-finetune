"""Tiny client to smoke-test the running service.

  # make sure the server is up first:  uvicorn serve.app:app --port 8000
  python serve/client_demo.py --question "How many singers are there?" --db-id concert_singer
  python serve/client_demo.py --question "List all stadium names" --schema-file my_schema.txt
  python serve/client_demo.py --metrics
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--question")
    ap.add_argument("--db-id", default=None)
    ap.add_argument("--schema-file", default=None, help="file with raw schema text")
    ap.add_argument("--execute", action="store_true", help="use /query (runs the SQL)")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--health", action="store_true")
    args = ap.parse_args()

    if args.health:
        print(json.dumps(_get(args.host + "/health"), indent=2)); return
    if args.metrics:
        print(json.dumps(_get(args.host + "/metrics"), indent=2)); return

    payload: dict = {"question": args.question}
    if args.db_id:
        payload["db_id"] = args.db_id
    if args.schema_file:
        payload["schema_text"] = open(args.schema_file, encoding="utf-8").read()

    endpoint = "/query" if args.execute else "/generate"
    print(json.dumps(_post(args.host + endpoint, payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
