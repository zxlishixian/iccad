#!/usr/bin/env python3
"""Populate completion feature caches with bounded request concurrency."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

import completion_case_features as ccf
import regr_fail_bucketing as rfb


_LOCAL = threading.local()


def _client(config: dict) -> OpenAI:
    client = getattr(_LOCAL, "client", None)
    if client is None:
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"],
        )
        _LOCAL.client = client
    return client


def _request(task: dict, config: dict, cache_dir: Path) -> str:
    cache_path = cache_dir / f"{task['digest']}.json"
    if cache_path.is_file():
        return "cached"
    system, user = ccf._prompt(task["evidence"])
    response = None
    for attempt in range(5):
        try:
            response = _client(config).chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=config["max_tokens"],
            )
            break
        except Exception as exc:
            if attempt == 4 or "429" not in str(exc):
                raise
            time.sleep(2 ** attempt)
    if response is None:
        raise RuntimeError("completion request did not return a response")
    feature = ccf._parse_json(
        response.choices[0].message.content or "", task["case_id"]
    )
    if feature.status != "ok":
        raise RuntimeError(f"invalid completion JSON for {task['case_id']}")
    cache_path.write_text(
        json.dumps(feature.__dict__, indent=2) + "\n", encoding="utf-8"
    )
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/regr_fail_completion_cache"),
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    config = ccf.load_completion_config()
    if config is None:
        raise RuntimeError("no valid completion endpoint in LLM_MODEL_CONFIG")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    for dataset in args.datasets:
        input_csv = (dataset / "input.csv").resolve()
        rows, fields = rfb.read_csv_rows(input_csv)
        for idx, row in enumerate(rows):
            case_id = ccf._case_id(row, fields, idx)
            evidence = ccf._compact_evidence(input_csv, row, fields)
            digest = hashlib.sha256(
                (config["model"] + "\n" + evidence).encode()
            ).hexdigest()
            tasks.append({
                "case_id": case_id,
                "evidence": evidence,
                "digest": digest,
            })

    counts = {"ok": 0, "cached": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(_request, task, config, args.cache_dir)
            for task in tasks
        ]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                counts[future.result()] += 1
            except Exception as exc:
                counts["error"] += 1
                print(
                    f"[error] {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if done % 10 == 0 or done == len(futures):
                print(f"[cache] {done}/{len(futures)} {counts}", flush=True)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
