#!/usr/bin/env python3
"""Smoke-test the configured OpenAI-compatible completion endpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from completion_case_features import load_completion_config


def main() -> int:
    config = load_completion_config()
    if config is None:
        print(
            "No valid completion endpoint in LLM_MODEL_CONFIG. "
            "Set the configured api_key_env and export YAML content.",
            file=sys.stderr,
        )
        return 2

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=config["timeout"],
    )
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{
            "role": "user",
            "content": (
                'Return exactly this JSON object and nothing else: '
                '{"status":"ok","task":"eda_failure_analysis"}'
            ),
        }],
        temperature=0,
        max_tokens=64,
    )
    text = response.choices[0].message.content or ""
    print(json.dumps({
        "endpoint": config["base_url"],
        "model": config["model"],
        "response": text.strip(),
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
