#!/usr/bin/env python3
"""Trace-tail completion: LLM-powered behavioral analysis of RISC-V trace logs.

Sends the last N retired instructions to a completion LLM for behavioral
analysis. The LLM identifies execution patterns (tight loops, CSR transitions,
exception sequences, branch behavior) that serve as semantic signatures for
cluster merge/split decisions.

Trace is factual (what the CPU did), not interpretive → less hallucination risk.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Sequence

TAB_RE = re.compile(
    r"^\s*\d+\s+\d+\s+(?P<pc>[0-9a-fA-F]{6,16})\s+"
    r"[0-9a-fA-F]+\s+(?P<decoded>.+?)(?:\s{2,}.*)?$",
    re.IGNORECASE,
)
OPCODE_RE = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9_.]*)", re.IGNORECASE)

TAIL_LINES = 80  # last N trace lines to send to LLM


def _read_trace_tail(trace_path: Path, n_lines: int = TAIL_LINES) -> list[str]:
    """Read last N lines of a trace file (.log or .log.gz). Returns decoded instruction lines."""
    opener = gzip.open if trace_path.suffix == ".gz" else open
    mode = "rt" if trace_path.suffix == ".gz" else "r"
    try:
        with opener(trace_path, mode, encoding="utf-8", errors="replace") as fh:
            all_lines = [line.rstrip("\n") for line in fh if line.strip()]
    except (OSError, gzip.BadGzipFile):
        return []

    # Parse trace lines, keep only decoded instructions
    decoded = []
    for line in all_lines[-n_lines * 3:]:  # oversample then trim
        m = TAB_RE.match(line)
        if m:
            decoded.append(m.group("decoded").strip())
    return decoded[-n_lines:]


def _format_trace_for_llm(instructions: list[str], max_instructions: int = 80) -> str:
    """Format decoded trace instructions as a compact RISC-V execution log."""
    if not instructions:
        return "(no trace data)"
    tail = instructions[-max_instructions:]
    return "\n".join(f"{i:4d}  {instr}" for i, instr in enumerate(tail, 1))


def _load_completion_config() -> dict | None:
    raw = os.getenv("LLM_MODEL_CONFIG", "").strip()
    if not raw: return None
    try:
        import yaml; data = yaml.safe_load(raw)
    except Exception: return None
    if not isinstance(data, dict): return None
    section = data.get("completion") or data.get("chat")
    if not isinstance(section, dict): return None
    cfg = section.get("config") if isinstance(section.get("config"), dict) else section
    model = section.get("model_name") or cfg.get("model")
    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    api_key_env = cfg.get("api_key_env")
    if not api_key and api_key_env: api_key = os.getenv(str(api_key_env), "")
    if not model or not base_url: return None
    return {"model": str(model), "base_url": str(base_url),
            "api_key": str(api_key or "dummy"),
            "timeout": float(cfg.get("timeout", 30)), "max_tokens": int(cfg.get("max_tokens", 384))}


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"trace|{model}|v1|{prompt}".encode()).hexdigest()[:16]


TRACE_PROMPT = """Analyze these last retired RISC-V instructions from a failing Ibex RISC-V CPU test.
Identify the ROOT CAUSE hypothesis — what design bug could produce this execution pattern?
Return strict JSON only:
{"root_cause":"<hypothesis slug>","stage":"fetch|decode|execute|memory|writeback|csr|debug|interrupt",
"mechanism":"<how the bug manifests>","end_state":"normal_exit|tight_loop|exception|csr_trap|debug_halt|timeout|unknown",
"pc_region":"<coarsened PC>","opcode_signature":"<key opcode pattern>",
"loop_type":"none|infinite|branch_mispredict|csr_loop|debug_loop",
"confidence":"low|medium|high","notes":"one sentence"}

Instructions:"""

DEEP_TRACE_PROMPT = """You are analyzing RISC-V CPU trace logs from regression tests of a MODIFIED Ibex core
that contains injected hardware bugs. The trace shows the last retired instructions before failure.

Your task: hypothesize what ROOT CAUSE BUG in the CPU microarchitecture could produce this trace pattern.
Think about: pipeline stage, instruction class, CSR state, debug mode, interrupt handling, memory access.

Return strict JSON only (no markdown):
{"root_cause":"<concise_hypothesis_slug>",
"stage":"fetch|decode|execute|memory|writeback|csr|debug|interrupt|multistage",
"mechanism":"<specific failure mechanism>",
"affected_ops":["<opcode>"],
"end_state":"normal_tail|tight_loop|exception_trap|csr_fault|debug_stall|timeout_kill|other",
"pc_signature":"<PC or region>",
"is_deterministic":true,
"confidence":"low|medium|high",
"notes":"one sentence"}

Instructions:"""


def _parse_trace_json(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", str(text))
    if not m: return {}
    try: return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError): return {}


def analyze_trace_root_cause(
    trace_paths: list[Path | None],
    cache_dir: Path,
    timeout_sec: float = 120,
) -> list[dict]:
    """Deep root-cause analysis of trace tails via completion LLM.

    Returns list of dicts with root_cause, stage, mechanism, affected_ops, etc.
    """
    config = _load_completion_config()
    if not config:
        return [{} for _ in trace_paths]

    from openai import OpenAI
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"],
                    timeout=min(config["timeout"], timeout_sec))
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for path in trace_paths:
        if path is None or not Path(path).exists():
            results.append({})
            continue

        instructions = _read_trace_tail(Path(path))
        trace_text = _format_trace_for_llm(instructions)
        prompt = DEEP_TRACE_PROMPT + "\n" + trace_text

        ck = _cache_key(config["model"], prompt)
        cp = cache_dir / f"deep_{ck}.json"
        if cp.is_file():
            try:
                results.append(json.loads(cp.read_text()))
                continue
            except (json.JSONDecodeError, OSError):
                pass

        try:
            resp = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=config["max_tokens"],
            )
            text = resp.choices[0].message.content or ""
            parsed = _parse_trace_json(text)
            parsed["_status"] = "ok"
        except Exception as exc:
            parsed = {"_status": f"error:{type(exc).__name__}"}
        cp.write_text(json.dumps(parsed, ensure_ascii=False))
        results.append(parsed)

    return results


def analyze_trace_tail(
    trace_paths: list[Path | None],
    cache_dir: Path,
    timeout_sec: float = 120,
) -> list[dict]:
    """Run completion LLM on trace tails for selected cases.

    Args:
        trace_paths: list of trace file paths (or None for missing)
        cache_dir: completion cache directory
        timeout_sec: total time budget

    Returns: list of parsed behavioral analyses (same order as input)
    """
    config = _load_completion_config()
    if not config:
        return [{} for _ in trace_paths]

    from openai import OpenAI
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"],
                    timeout=min(config["timeout"], timeout_sec))
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for path in trace_paths:
        if path is None or not Path(path).exists():
            results.append({})
            continue

        instructions = _read_trace_tail(Path(path))
        trace_text = _format_trace_for_llm(instructions)
        prompt = TRACE_PROMPT + "\n" + trace_text

        ck = _cache_key(config["model"], prompt)
        cp = cache_dir / f"trace_{ck}.json"
        if cp.is_file():
            try:
                results.append(json.loads(cp.read_text()))
                continue
            except (json.JSONDecodeError, OSError):
                pass

        try:
            resp = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=config["max_tokens"],
            )
            text = resp.choices[0].message.content or ""
            parsed = _parse_trace_json(text)
            parsed["_status"] = "ok"
        except Exception as exc:
            parsed = {"_status": f"error:{type(exc).__name__}"}
        cp.write_text(json.dumps(parsed, ensure_ascii=False))
        results.append(parsed)

    return results


def trace_behavior_match(a: dict, b: dict) -> bool:
    """Check if two trace completion results suggest the same bug behavior."""
    if not a or not b: return False
    behavior_a = str(a.get("behavior", "")).strip().lower()
    behavior_b = str(b.get("behavior", "")).strip().lower()
    anomaly_a = str(a.get("anomaly", "")).strip().lower()
    anomaly_b = str(b.get("anomaly", "")).strip().lower()
    if behavior_a == behavior_b and behavior_a not in ("", "normal", "other"):
        if anomaly_a == anomaly_b and anomaly_a not in ("", "none"):
            return True
    return False


def trace_root_cause_match(a: dict, b: dict, level: str = "mechanism") -> bool:
    """Check if two trace root-cause analyses suggest the same underlying bug.

    Args:
        a, b: parsed completion results from analyze_trace_root_cause()
        level: "strict" (same root_cause slug), "mechanism" (same stage+end_state),
               "relaxed" (same stage)

    Returns True if the analyses are consistent with the same root bug.
    """
    if not a or not b: return False
    if not a.get("_status") == "ok" or not b.get("_status") == "ok":
        return False
    if a.get("confidence") == "low" or b.get("confidence") == "low":
        return False

    rc_a = str(a.get("root_cause", "")).strip().lower()
    rc_b = str(b.get("root_cause", "")).strip().lower()
    stage_a = str(a.get("stage", "")).strip().lower()
    stage_b = str(b.get("stage", "")).strip().lower()
    end_a = str(a.get("end_state", "")).strip().lower()
    end_b = str(b.get("end_state", "")).strip().lower()
    mech_a = str(a.get("mechanism", "")).strip().lower()
    mech_b = str(b.get("mechanism", "")).strip().lower()

    if level == "strict":
        return rc_a == rc_b and rc_a not in ("", "unknown")
    elif level == "mechanism":
        return stage_a == stage_b and end_a == end_b and stage_a not in ("", "unknown")
    else:  # relaxed
        return stage_a == stage_b and stage_a not in ("", "unknown")


def find_trace_column(fields: Sequence[str]) -> str | None:
    """Find trace log column in input CSV fields."""
    norm = {"".join(ch for ch in f.lower() if ch.isalnum()): f for f in fields}
    for cand in ("tracelog", "tracelog.gz", "trace", "trace_log", "tracefile"):
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        if key in norm:
            return norm[key]
    for f in fields:
        if "trace" in "".join(ch for ch in f.lower() if ch.isalnum()):
            return f
    return None


def resolve_trace_path(input_csv: Path, value: str | None) -> Path | None:
    """Resolve trace log path relative to input CSV."""
    if not value: return None
    path = Path(str(value))
    return (input_csv.parent / path).resolve() if not path.is_absolute() else path
