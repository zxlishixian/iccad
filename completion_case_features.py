#!/usr/bin/env python3
"""Cached completion-LLM case features for experimental training.

The completion model never outputs buckets. It converts a locally extracted,
de-identified whitelist of sim/regr signals into a canonical JSON record.
Gold, meta, trace, paths, addresses, identifiers, and raw log lines are not
sent to the completion endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import regr_fail_bucketing as rfb


CANONICAL_KEYS = (
    "failure_stage",
    "primary_mechanism",
    "mismatch_object",
    "dut_state",
    "state_transition",
    "trigger",
)


@dataclass
class CompletionCaseFeature:
    case_id: str
    status: str
    failure_stage: str = ""
    primary_mechanism: str = ""
    mismatch_object: str = ""
    dut_state: str = ""
    state_transition: str = ""
    trigger: str = ""
    candidate_explanations: tuple[str, ...] = ()
    evidence_tags: tuple[str, ...] = ()
    conflict_tags: tuple[str, ...] = ()
    confidence: float = 0.0
    rationale: str = ""


def _normalize(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_:+.-]+", "_", text)
    return text.strip("_")[:96]


def load_completion_config() -> dict | None:
    raw = os.getenv("LLM_MODEL_CONFIG", "").strip()
    if not raw:
        return None
    try:
        import yaml

        data = yaml.safe_load(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    section = data.get("completion") or data.get("chat") or data.get("generation")
    if not isinstance(section, dict):
        return None
    cfg = section.get("config") if isinstance(section.get("config"), dict) else section
    model = section.get("model_name") or cfg.get("model")
    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    api_key_env = cfg.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(str(api_key_env), "")
    if not model or not base_url:
        return None
    return {
        "model": str(model),
        "base_url": str(base_url),
        "api_key": str(api_key),
        "timeout": float(cfg.get("timeout", 60.0)),
        "max_tokens": int(cfg.get("max_tokens", 320)),
    }


def _case_id(row: dict, fields: Sequence[str], idx: int) -> str:
    norm = {"".join(ch for ch in f.lower() if ch.isalnum()): f for f in fields}
    for key in ("case", "caseid", "id"):
        field = norm.get(key)
        if field and str(row.get(field, "")).strip():
            return str(row[field]).strip()
    return f"case_{idx + 1:06d}"


def _compact_evidence(input_csv: Path, row: dict, fields: Sequence[str]) -> str:
    sim_col = rfb.pick_column(fields, "sim")
    regr_col = rfb.pick_column(fields, "regr")
    sim_path = rfb.resolve_log_path(input_csv, row.get(sim_col) if sim_col else None)
    regr_path = rfb.resolve_log_path(input_csv, row.get(regr_col) if regr_col else None)
    sim_text, sim_status = rfb.read_log_sample(sim_path)
    regr_text, regr_status = rfb.read_log_sample(regr_path)
    sim_lines = rfb.select_lines(sim_text) if sim_status == "ok" else []
    regr_lines = rfb.select_lines(regr_text) if regr_status == "ok" else []

    if os.getenv("ICCAD_COMPLETION_EVIDENCE_MODE", "structured") == "raw":
        blocks: list[str] = []
        for name, status, selected in (
            ("SIM", sim_status, sim_lines),
            ("REGR", regr_status, regr_lines),
        ):
            if status != "ok":
                blocks.append(f"{name}: <{status}>")
                continue
            signal = [
                line for line in selected
                if re.search(
                    r"fatal|error|failed|mismatch|timeout|interrupt|irq|debug|"
                    r"exception|illegal|ebreak|dret|mret|csr|mcause|retired",
                    line,
                    re.IGNORECASE,
                )
            ]
            if not signal:
                signal = list(selected[-20:])
            blocks.append(f"{name}:\n" + "\n".join(signal[:80]))
        return "\n\n".join(blocks)[:12000]

    import pairwise_llm_features as plf

    primary_tokens = rfb.extract_primary_signature({}, {}, sim_lines, regr_lines)
    info = plf._extract_rich_case_info(sim_lines, regr_lines, primary_tokens)
    joined = (sim_text + "\n" + regr_text).lower()
    keyword_flags = {
        "irq": bool(re.search(r"\b(?:irq|interrupt|handling_irq)\b", joined)),
        "debug": bool(re.search(r"\bdebug\b|in_debug_mode", joined)),
        "dret": bool(re.search(r"\bdret\b", joined)),
        "mret": bool(re.search(r"\bmret\b", joined)),
        "ebreak": bool(re.search(r"\bebreak\b", joined)),
        "csr": bool(re.search(r"\bcsr\b|mcause|mstatus|mepc|dcsr", joined)),
        "illegal_instruction": "illegal instruction" in joined,
        "timeout": "timeout" in joined,
        "exception": "exception" in joined,
        "pc_mismatch": "pc mismatch" in joined,
        "register_mismatch": "register write data mismatch" in joined,
        "memory_mismatch": bool(
            re.search(r"memory mismatch|\b(?:load|store)\b.{0,40}mismatch", joined)
        ),
    }
    def local_events(lines: list[str], limit: int = 3) -> list[str]:
        pattern = re.compile(
            r"fatal|error|failed|mismatch|timeout|interrupt|irq|debug|exception|"
            r"illegal|ebreak|dret|mret|csr|mcause|retired|log-extract",
            re.IGNORECASE,
        )
        ranked: list[tuple[int, int, str]] = []
        for idx, line in enumerate(lines):
            if not pattern.search(line):
                continue
            score = 3 if re.search(r"fatal|mismatch|timeout", line, re.I) else 2
            start, stop = max(0, idx - 2), min(len(lines), idx + 3)
            context = " | ".join(lines[start:stop])
            context = re.sub(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{8,16}\b", "<ADDR>", context)
            context = re.sub(r"\b\d+\b", "<NUM>", context)
            context = re.sub(r"\s+", " ", context).strip()[:700]
            ranked.append((-score, idx, context))
        return [text for _, _, text in sorted(ranked)[:limit]]

    payload = {
        "sim_status": sim_status,
        "regr_status": regr_status,
        "primary_type": _normalize(info.get("primary_type")) or "unknown",
        "mismatch_type": _normalize(info.get("mismatch_type")) or "unknown",
        "primary_signature": _normalize(info.get("primary_signature")) or "unknown",
        "op_pair": _normalize(info.get("op_pair")) or "unknown",
        "register_name": _normalize(info.get("register_name")) or "unknown",
        "has_uvm_fatal": bool(info.get("has_uvm_fatal")),
        "has_uvm_error": bool(info.get("has_uvm_error")),
        "has_regr_mismatch": bool(info.get("has_regr_mismatch")),
        "signals": keyword_flags,
        "top_sim_events": local_events(sim_lines),
        "top_regr_events": local_events(regr_lines),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _prompt(evidence: str) -> tuple[str, str]:
    system = (
        "You are an EDA regression-failure analyst. Convert the supplied sim/regr "
        "evidence into canonical root-cause attributes. Do not cluster cases and "
        "do not guess a bug id. Return one JSON object only."
    )
    user = f"""Analyze this single failure case.

Use concise snake_case values. If evidence is absent, use "unknown".
Schema:
{{
  "failure_stage": "fetch|decode|execute|memory|csr|debug|irq|retire|cosim|timeout|unknown",
  "primary_mechanism": "canonical mechanism",
  "mismatch_object": "pc|register|memory|csr|opcode|state|timeout|unknown",
  "dut_state": "canonical DUT state",
  "state_transition": "from_state_to_state_or_unknown",
  "trigger": "canonical trigger",
  "candidate_explanations": ["short_candidate"],
  "evidence_tags": ["short_tag"],
  "conflict_tags": ["short_conflict"],
  "confidence": 0.0,
  "rationale": "one short sentence"
}}

Evidence:
{evidence}
"""
    return system, user


def _parse_json(text: str, case_id: str) -> CompletionCaseFeature:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return CompletionCaseFeature(case_id, "invalid_json")
    try:
        data = json.loads(match.group(0))
    except Exception:
        return CompletionCaseFeature(case_id, "invalid_json")
    tags = data.get("evidence_tags") or []
    conflicts = data.get("conflict_tags") or []
    candidates = data.get("candidate_explanations") or []
    if not isinstance(tags, list):
        tags = [tags]
    if not isinstance(conflicts, list):
        conflicts = [conflicts]
    if not isinstance(candidates, list):
        candidates = [candidates]
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0)))
    return CompletionCaseFeature(
        case_id=case_id,
        status="ok",
        failure_stage=_normalize(data.get("failure_stage")),
        primary_mechanism=_normalize(data.get("primary_mechanism")),
        mismatch_object=_normalize(data.get("mismatch_object")),
        dut_state=_normalize(data.get("dut_state")),
        state_transition=_normalize(data.get("state_transition")),
        trigger=_normalize(data.get("trigger")),
        candidate_explanations=tuple(sorted({_normalize(tag) for tag in candidates if _normalize(tag)})),
        evidence_tags=tuple(sorted({_normalize(tag) for tag in tags if _normalize(tag)})),
        conflict_tags=tuple(sorted({_normalize(tag) for tag in conflicts if _normalize(tag)})),
        confidence=confidence,
        rationale=str(data.get("rationale", ""))[:240],
    )


SCHEMA_VERSION = 2  # bump when prompt/schema/parsing changes


def build_completion_case_features(
    input_csvs: Sequence[str | Path],
    cache_dir: str | Path,
    strict: bool = False,
    selected_indices: set[int] | None = None,
    cache_ignore_errors: bool = True,
    cache_error_ttl_sec: float = 3600.0,
) -> tuple[list[CompletionCaseFeature], list[dict]]:
    config = load_completion_config()
    if config is None and strict:
        raise RuntimeError(
            "LLM_MODEL_CONFIG has no valid completion/chat section; "
            "completion features cannot be generated"
        )
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    client = None
    if config is not None:
        from openai import OpenAI

        client = OpenAI(
            api_key=config["api_key"] or "dummy",
            base_url=config["base_url"],
            timeout=config["timeout"],
        )

    features: list[CompletionCaseFeature] = []
    debug: list[dict] = []
    global_index = 0
    for input_raw in input_csvs:
        input_csv = Path(input_raw).resolve()
        rows, fields = rfb.read_csv_rows(input_csv)
        for idx, row in enumerate(rows):
            case_id = _case_id(row, fields, idx)
            selected = selected_indices is None or global_index in selected_indices
            global_index += 1
            if not selected:
                feature = CompletionCaseFeature(case_id, "not_selected")
                features.append(feature)
                debug.append({
                    "input_csv": str(input_csv),
                    "case_id": case_id,
                    "status": feature.status,
                    "cached": False,
                    "failure_stage": "",
                    "primary_mechanism": "",
                    "mismatch_object": "",
                    "dut_state": "",
                    "trigger": "",
                    "state_transition": "",
                    "candidate_explanations": "",
                    "evidence_tags": "",
                    "conflict_tags": "",
                    "confidence": 0.0,
                })
                continue
            evidence = _compact_evidence(input_csv, row, fields)
            # cache key includes model, base_url, schema_version, evidence hash
            cache_identity = "|".join([
                str(config["model"] if config else "missing"),
                str(config["base_url"] if config else "missing"),
                f"v{SCHEMA_VERSION}",
            ])
            digest = hashlib.sha256(
                (cache_identity + "\n" + evidence).encode()
            ).hexdigest()
            cache_path = cache_root / f"{digest}.json"
            cached = False
            if cache_path.is_file():
                try:
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    # Check TTL for error caches
                    if raw.get("status") != "ok" and cache_ignore_errors:
                        created = raw.get("created_at", 0)
                        if created and (time.time() - created) < cache_error_ttl_sec:
                            # Error cache still fresh, reuse it
                            pass
                        else:
                            # Error cache expired or ignore_errors, delete and re-fetch
                            cache_path.unlink()
                            raw = None
                    if raw is not None:
                        # Load feature from cache
                        feature = CompletionCaseFeature(
                            case_id=case_id,
                            status=raw.get("status", "missing"),
                            failure_stage=raw.get("failure_stage", ""),
                            primary_mechanism=raw.get("primary_mechanism", ""),
                            mismatch_object=raw.get("mismatch_object", ""),
                            dut_state=raw.get("dut_state", ""),
                            state_transition=raw.get("state_transition", ""),
                            trigger=raw.get("trigger", ""),
                            candidate_explanations=tuple(raw.get("candidate_explanations", ())),
                            evidence_tags=tuple(raw.get("evidence_tags", ())),
                            conflict_tags=tuple(raw.get("conflict_tags", ())),
                            confidence=float(raw.get("confidence", 0.0)),
                            rationale=str(raw.get("rationale", "")),
                        )
                        cached = True
                except (json.JSONDecodeError, OSError, TypeError) as exc:
                    print(f"[completion] cache read error for {case_id}: {exc}", file=sys.stderr)
                    cache_path.unlink(missing_ok=True)
            if not cached:
                if client is None:
                    feature = CompletionCaseFeature(case_id, "missing_config")
                else:
                    try:
                        system, user = _prompt(evidence)
                        response = client.chat.completions.create(
                            model=config["model"],
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            temperature=0,
                            max_tokens=config["max_tokens"],
                        )
                        text = response.choices[0].message.content or ""
                        feature = _parse_json(text, case_id)
                        # Always cache, including errors (with TTL)
                        cache_data = {
                            "case_id": case_id,
                            "status": feature.status,
                            "failure_stage": feature.failure_stage,
                            "primary_mechanism": feature.primary_mechanism,
                            "mismatch_object": feature.mismatch_object,
                            "dut_state": feature.dut_state,
                            "state_transition": feature.state_transition,
                            "trigger": feature.trigger,
                            "candidate_explanations": list(feature.candidate_explanations),
                            "evidence_tags": list(feature.evidence_tags),
                            "conflict_tags": list(feature.conflict_tags),
                            "confidence": feature.confidence,
                            "rationale": feature.rationale,
                            "raw_response": text[:500],
                            "created_at": time.time(),
                            "schema_version": SCHEMA_VERSION,
                            "model": config["model"],
                        }
                        cache_path.write_text(
                            json.dumps(cache_data, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                    except Exception as exc:
                        if strict:
                            raise
                        feature = CompletionCaseFeature(
                            case_id, f"request_error:{type(exc).__name__}"
                        )
                        # Cache error too (with short TTL)
                        cache_data = {
                            "case_id": case_id,
                            "status": feature.status,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:500],
                            "created_at": time.time(),
                            "schema_version": SCHEMA_VERSION,
                            "model": config["model"] if config else "unknown",
                        }
                        try:
                            cache_path.write_text(
                                json.dumps(cache_data, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8",
                            )
                        except OSError:
                            pass
            features.append(feature)
            debug.append({
                "input_csv": str(input_csv),
                "case_id": case_id,
                "status": feature.status,
                "cached": cached,
                "failure_stage": feature.failure_stage,
                "primary_mechanism": feature.primary_mechanism,
                "mismatch_object": feature.mismatch_object,
                "dut_state": feature.dut_state,
                "trigger": feature.trigger,
                "state_transition": feature.state_transition,
                "candidate_explanations": ";".join(feature.candidate_explanations),
                "evidence_tags": ";".join(feature.evidence_tags),
                "conflict_tags": ";".join(feature.conflict_tags),
                "confidence": feature.confidence,
            })
    return features, debug


def _same(a: str, b: str) -> float:
    return float(bool(a and b and a == b))


def _conflict(a: str, b: str) -> float:
    return float(bool(a and b and a != b))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def build_completion_pair_feature_vector(
    a: CompletionCaseFeature,
    b: CompletionCaseFeature,
) -> np.ndarray:
    values: list[float] = []
    for key in CANONICAL_KEYS:
        av = str(getattr(a, key))
        bv = str(getattr(b, key))
        values.extend((_same(av, bv), _conflict(av, bv)))
    values.extend([
        _jaccard(set(a.candidate_explanations), set(b.candidate_explanations)),
        _jaccard(set(a.evidence_tags), set(b.evidence_tags)),
        float(len(set(a.evidence_tags) & set(b.evidence_tags))),
        _jaccard(set(a.conflict_tags), set(b.conflict_tags)),
        float(len(set(a.conflict_tags) ^ set(b.conflict_tags))),
        min(a.confidence, b.confidence),
        abs(a.confidence - b.confidence),
        float(a.status == "ok" and b.status == "ok"),
        float((a.status == "ok") != (b.status == "ok")),
    ])
    return np.asarray(values, dtype=np.float32)


def completion_pair_feature_dim() -> int:
    empty = CompletionCaseFeature("x", "missing")
    return len(build_completion_pair_feature_vector(empty, empty))


def build_completion_pair_feature_matrix(
    features: Sequence[CompletionCaseFeature],
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    if not pairs:
        return np.zeros((0, completion_pair_feature_dim()), dtype=np.float32)
    matrix = np.empty((len(pairs), completion_pair_feature_dim()), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        matrix[row] = build_completion_pair_feature_vector(features[i], features[j])
    return matrix
