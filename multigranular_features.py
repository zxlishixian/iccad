#!/usr/bin/env python3
"""Experimental multi-granular sim/regr evidence features.

This module reads only paths listed in input.csv. It does not read gold, meta,
or trace logs. Local failure windows may be embedded through the configured
embedding endpoint, while all structured evidence is extracted locally.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

import regr_fail_bucketing as rfb


EVENT_PATTERNS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("fatal", re.compile(r"\buvm_fatal\b|\bfatal\b", re.I), 5),
    ("error", re.compile(r"\buvm_error\b|\berror\b", re.I), 4),
    ("pc_mismatch", re.compile(r"\bpc mismatch\b", re.I), 4),
    ("register_mismatch", re.compile(r"register write data mismatch", re.I), 4),
    ("memory_mismatch", re.compile(r"\b(?:load|store|memory)\b.{0,60}\bmismatch\b", re.I), 4),
    ("trap_mismatch", re.compile(r"synchronous trap|mcause.{0,40}mismatch", re.I), 4),
    ("generic_mismatch", re.compile(r"\bmismatch\b", re.I), 3),
    ("timeout", re.compile(r"\btimeout\b|timed out", re.I), 3),
    ("assertion", re.compile(r"\bassert(?:ion)?\b", re.I), 3),
    ("failed", re.compile(r"\btest failed\b|\[failed\]|\bfailed\b", re.I), 2),
    ("irq", re.compile(r"\b(?:irq|interrupt|handling_irq)\b", re.I), 2),
    ("debug", re.compile(r"\bdebug\b|in_debug_mode", re.I), 2),
)
TIME_RE = re.compile(r"@\s*(\d+)")
PC_RE = re.compile(r"\b(?:0x)?([0-9a-fA-F]{6,16})\b")
REG_RE = re.compile(r"\b(x(?:[0-2]?\d|3[01])|zero|ra|sp|gp|tp|a[0-7]|t[0-6]|s(?:[0-9]|1[01]))\b", re.I)
CSR_RE = re.compile(r"\b(mcause|mstatus|mepc|mtvec|dcsr|dpc|mie|mip|csr[a-z0-9_]*)\b", re.I)
OPCODE_RE = re.compile(
    r"\b(lui|auipc|jalr?|beq|bne|blt|bge|bltu|bgeu|lb|lh|lw|lbu|lhu|"
    r"sb|sh|sw|addi|sltiu?|xori|ori|andi|slli|srli|srai|add|sub|sll|"
    r"sltu?|xor|srl|sra|or|and|fence|ecall|ebreak|mret|dret|wfi|"
    r"csrrw|csrrs|csrrc|csrrwi|csrrsi|csrrci|mul[a-z]*|div[a-z]*|rem[a-z]*)\b",
    re.I,
)
STATE_PATTERNS = {
    "irq": re.compile(r"\b(?:irq|interrupt|handling_irq)\b", re.I),
    "debug": re.compile(r"\bdebug\b|in_debug_mode", re.I),
    "exception": re.compile(r"\bexception\b|\btrap\b", re.I),
    "retire": re.compile(r"\bretir(?:e|ed|ing)\b", re.I),
    "csr": re.compile(r"\bcsr\b|mcause|mstatus|mepc|dcsr", re.I),
    "memory": re.compile(r"\b(?:load|store|memory)\b", re.I),
}


@dataclass(frozen=True)
class FailureEvent:
    source: str
    event_type: str
    position: int
    relative_position: float
    time: int | None
    severity: int
    object_type: str
    context: str


@dataclass
class CaseEvidence:
    case_id: str
    sim_status: str
    regr_status: str
    events: list[FailureEvent] = field(default_factory=list)
    pc_regions: tuple[str, ...] = ()
    opcodes: tuple[str, ...] = ()
    registers: tuple[str, ...] = ()
    csrs: tuple[str, ...] = ()
    states: tuple[str, ...] = ()
    sim_local_doc: str = ""
    regr_local_doc: str = ""
    sim_local_vec: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    regr_local_vec: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sim_local_reduced: np.ndarray | None = None
    regr_local_reduced: np.ndarray | None = None

    @property
    def effective_sim_vec(self) -> np.ndarray:
        return self.sim_local_reduced if self.sim_local_reduced is not None else self.sim_local_vec

    @property
    def effective_regr_vec(self) -> np.ndarray:
        return self.regr_local_reduced if self.regr_local_reduced is not None else self.regr_local_vec


def _normalize_line(line: str) -> str:
    line = re.sub(r"\b0x[0-9a-fA-F]+\b", "<HEX>", line)
    line = re.sub(r"\b[0-9a-fA-F]{8,16}\b", "<ADDR>", line)
    line = re.sub(r"\b\d+\b", "<NUM>", line)
    return re.sub(r"\s+", " ", line).strip()[:300]


def _pc_region(value: str) -> str:
    value = value.lower().removeprefix("0x")
    return "0x" + value[:5] if len(value) >= 5 else value


def _object_type(line: str) -> str:
    low = line.lower()
    if "pc mismatch" in low or "retired" in low:
        return "pc"
    if "register write data mismatch" in low or REG_RE.search(line):
        return "register"
    if CSR_RE.search(line):
        return "csr"
    if re.search(r"\b(?:load|store|memory)\b", line, re.I):
        return "memory"
    if OPCODE_RE.search(line):
        return "opcode"
    if "timeout" in low:
        return "timeout"
    return "state"


def _extract_events(source: str, lines: Sequence[str], context_radius: int) -> list[FailureEvent]:
    events: list[FailureEvent] = []
    total = max(1, len(lines) - 1)
    for pos, line in enumerate(lines):
        matches = [(name, severity) for name, pattern, severity in EVENT_PATTERNS if pattern.search(line)]
        if any(name.endswith("_mismatch") and name != "generic_mismatch" for name, _ in matches):
            matches = [(name, severity) for name, severity in matches if name != "generic_mismatch"]
        if not matches:
            continue
        start = max(0, pos - context_radius)
        stop = min(len(lines), pos + context_radius + 1)
        context = " | ".join(_normalize_line(x) for x in lines[start:stop] if x.strip())
        time_match = TIME_RE.search(line)
        for name, severity in matches:
            events.append(FailureEvent(
                source=source,
                event_type=name,
                position=pos,
                relative_position=float(pos) / total,
                time=int(time_match.group(1)) if time_match else None,
                severity=severity,
                object_type=_object_type(line),
                context=context,
            ))
    return events[:32]


def _local_doc(source: str, events: Sequence[FailureEvent], max_events: int = 8) -> str:
    selected = sorted(events, key=lambda e: (-e.severity, e.position))[:max_events]
    lines = [f"source: {source}", "task: local regression failure evidence"]
    for event in selected:
        lines.append(
            f"event={event.event_type} object={event.object_type} "
            f"position={event.relative_position:.3f} severity={event.severity}"
        )
        lines.append(f"context: {event.context}")
    return "\n".join(lines)


def _case_id(row: dict, fields: Sequence[str], idx: int) -> str:
    normalized = {"".join(ch for ch in f.lower() if ch.isalnum()): f for f in fields}
    for key in ("case", "caseid", "id"):
        field = normalized.get(key)
        if field and str(row.get(field, "")).strip():
            return str(row[field]).strip()
    return str(idx + 1)


def build_case_evidence(
    input_csvs: Sequence[str | Path],
    context_radius: int = 2,
) -> tuple[list[CaseEvidence], list[dict]]:
    evidence: list[CaseEvidence] = []
    debug: list[dict] = []
    for input_raw in input_csvs:
        input_csv = Path(input_raw).resolve()
        rows, fields = rfb.read_csv_rows(input_csv)
        sim_col = rfb.pick_column(fields, "sim")
        regr_col = rfb.pick_column(fields, "regr")
        for idx, row in enumerate(rows):
            case_id = _case_id(row, fields, idx)
            sim_path = rfb.resolve_log_path(input_csv, row.get(sim_col) if sim_col else None)
            regr_path = rfb.resolve_log_path(input_csv, row.get(regr_col) if regr_col else None)
            sim_text, sim_status = rfb.read_log_sample(sim_path)
            regr_text, regr_status = rfb.read_log_sample(regr_path)
            sim_lines = sim_text.splitlines() if sim_status == "ok" else []
            regr_lines = regr_text.splitlines() if regr_status == "ok" else []
            sim_events = _extract_events("sim", sim_lines, context_radius)
            regr_events = _extract_events("regr", regr_lines, context_radius)
            events = sorted(sim_events + regr_events, key=lambda e: (e.source, e.position))
            joined = sim_text + "\n" + regr_text
            pcs = tuple(dict.fromkeys(_pc_region(x) for x in PC_RE.findall(joined)))[:16]
            opcodes = tuple(dict.fromkeys(x.lower() for x in OPCODE_RE.findall(joined)))[:24]
            registers = tuple(dict.fromkeys(x.lower() for x in REG_RE.findall(joined)))[:16]
            csrs = tuple(dict.fromkeys(x.lower() for x in CSR_RE.findall(joined)))[:16]
            states = tuple(name for name, pattern in STATE_PATTERNS.items() if pattern.search(joined))
            item = CaseEvidence(
                case_id=case_id,
                sim_status=sim_status,
                regr_status=regr_status,
                events=events,
                pc_regions=pcs,
                opcodes=opcodes,
                registers=registers,
                csrs=csrs,
                states=states,
                sim_local_doc=_local_doc("sim", sim_events),
                regr_local_doc=_local_doc("regr", regr_events),
            )
            evidence.append(item)
            debug.append({
                "input_csv": str(input_csv),
                "case_id": case_id,
                "sim_status": sim_status,
                "regr_status": regr_status,
                "event_count": len(events),
                "event_sequence": ";".join(e.event_type for e in events),
                "pc_regions": ";".join(pcs),
                "opcodes": ";".join(opcodes),
                "registers": ";".join(registers),
                "csrs": ";".join(csrs),
                "states": ";".join(states),
            })
    return evidence, debug


def embed_local_documents(
    evidence: Sequence[CaseEvidence],
    cache_dir: str | Path,
    batch_size: int = 32,
    timeout_sec: float = 120.0,
) -> str:
    docs = [item.sim_local_doc for item in evidence] + [item.regr_local_doc for item in evidence]
    args = argparse.Namespace(
        llm_cache_dir=Path(cache_dir),
        llm_batch_size=batch_size,
        llm_timeout_sec=timeout_sec,
    )
    vectors, model_name = rfb.fetch_llm_embeddings(docs, args)
    matrix = np.asarray(vectors, dtype=np.float32)
    count = len(evidence)
    for idx, item in enumerate(evidence):
        item.sim_local_vec = matrix[idx]
        item.regr_local_vec = matrix[count + idx]
    return model_name


def fit_local_reducers(
    evidence: Sequence[CaseEvidence],
    dim: int,
    random_state: int,
) -> tuple[object | None, object | None]:
    from sklearn.decomposition import PCA

    def fit(values: list[np.ndarray]) -> object | None:
        if not values or not values[0].size:
            return None
        matrix = np.vstack(values)
        actual = min(int(dim), matrix.shape[0], matrix.shape[1])
        if actual <= 0 or actual >= matrix.shape[1]:
            return None
        return PCA(n_components=actual, random_state=random_state).fit(matrix)

    return (
        fit([item.sim_local_vec for item in evidence]),
        fit([item.regr_local_vec for item in evidence]),
    )


def apply_local_reducers(
    evidence: Sequence[CaseEvidence],
    reducers: tuple[object | None, object | None],
) -> None:
    sim_reducer, regr_reducer = reducers
    if evidence and sim_reducer is not None:
        transformed = sim_reducer.transform(np.vstack([x.sim_local_vec for x in evidence]))
        for item, vec in zip(evidence, transformed):
            item.sim_local_reduced = np.asarray(vec, dtype=np.float32)
    if evidence and regr_reducer is not None:
        transformed = regr_reducer.transform(np.vstack([x.regr_local_vec for x in evidence]))
        for item, vec in zip(evidence, transformed):
            item.regr_local_reduced = np.asarray(vec, dtype=np.float32)


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _counter_cosine(a: Sequence[str], b: Sequence[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 1.0
    dot = sum(ca[k] * cb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / max(na * nb, 1e-12)


@lru_cache(maxsize=65536)
def _lcs_ratio_cached(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for idx, right in enumerate(b, 1):
            current.append(previous[idx - 1] + 1 if left == right else max(previous[idx], current[-1]))
        previous = current
    return previous[-1] / max(len(a), len(b))


def _lcs_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    # The last 16 semantic events retain local ordering while keeping the
    # all-pairs LODO feature build bounded on 640-case episodes.
    return _lcs_ratio_cached(tuple(a[-16:]), tuple(b[-16:]))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if not a.size or not b.size:
        return 0.0
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def _relation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not a.size or not b.size or a.shape != b.shape:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([np.abs(a - b), a * b]).astype(np.float32)


def build_multigranular_pair_feature_vector(
    a: CaseEvidence,
    b: CaseEvidence,
    include_event_order: bool = True,
    include_local_embeddings: bool = True,
) -> np.ndarray:
    ae = [e.event_type for e in a.events]
    be = [e.event_type for e in b.events]
    ao = [e.object_type for e in a.events]
    bo = [e.object_type for e in b.events]
    at = [e.time for e in a.events if e.time is not None]
    bt = [e.time for e in b.events if e.time is not None]
    apos = {e.event_type: e.relative_position for e in a.events}
    bpos = {e.event_type: e.relative_position for e in b.events}
    common = set(apos) & set(bpos)
    position_diff = float(np.mean([abs(apos[k] - bpos[k]) for k in common])) if common else 1.0
    time_diff = (
        abs(math.log1p(float(np.median(at))) - math.log1p(float(np.median(bt))))
        if at and bt else 0.0
    )
    severity_a = max((e.severity for e in a.events), default=0)
    severity_b = max((e.severity for e in b.events), default=0)
    structural = np.asarray([
        _jaccard(ae, be),
        _counter_cosine(ae, be),
        _lcs_ratio(ae, be) if include_event_order else 0.0,
        _jaccard(ao, bo),
        position_diff if include_event_order else 0.0,
        time_diff,
        abs(float(severity_a - severity_b)) / 5.0,
        _jaccard(a.pc_regions, b.pc_regions),
        _jaccard(a.opcodes, b.opcodes),
        _counter_cosine(a.opcodes, b.opcodes),
        _jaccard(a.registers, b.registers),
        _jaccard(a.csrs, b.csrs),
        _jaccard(a.states, b.states),
        float(a.sim_status == b.sim_status),
        float(a.regr_status == b.regr_status),
        float(bool(a.pc_regions) != bool(b.pc_regions)),
        float(bool(a.registers) != bool(b.registers)),
        float(bool(a.csrs) != bool(b.csrs)),
        math.log1p(len(set(ae) & set(be))),
        math.log1p(len(set(a.opcodes) & set(b.opcodes))),
    ], dtype=np.float32)
    if not include_local_embeddings:
        return structural
    sim_a, sim_b = a.effective_sim_vec, b.effective_sim_vec
    regr_a, regr_b = a.effective_regr_vec, b.effective_regr_vec
    scalars = np.asarray([
        _cosine(sim_a, sim_b),
        _cosine(regr_a, regr_b),
        _cosine(sim_a, regr_b),
        _cosine(regr_a, sim_b),
    ], dtype=np.float32)
    return np.concatenate([
        structural,
        _relation(sim_a, sim_b),
        _relation(regr_a, regr_b),
        scalars,
    ]).astype(np.float32)


def build_multigranular_pair_feature_matrix(
    evidence: Sequence[CaseEvidence],
    pairs: Sequence[tuple[int, int]],
    include_event_order: bool = True,
    include_local_embeddings: bool = True,
) -> np.ndarray:
    if not pairs:
        if not evidence:
            return np.zeros((0, 20), dtype=np.float32)
        sample = build_multigranular_pair_feature_vector(
            evidence[0], evidence[0], include_event_order, include_local_embeddings
        )
        return np.zeros((0, len(sample)), dtype=np.float32)
    sample = build_multigranular_pair_feature_vector(
        evidence[pairs[0][0]], evidence[pairs[0][1]],
        include_event_order, include_local_embeddings,
    )
    matrix = np.empty((len(pairs), len(sample)), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        matrix[row] = build_multigranular_pair_feature_vector(
            evidence[i], evidence[j], include_event_order, include_local_embeddings
        )
    return matrix


def pair_conflict_score(a: CaseEvidence, b: CaseEvidence) -> float:
    conflicts = 0.0
    evidence = 0.0
    for left, right in (
        (a.pc_regions, b.pc_regions),
        (a.opcodes, b.opcodes),
        (a.registers, b.registers),
        (a.csrs, b.csrs),
        (a.states, b.states),
    ):
        if left and right:
            evidence += 1.0
            conflicts += 1.0 - _jaccard(left, right)
    ae = [e.event_type for e in a.events]
    be = [e.event_type for e in b.events]
    if ae and be:
        evidence += 1.0
        conflicts += 1.0 - _jaccard(ae, be)
    return float(conflicts / evidence) if evidence else 0.0


def build_conflict_matrix(evidence: Sequence[CaseEvidence]) -> np.ndarray:
    n = len(evidence)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = pair_conflict_score(evidence[i], evidence[j])
            matrix[i, j] = matrix[j, i] = value
    return matrix
