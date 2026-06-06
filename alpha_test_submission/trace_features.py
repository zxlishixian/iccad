#!/usr/bin/env python3
"""Lightweight trace.log structure features for experimental refinement."""

from __future__ import annotations

import csv
import gzip
import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import regr_fail_bucketing as rfb

TRACE_COLUMN_CANDIDATES = (
    "trace_log", "trace log", "trace", "trace_path", "tracefile", "trace_file",
    "rtl_trace", "trace core", "trace_core",
)
REG_RE = re.compile(
    r"\b(?:x(?:[0-2]?\d|3[01])|zero|ra|sp|gp|tp|t[0-6]|s(?:[0-9]|1[01])|fp|a[0-7])\b",
    re.IGNORECASE,
)
HEX_RE = re.compile(r"\b(?:0x)?([0-9a-fA-F]{6,16})\b")
EXCEPTION_RE = re.compile(r"exception|trap|interrupt|illegal|fault|timeout|fatal|error", re.IGNORECASE)
LOAD_OPS = {"lb", "lh", "lw", "lbu", "lhu", "ld", "c.lw", "c.ld", "c.lwsp", "c.ldsp"}
STORE_OPS = {"sb", "sh", "sw", "sd", "c.sw", "c.sd", "c.swsp", "c.sdsp"}
BRANCH_PREFIXES = ("b", "c.b", "j", "jal", "jalr", "c.j", "c.jal", "ret")
CSR_PREFIXES = ("csr",)
SYSTEM_OPS = {"mret", "dret", "wfi", "ecall", "ebreak", "fence", "sfence.vma"}


@dataclass
class TraceCaseFeature:
    case_id: str
    trace_path: str
    file_status: str
    tail_lines_used: int
    opcodes: set[str]
    opcode_counts: Counter[str]
    regs: set[str]
    reg_counts: Counter[str]
    pc_regions: set[str]
    pc_prefixes: set[str]
    csr_ops: Counter[str]
    branch_count: int
    load_count: int
    store_count: int
    exception_markers: set[str]
    tail_opcode_sequence: list[str]
    tail_pc_sequence: list[str]
    tail_reg_sequence: list[str]
    length_log: float
    parse_warnings: list[str]

    @property
    def missing(self) -> bool:
        return self.file_status != "ok"


def _normalize_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def pick_trace_column(fields: Sequence[str]) -> str | None:
    by_key = {_normalize_key(field): field for field in fields}
    for candidate in TRACE_COLUMN_CANDIDATES:
        key = _normalize_key(candidate)
        if key in by_key:
            return by_key[key]
    for field in fields:
        if "trace" in _normalize_key(field):
            return field
    return None


def _case_id_from_row(row: dict, idx: int, fields: Sequence[str]) -> str:
    wanted = {_normalize_key(name) for name in ("case_id", "case", "id")}
    for field in fields:
        if _normalize_key(field) in wanted and row.get(field):
            return str(row[field])
    return f"case_{idx + 1:06d}"


def _resolve_trace_path(input_csv: Path, value: str | None) -> tuple[Path | None, str]:
    if not value:
        return None, "missing_trace_value"
    path = Path(str(value))
    if not path.is_absolute():
        path = (input_csv.parent / path).resolve()
    if not path.exists():
        return path, "missing_file"
    if not path.is_file():
        return path, "not_file"
    return path, "ok"


def _tail_lines(path: Path, tail_lines: int) -> list[str]:
    dq: deque[str] = deque(maxlen=max(1, int(tail_lines)))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)


def _pc_region(pc: str, digits: int = 5) -> str:
    pc = pc.lower().removeprefix("0x")
    if len(pc) < 3:
        return pc
    return "0x" + pc[: min(digits, len(pc))]


def _pc_prefix(pc: str, digits: int = 4) -> str:
    pc = pc.lower().removeprefix("0x")
    if len(pc) < 3:
        return pc
    return "0x" + pc[: min(digits, len(pc))]


def _extract_opcode_and_pc(line: str) -> tuple[str, str]:
    fields = [part.strip() for part in line.split("\t")]
    pc = ""
    opcode = ""
    if len(fields) >= 5 and fields[2] and fields[4]:
        pc = fields[2].lower().removeprefix("0x")
        opcode = re.split(r"[\s,]+", fields[4].strip(), maxsplit=1)[0].lower()
    else:
        hexes = HEX_RE.findall(line)
        if len(hexes) >= 3:
            pc = hexes[2].lower()
        elif hexes:
            pc = hexes[0].lower()
        match = re.search(r"\b([a-z][a-z0-9_.]*)\b", line.lower())
        if match and match.group(1) not in {"time", "cycle", "pc", "insn", "decoded"}:
            opcode = match.group(1)
    opcode = opcode.strip().lower().removesuffix(":")
    return opcode, pc


def _opcode_class_counts(opcode: str) -> tuple[int, int, int, int]:
    if not opcode:
        return 0, 0, 0, 0
    load = int(opcode in LOAD_OPS or (opcode.startswith("l") and opcode != "lui"))
    store = int(opcode in STORE_OPS or (opcode.startswith("s") and opcode not in {"sll", "slt", "sltu", "srl", "sra", "sub"}))
    branch = int(opcode.startswith(BRANCH_PREFIXES) or opcode == "ret")
    csr = int(opcode.startswith(CSR_PREFIXES) or opcode in SYSTEM_OPS)
    return load, store, branch, csr


def _empty_feature(case_id: str, trace_path: str = "", status: str = "missing") -> TraceCaseFeature:
    return TraceCaseFeature(
        case_id=case_id,
        trace_path=trace_path,
        file_status=status,
        tail_lines_used=0,
        opcodes=set(),
        opcode_counts=Counter(),
        regs=set(),
        reg_counts=Counter(),
        pc_regions=set(),
        pc_prefixes=set(),
        csr_ops=Counter(),
        branch_count=0,
        load_count=0,
        store_count=0,
        exception_markers=set(),
        tail_opcode_sequence=[],
        tail_pc_sequence=[],
        tail_reg_sequence=[],
        length_log=0.0,
        parse_warnings=[] if status == "ok" else [status],
    )


def parse_trace_lines(case_id: str, trace_path: str, lines: Sequence[str]) -> TraceCaseFeature:
    opcode_counts: Counter[str] = Counter()
    reg_counts: Counter[str] = Counter()
    pc_regions: set[str] = set()
    pc_prefixes: set[str] = set()
    csr_ops: Counter[str] = Counter()
    exception_markers: set[str] = set()
    tail_opcodes: list[str] = []
    tail_pcs: list[str] = []
    tail_regs: list[str] = []
    load_count = store_count = branch_count = 0
    for line in lines:
        if not line or line.lower().startswith("time\tcycle"):
            continue
        opcode, pc = _extract_opcode_and_pc(line)
        if opcode:
            opcode_counts[opcode] += 1
            tail_opcodes.append(opcode)
            load, store, branch, csr = _opcode_class_counts(opcode)
            load_count += load
            store_count += store
            branch_count += branch
            if csr:
                csr_ops[opcode] += 1
        if pc:
            pc_regions.add(_pc_region(pc))
            pc_prefixes.add(_pc_prefix(pc))
            tail_pcs.append(pc.lower().removeprefix("0x"))
        regs = [reg.lower() for reg in REG_RE.findall(line)]
        if regs:
            reg_counts.update(regs)
            tail_regs.extend(regs[:4])
        if EXCEPTION_RE.search(line):
            low = line.lower()
            for marker in ("exception", "trap", "interrupt", "illegal", "fault", "timeout", "fatal", "error"):
                if marker in low:
                    exception_markers.add(marker)
    return TraceCaseFeature(
        case_id=case_id,
        trace_path=trace_path,
        file_status="ok",
        tail_lines_used=len(lines),
        opcodes=set(opcode_counts),
        opcode_counts=opcode_counts,
        regs=set(reg_counts),
        reg_counts=reg_counts,
        pc_regions=pc_regions,
        pc_prefixes=pc_prefixes,
        csr_ops=csr_ops,
        branch_count=branch_count,
        load_count=load_count,
        store_count=store_count,
        exception_markers=exception_markers,
        tail_opcode_sequence=tail_opcodes[-32:],
        tail_pc_sequence=tail_pcs[-32:],
        tail_reg_sequence=tail_regs[-64:],
        length_log=math.log1p(len(lines)),
        parse_warnings=[] if opcode_counts else ["no_opcodes"],
    )


def build_trace_case_features(
    input_csv: str | Path,
    tail_lines: int = 500,
    window_mode: str = "tail",
) -> list[TraceCaseFeature]:
    if window_mode != "tail":
        raise ValueError(f"unsupported trace window_mode: {window_mode}")
    input_csv = Path(input_csv).resolve()
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    trace_col = pick_trace_column(fields)
    out: list[TraceCaseFeature] = []
    for idx, row in enumerate(rows):
        case_id = _case_id_from_row(row, idx, fields)
        if not trace_col:
            out.append(_empty_feature(case_id, status="missing_trace_column"))
            continue
        path, status = _resolve_trace_path(input_csv, row.get(trace_col))
        if status != "ok" or path is None:
            out.append(_empty_feature(case_id, str(path or ""), status=status))
            continue
        try:
            out.append(parse_trace_lines(case_id, str(path), _tail_lines(path, tail_lines)))
        except OSError as exc:
            out.append(_empty_feature(case_id, str(path), status=f"read_error:{type(exc).__name__}"))
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _counter_cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(float(a.get(k, 0)) * float(b.get(k, 0)) for k in keys)
    na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    return dot / max(na * nb, 1e-12)


def _ngrams(seq: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if len(seq) < n:
        return set()
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}


def _lcs_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1] / max(min(len(a), len(b)), 1)


def _last_n_overlap(a: Sequence[str], b: Sequence[str], n: int) -> float:
    return _jaccard(set(a[-n:]), set(b[-n:]))


def _ratio(count: int, feat: TraceCaseFeature) -> float:
    denom = max(1, sum(feat.opcode_counts.values()))
    return float(count) / float(denom)


def build_trace_pair_feature_vector(a: TraceCaseFeature, b: TraceCaseFeature) -> np.ndarray:
    both_missing = float(a.missing and b.missing)
    one_missing = float(a.missing != b.missing)
    last_opcode_same = 0.0
    if a.tail_opcode_sequence and b.tail_opcode_sequence:
        last_opcode_same = float(a.tail_opcode_sequence[-1] == b.tail_opcode_sequence[-1])
    exception_marker_same = float(a.exception_markers == b.exception_markers) if (a.exception_markers or b.exception_markers) else 1.0
    values = [
        _jaccard(a.opcodes, b.opcodes),
        _counter_cosine(a.opcode_counts, b.opcode_counts),
        _jaccard(a.regs, b.regs),
        _counter_cosine(a.reg_counts, b.reg_counts),
        _jaccard(a.pc_regions, b.pc_regions),
        _jaccard(a.pc_prefixes, b.pc_prefixes),
        _lcs_ratio(a.tail_opcode_sequence[-24:], b.tail_opcode_sequence[-24:]),
        _jaccard(_ngrams(a.tail_opcode_sequence, 2), _ngrams(b.tail_opcode_sequence, 2)),
        _jaccard(_ngrams(a.tail_opcode_sequence, 3), _ngrams(b.tail_opcode_sequence, 3)),
        last_opcode_same,
        _last_n_overlap(a.tail_opcode_sequence, b.tail_opcode_sequence, 5),
        exception_marker_same,
        abs(_ratio(a.branch_count, a) - _ratio(b.branch_count, b)),
        abs(_ratio(a.load_count, a) - _ratio(b.load_count, b)),
        abs(_ratio(a.store_count, a) - _ratio(b.store_count, b)),
        abs(_ratio(sum(a.csr_ops.values()), a) - _ratio(sum(b.csr_ops.values()), b)),
        abs(a.length_log - b.length_log),
        both_missing,
        one_missing,
    ]
    return np.asarray(values, dtype=np.float32)


def trace_pair_feature_dim() -> int:
    return len(build_trace_pair_feature_vector(_empty_feature("a"), _empty_feature("b")))


def build_trace_pair_feature_matrix(
    features: Sequence[TraceCaseFeature],
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    if not pairs:
        return np.zeros((0, trace_pair_feature_dim()), dtype=np.float32)
    mat = np.empty((len(pairs), trace_pair_feature_dim()), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        mat[row] = build_trace_pair_feature_vector(features[i], features[j])
    return mat
