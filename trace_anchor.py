#!/usr/bin/env python3
"""Anchor-guided trace windows for experimental trace features.

Uses sim.log/regr.log to locate a local trace window. Never reads gold/meta.
"""

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
import trace_features as tf

INDEX_RE = re.compile(r"\b(?:ibex|dut|rtl|spike|iss)\[(\d+)\]", re.IGNORECASE)
PC_BRACKET_RE = re.compile(r"\bpc\[(0x[0-9a-fA-F]+|[0-9a-fA-F]{6,16})\]", re.IGNORECASE)
UVM_TIME_RE = re.compile(r"@\s*(\d+)\s*:")
OP_PAIR_RE = rfb.PC_OP_RE
DUT_RETIRED_PC_RE = re.compile(r"DUT\s+retired\s*:\s*(0x[0-9a-fA-F]+|[0-9a-fA-F]{6,16})", re.IGNORECASE)
ISS_RETIRED_PC_RE = re.compile(r"ISS\s+retired\s*:\s*(0x[0-9a-fA-F]+|[0-9a-fA-F]{6,16})", re.IGNORECASE)
REG_MISMATCH_RE = re.compile(r"register write data mismatch to\s+(x(?:[0-2]?\d|3[01]))", re.IGNORECASE)
ANCHOR_TAG_PATTERNS = {
    "illegal_instruction": re.compile(r"illegal instruction|illegal", re.IGNORECASE),
    "debug_entry": re.compile(r"IN_DEBUG_MODE|debug", re.IGNORECASE),
    "irq_entry": re.compile(r"HANDLING_IRQ|interrupt|irq", re.IGNORECASE),
    "csr": re.compile(r"CSR|IbexCsr", re.IGNORECASE),
    "ebreak": re.compile(r"\bebreak\b", re.IGNORECASE),
    "dret": re.compile(r"\bdret\b", re.IGNORECASE),
    "mret": re.compile(r"\bmret\b", re.IGNORECASE),
    "timeout": re.compile(r"timeout", re.IGNORECASE),
}


@dataclass
class TraceAnchor:
    case_id: str
    source: str
    instr_index: int | None
    dut_pc: str
    iss_pc: str
    sim_time: int | None
    ibex_opcode: str
    spike_opcode: str
    mismatch_type: str
    primary_type: str
    reason: str
    mismatch_register: str = ""
    anchor_tags: tuple[str, ...] = ()


@dataclass
class AnchorTraceFeature(tf.TraceCaseFeature):
    anchor: TraceAnchor | None = None
    anchor_source: str = ""
    located_by: str = "tail"
    center_line: int = -1
    window_start: int = 0
    window_end: int = 0
    center_opcode: str = ""
    center_pc_region: str = ""
    pre_opcode_sequence: list[str] | None = None
    post_opcode_sequence: list[str] | None = None
    repeated_pc_count: int = 0


def _normalize_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pick(fields: Sequence[str], names: Sequence[str]) -> str | None:
    by_key = {_normalize_key(f): f for f in fields}
    for name in names:
        key = _normalize_key(name)
        if key in by_key:
            return by_key[key]
    return None


def _case_id(row: dict, idx: int, fields: Sequence[str]) -> str:
    col = _pick(fields, ("Case", "case_id", "case", "id"))
    if col and row.get(col):
        value = str(row[col])
        return value if value.startswith("case_") else value
    return f"case_{idx + 1:06d}"


def _resolve(input_csv: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (input_csv.parent / path).resolve()
    return path if path.exists() else path


def _read_lines(path: Path | None, max_bytes: int = 2_000_000) -> list[str]:
    if path is None or not path.exists():
        return []
    text, status = rfb.read_log_sample(path)
    if status != "ok":
        return []
    return text.splitlines()


def _clean_pc(value: str) -> str:
    return value.lower().removeprefix("0x") if value else ""


def _pc_region(value: str, digits: int = 5) -> str:
    value = _clean_pc(value)
    if len(value) < 3:
        return value
    return "0x" + value[: min(digits, len(value))]


def _mismatch_type(text: str) -> str:
    low = text.lower()
    if "register write data mismatch" in low:
        return "register_write_data_mismatch"
    if "pc mismatch" in low:
        return "pc_mismatch"
    if "synchronous trap" in low:
        return "sync_trap_mismatch"
    if "cosim mismatch" in low or "co-sim" in low:
        return "cosim_mismatch"
    if "mismatch" in low:
        return "generic_mismatch"
    if "timeout" in low:
        return "timeout"
    return ""


def _primary_type(primary_tokens: Sequence[str]) -> str:
    if not primary_tokens:
        return ""
    token = primary_tokens[0]
    if token.startswith("PRIMARY_UVM_FATAL"):
        return "UVM_FATAL"
    if token.startswith("PRIMARY_UVM_ERROR"):
        return "UVM_ERROR"
    if token.startswith("PRIMARY_REGR"):
        return "REGR_MISMATCH"
    if token.startswith("PRIMARY_FAILED"):
        return "FAILED"
    return token


def extract_trace_anchor(case_id: str, sim_lines: Sequence[str], regr_lines: Sequence[str]) -> TraceAnchor:
    joined_regr = "\n".join(regr_lines)
    joined_sim = "\n".join(sim_lines)
    joined = joined_regr + "\n" + joined_sim
    instr_index = None
    m = INDEX_RE.search(joined_regr)
    if m:
        instr_index = int(m.group(1))
    pcs = [_clean_pc(x) for x in PC_BRACKET_RE.findall(joined_regr)]
    dut_pc = pcs[0] if pcs else ""
    iss_pc = pcs[1] if len(pcs) > 1 else ""
    if not dut_pc:
        m_dut = DUT_RETIRED_PC_RE.search(joined)
        if m_dut:
            dut_pc = _clean_pc(m_dut.group(1))
    if not iss_pc:
        m_iss = ISS_RETIRED_PC_RE.search(joined)
        if m_iss:
            iss_pc = _clean_pc(m_iss.group(1))
    sim_time = None
    time_lines = list(sim_lines) + list(regr_lines)
    time_priority_groups = [
        [line for line in time_lines if re.search(r"uvm_(?:fatal|error)|cosim mismatch|\bmismatch\b", line, re.IGNORECASE)],
        [line for line in time_lines if re.search(r"\[failed\]|test failed|assert", line, re.IGNORECASE)],
        [line for line in time_lines if re.search(r"timeout", line, re.IGNORECASE)],
        time_lines,
    ]
    for group in time_priority_groups:
        for line in group:
            tm = UVM_TIME_RE.search(line)
            if tm:
                sim_time = int(tm.group(1))
                break
        if sim_time is not None:
            break
    ibex_opcode = ""
    spike_opcode = ""
    for side, op in OP_PAIR_RE.findall(joined):
        side_l = side.lower()
        op_l = op.lower()
        if side_l in {"ibex", "dut", "rtl"} and not ibex_opcode:
            ibex_opcode = op_l
        elif side_l in {"spike", "iss"} and not spike_opcode:
            spike_opcode = op_l
    primary_tokens = rfb.extract_primary_signature({}, {}, list(sim_lines), list(regr_lines))
    reg_match = REG_MISMATCH_RE.search(joined)
    mismatch_register = reg_match.group(1).lower() if reg_match else ""
    anchor_tags = tuple(sorted(name for name, pat in ANCHOR_TAG_PATTERNS.items() if pat.search(joined)))
    reason = ""
    for line in list(regr_lines) + list(sim_lines):
        low = line.lower()
        if "uvm_fatal" in low or "uvm_error" in low or "[failed]" in low or "mismatch" in low or "timeout" in low:
            reason = rfb.sanitize_primary_part(line)
            break
    source = "fallback_tail"
    if instr_index is not None:
        source = "regr_index"
    elif dut_pc:
        source = "regr_pc"
    elif sim_time is not None:
        source = "sim_time"
    elif ibex_opcode or spike_opcode:
        source = "opcode"
    return TraceAnchor(
        case_id=case_id,
        source=source,
        instr_index=instr_index,
        dut_pc=dut_pc,
        iss_pc=iss_pc,
        sim_time=sim_time,
        ibex_opcode=ibex_opcode,
        spike_opcode=spike_opcode,
        mismatch_type=_mismatch_type(joined),
        primary_type=_primary_type(primary_tokens),
        reason=reason,
        mismatch_register=mismatch_register,
        anchor_tags=anchor_tags,
    )


def _open_trace(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if path.suffix == ".gz" else open(path, "rt", encoding="utf-8", errors="ignore")


def _parse_trace_meta(line: str) -> tuple[int | None, int | None, str, str]:
    fields = [p.strip() for p in line.split("\t")]
    time_v = cycle_v = None
    pc = opcode = ""
    if len(fields) >= 5:
        try:
            time_v = int(fields[0])
        except ValueError:
            time_v = None
        try:
            cycle_v = int(fields[1])
        except ValueError:
            cycle_v = None
        pc = _clean_pc(fields[2])
        opcode = re.split(r"[\s,]+", fields[4].strip(), maxsplit=1)[0].lower().removesuffix(":")
    else:
        opcode, pc = tf._extract_opcode_and_pc(line)
    return time_v, cycle_v, pc, opcode


def locate_trace_anchor(trace_path: str | Path, anchor: TraceAnchor) -> tuple[int | None, str, list[str]]:
    path = Path(trace_path)
    if not path.exists():
        return None, "missing", []
    lines: list[str] = []
    best_idx = None
    best_by = "tail"
    best_cycle = (10**30, None)
    best_time = (10**30, None)
    first_opcode = None
    try:
        with _open_trace(path) as f:
            for idx, raw in enumerate(f):
                line = raw.rstrip("\n")
                lines.append(line)
                if not line or line.lower().startswith("time\tcycle"):
                    continue
                time_v, cycle_v, pc, opcode = _parse_trace_meta(line)
                if anchor.dut_pc and pc and pc == anchor.dut_pc:
                    return idx, "pc", lines
                if anchor.instr_index is not None and cycle_v is not None:
                    delta = abs(cycle_v - anchor.instr_index)
                    if delta < best_cycle[0]:
                        best_cycle = (delta, idx)
                if anchor.sim_time is not None and time_v is not None:
                    delta = abs(time_v - anchor.sim_time)
                    if delta < best_time[0]:
                        best_time = (delta, idx)
                if first_opcode is None and opcode and opcode in {anchor.ibex_opcode, anchor.spike_opcode}:
                    first_opcode = idx
    except OSError:
        return None, "read_error", []
    if best_cycle[1] is not None and best_cycle[0] <= 2000:
        return best_cycle[1], "index", lines
    if best_time[1] is not None:
        return best_time[1], "time", lines
    if first_opcode is not None:
        return first_opcode, "opcode", lines
    return None, "tail", lines


def extract_anchor_window(trace_path: str | Path, anchor: TraceAnchor, window_size: int = 64) -> tuple[list[str], dict]:
    center, located_by, lines = locate_trace_anchor(trace_path, anchor)
    if not lines:
        return [], {"located_by": located_by, "center_line": -1, "window_start": 0, "window_end": 0, "trace_lines_used": 0}
    half = max(1, int(window_size) // 2)
    if center is None:
        end = len(lines)
        start = max(0, end - int(window_size))
        center = start + (end - start) // 2
    else:
        start = max(0, center - half)
        end = min(len(lines), center + half + 1)
    return lines[start:end], {"located_by": located_by, "center_line": center, "window_start": start, "window_end": end, "trace_lines_used": end - start}


def _anchor_feature_from_base(base: tf.TraceCaseFeature, anchor: TraceAnchor, meta: dict, window_lines: Sequence[str]) -> AnchorTraceFeature:
    center_opcode = ""
    center_pc = ""
    center_line = int(meta.get("center_line", -1))
    rel = center_line - int(meta.get("window_start", 0))
    parsed_ops: list[str] = []
    parsed_pcs: list[str] = []
    for line in window_lines:
        op, pc = tf._extract_opcode_and_pc(line)
        if op:
            parsed_ops.append(op)
        if pc:
            parsed_pcs.append(pc)
    if 0 <= rel < len(window_lines):
        center_opcode, center_pc = tf._extract_opcode_and_pc(window_lines[rel])
    if not center_opcode and parsed_ops:
        center_opcode = parsed_ops[len(parsed_ops) // 2]
    if not center_pc and parsed_pcs:
        center_pc = parsed_pcs[len(parsed_pcs) // 2]
    pc_counts = Counter(parsed_pcs)
    return AnchorTraceFeature(
        **base.__dict__,
        anchor=anchor,
        anchor_source=anchor.source,
        located_by=str(meta.get("located_by", "tail")),
        center_line=center_line,
        window_start=int(meta.get("window_start", 0)),
        window_end=int(meta.get("window_end", 0)),
        center_opcode=center_opcode,
        center_pc_region=_pc_region(center_pc),
        pre_opcode_sequence=parsed_ops[: max(0, len(parsed_ops) // 2)][-16:],
        post_opcode_sequence=parsed_ops[len(parsed_ops) // 2:][:16],
        repeated_pc_count=sum(1 for v in pc_counts.values() if v > 1),
    )


def build_anchor_trace_case_features(input_csvs: Sequence[str | Path], window_size: int = 64) -> tuple[list[AnchorTraceFeature], list[dict]]:
    out: list[AnchorTraceFeature] = []
    debug: list[dict] = []
    for input_csv_raw in input_csvs:
        input_csv = Path(input_csv_raw).resolve()
        with input_csv.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = reader.fieldnames or []
        sim_col = _pick(fields, ("sim_log", "Sim Log", "sim", "sim_path"))
        regr_col = _pick(fields, ("regr_log", "Regr Log", "regr", "regr_path"))
        trace_col = tf.pick_trace_column(fields)
        for idx, row in enumerate(rows):
            cid = _case_id(row, idx, fields)
            sim_path = _resolve(input_csv, row.get(sim_col) if sim_col else None)
            regr_path = _resolve(input_csv, row.get(regr_col) if regr_col else None)
            trace_path = _resolve(input_csv, row.get(trace_col) if trace_col else None)
            sim_lines = _read_lines(sim_path)
            regr_lines = _read_lines(regr_path)
            anchor = extract_trace_anchor(cid, sim_lines, regr_lines)
            if trace_path is None or not trace_path.exists():
                base = tf._empty_feature(cid, str(trace_path or ""), "missing_file")
                feat = _anchor_feature_from_base(base, anchor, {"located_by": "missing", "center_line": -1}, [])
            else:
                window, meta = extract_anchor_window(trace_path, anchor, window_size=window_size)
                base = tf.parse_trace_lines(cid, str(trace_path), window) if window else tf._empty_feature(cid, str(trace_path), "empty_window")
                feat = _anchor_feature_from_base(base, anchor, meta, window)
            out.append(feat)
            debug.append({
                "input_csv": str(input_csv),
                "case_id": cid,
                "anchor_source": anchor.source,
                "instr_index": anchor.instr_index if anchor.instr_index is not None else "",
                "dut_pc": anchor.dut_pc,
                "iss_pc": anchor.iss_pc,
                "sim_time": anchor.sim_time if anchor.sim_time is not None else "",
                "ibex_opcode": anchor.ibex_opcode,
                "spike_opcode": anchor.spike_opcode,
                "mismatch_type": anchor.mismatch_type,
                "primary_type": anchor.primary_type,
                "reason": anchor.reason,
                "mismatch_register": anchor.mismatch_register,
                "anchor_tags": ";".join(anchor.anchor_tags),
                "located_by": feat.located_by,
                "center_line": feat.center_line,
                "window_start": feat.window_start,
                "window_end": feat.window_end,
                "trace_lines_used": feat.tail_lines_used,
                "center_opcode": feat.center_opcode,
                "center_pc_region": feat.center_pc_region,
            })
    return out, debug


def _jaccard(a: set, b: set) -> float:
    return tf._jaccard(a, b)


def build_anchor_trace_pair_feature_vector(a: AnchorTraceFeature, b: AnchorTraceFeature) -> np.ndarray:
    base = tf.build_trace_pair_feature_vector(a, b)
    pre_a = a.pre_opcode_sequence or []
    pre_b = b.pre_opcode_sequence or []
    post_a = a.post_opcode_sequence or []
    post_b = b.post_opcode_sequence or []
    extra = np.asarray([
        1.0 if a.center_opcode and b.center_opcode and a.center_opcode == b.center_opcode else 0.0,
        1.0 if a.center_pc_region and b.center_pc_region and a.center_pc_region == b.center_pc_region else 0.0,
        _jaccard(set(pre_a), set(pre_b)),
        _jaccard(set(post_a), set(post_b)),
        tf._lcs_ratio((pre_a + post_a)[-32:], (pre_b + post_b)[-32:]),
        float(a.anchor_source == b.anchor_source),
        float(a.located_by == b.located_by),
        abs(float(a.repeated_pc_count) - float(b.repeated_pc_count)),
        1.0 if a.anchor and b.anchor and a.anchor.mismatch_register and a.anchor.mismatch_register == b.anchor.mismatch_register else 0.0,
        _jaccard(set(a.anchor.anchor_tags if a.anchor else ()), set(b.anchor.anchor_tags if b.anchor else ())),
    ], dtype=np.float32)
    return np.concatenate([base, extra]).astype(np.float32, copy=False)


def anchor_trace_pair_feature_dim() -> int:
    dummy = _anchor_feature_from_base(tf._empty_feature("a"), TraceAnchor("a", "fallback_tail", None, "", "", None, "", "", "", "", ""), {"located_by": "tail", "center_line": -1}, [])
    return len(build_anchor_trace_pair_feature_vector(dummy, dummy))


def build_anchor_trace_pair_feature_matrix(features: Sequence[AnchorTraceFeature], pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros((0, anchor_trace_pair_feature_dim()), dtype=np.float32)
    mat = np.empty((len(pairs), anchor_trace_pair_feature_dim()), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        mat[row] = build_anchor_trace_pair_feature_vector(features[i], features[j])
    return mat
