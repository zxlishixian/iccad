#!/usr/bin/env python3
"""Official-style root-cause features for experimental training.

These helpers are intentionally separate from the official predictor. They may
be used by training/evaluation scripts that read gold labels, but they do not
change ``regr_fail_bucketing.py`` or its default no-trace path.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
import trace_anchor as ta


TEST_RE = re.compile(r"\b([A-Za-z0-9_]*ibex[A-Za-z0-9_]*_test|[A-Za-z0-9_]+_test)\b")

ROOT_TAG_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("irq_entry", re.compile(r"\b(?:irq|interrupt|HANDLING_IRQ)\b", re.IGNORECASE)),
    ("debug_entry", re.compile(r"\b(?:debug|IN_DEBUG_MODE)\b", re.IGNORECASE)),
    ("dret_return", re.compile(r"\b(?:dret|NO_DRET)\b", re.IGNORECASE)),
    ("mret_return", re.compile(r"\bmret\b", re.IGNORECASE)),
    ("csr", re.compile(r"\b(?:csr|mcause|mstatus|mepc|dcsr|IbexCsr)\b", re.IGNORECASE)),
    ("mcause_exception", re.compile(r"mcause|exception_code|wrong_exception", re.IGNORECASE)),
    ("memory_fault", re.compile(r"memory fault|load_store|IBEXDATAOFFSETKNOWN|fault", re.IGNORECASE)),
    ("illegal_instruction", re.compile(r"illegal instruction|illegal", re.IGNORECASE)),
    ("core_status_timeout", re.compile(r"core_status|timeout", re.IGNORECASE)),
    ("cosim_pc_divergence", re.compile(r"pc mismatch|DUT retired|ISS retired", re.IGNORECASE)),
    ("cosim_reg_divergence", re.compile(r"register write data mismatch", re.IGNORECASE)),
    ("scoreboard", re.compile(r"scoreboard|cosim", re.IGNORECASE)),
    ("ebreak", re.compile(r"\bebreak\b", re.IGNORECASE)),
]


@dataclass
class OfficialCaseRecord:
    dataset: str
    case_id: str
    gold: str
    info: dict
    primary_signature: str
    primary_type: str
    mismatch_type: str
    test_name: str
    root_tags: frozenset[str]
    anchor: ta.TraceAnchor


def normalize_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def pick_col(fields: Sequence[str], names: Sequence[str], fallback: int = 0) -> str:
    by_key = {normalize_key(f): f for f in fields}
    for name in names:
        key = normalize_key(name)
        if key in by_key:
            return by_key[key]
    return fields[min(fallback, len(fields) - 1)]


def gold_path(dataset: Path) -> Path:
    for name in ("gold.csv", "golden.csv", "answer.csv", "answers.csv", "labels.csv"):
        path = dataset / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no gold/golden csv found under {dataset}")


def read_gold_map(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    case_col = pick_col(fields, ("Case", "case", "case_id", "id"), 0)
    bug_col = pick_col(fields, ("Bug", "bug", "bug_id", "gold", "label", "bucket"), 1 if len(fields) > 1 else 0)
    return {str(row[case_col]).strip(): str(row[bug_col]).strip() for row in rows}


def read_cases(input_csv: Path) -> list[str]:
    rows, fields = rfb.read_csv_rows(input_csv)
    case_col = pick_col(fields, ("Case", "case", "case_id", "id"), 0)
    out: list[str] = []
    for idx, row in enumerate(rows):
        value = str(row.get(case_col, "")).strip()
        out.append(value if value else str(idx + 1))
    return out


def _root_tags(text: str, info: dict, anchor: ta.TraceAnchor) -> frozenset[str]:
    tags = {name for name, pat in ROOT_TAG_PATTERNS if pat.search(text)}
    tags.update(anchor.anchor_tags)
    mismatch = str(info.get("mismatch_type", ""))
    if mismatch:
        tags.add(f"mismatch:{mismatch}")
    primary_type = str(info.get("primary_type", ""))
    if primary_type:
        tags.add(f"primary:{primary_type}")
    fatal_file = str(info.get("fatal_file", ""))
    if fatal_file and fatal_file != "unknown_source":
        tags.add(f"fatal:{fatal_file}")
    register = str(info.get("register_name", ""))
    if register:
        tags.add(f"reg:{register}")
    return frozenset(tags)


def build_case_records(dataset: str, input_csv: Path, gold_csv: Path | None = None) -> list[OfficialCaseRecord]:
    input_csv = Path(input_csv)
    gold = read_gold_map(gold_csv) if gold_csv is not None else {}
    rows, fields = rfb.read_csv_rows(input_csv)
    case_col = pick_col(fields, ("Case", "case", "case_id", "id"), 0)
    sim_col = rfb.pick_column(fields, "sim")
    regr_col = rfb.pick_column(fields, "regr")
    out: list[OfficialCaseRecord] = []
    for idx, row in enumerate(rows):
        case_id = str(row.get(case_col, "")).strip() or str(idx + 1)
        sim_path = rfb.resolve_log_path(input_csv, row.get(sim_col) if sim_col else None)
        regr_path = rfb.resolve_log_path(input_csv, row.get(regr_col) if regr_col else None)
        sim_text, _ = rfb.read_log_sample(sim_path)
        regr_text, _ = rfb.read_log_sample(regr_path)
        sim_lines = rfb.select_lines(sim_text)
        regr_lines = rfb.select_lines(regr_text)
        primary_tokens = rfb.extract_primary_signature({}, {}, sim_lines, regr_lines)
        info = plf._extract_rich_case_info(sim_lines, regr_lines, primary_tokens)
        anchor = ta.extract_trace_anchor(case_id, sim_text.splitlines(), regr_text.splitlines())
        tests = TEST_RE.findall(sim_text + "\n" + regr_text)
        test_name = str(info.get("uvm_testname") or (tests[0] if tests else ""))
        joined = sim_text + "\n" + regr_text
        out.append(OfficialCaseRecord(
            dataset=dataset,
            case_id=case_id,
            gold=gold.get(case_id, ""),
            info=info,
            primary_signature=str(info.get("primary_signature", "")),
            primary_type=str(info.get("primary_type", "")),
            mismatch_type=str(info.get("mismatch_type", "")),
            test_name=test_name,
            root_tags=_root_tags(joined, info, anchor),
            anchor=anchor,
        ))
    return out


def all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(set(a) | set(b))
    return len(set(a) & set(b)) / union if union else 0.0


def _same_nonempty(a: str, b: str) -> float:
    return float(bool(a and b and a == b))


def _conflict(a: str, b: str) -> float:
    return float(bool(a and b and a != b))


def logit(p: float) -> float:
    p = min(1.0 - 1e-5, max(1e-5, float(p)))
    return math.log(p / (1.0 - p))


def graph_context(prob: np.ndarray) -> np.ndarray:
    """Return per-case graph context from a no-trace probability matrix."""
    n = prob.shape[0]
    out = np.zeros((n, 8), dtype=np.float32)
    for i in range(n):
        vals = [float(prob[i, j]) for j in range(n) if j != i]
        vals.sort(reverse=True)
        if not vals:
            continue
        top1 = vals[0]
        top2 = np.mean(vals[:2]) if len(vals) >= 2 else top1
        top3 = np.mean(vals[:3]) if len(vals) >= 3 else np.mean(vals)
        top5 = np.mean(vals[:5]) if len(vals) >= 5 else np.mean(vals)
        arr = np.asarray(vals, dtype=np.float32)
        out[i] = np.asarray([
            float(arr.mean()),
            float(arr.std()),
            float(np.quantile(arr, 0.25)),
            float(np.quantile(arr, 0.50)),
            float(np.quantile(arr, 0.75)),
            float(top1),
            float(top3),
            float(top5),
        ], dtype=np.float32)
    return out


def pair_feature_names(include_graph: bool = True, include_anchor: bool = False) -> list[str]:
    names = [
        "p_base",
        "logit_base",
        "abs_p_minus_half",
        "same_primary_signature",
        "same_primary_type",
        "primary_type_conflict",
        "same_mismatch_type",
        "mismatch_type_conflict",
        "same_test_name",
        "test_name_conflict",
        "same_fatal_file",
        "fatal_file_conflict",
        "same_register",
        "root_tag_jaccard",
        "root_tag_overlap_count",
        "same_anchor_source",
        "anchor_tag_jaccard",
        "same_dut_pc",
        "same_mismatch_register",
    ]
    if include_graph:
        base = ["mean", "std", "q25", "q50", "q75", "top1", "top3", "top5"]
        for prefix in ("a", "b", "absdiff"):
            names.extend(f"graph_{prefix}_{name}" for name in base)
    if include_anchor:
        # trace_anchor.build_anchor_trace_pair_feature_vector currently returns
        # 29 dimensions. Name them generically to keep this helper stable.
        names.extend(f"anchor_pair_{idx:02d}" for idx in range(29))
    return names


def build_pair_feature_matrix(
    records: Sequence[OfficialCaseRecord],
    pairs: Sequence[tuple[int, int]],
    prob_base: np.ndarray,
    include_graph: bool = True,
    include_anchor: bool = False,
    anchor_pair_matrix: np.ndarray | None = None,
) -> np.ndarray:
    graph = graph_context(prob_base) if include_graph else np.zeros((len(records), 0), dtype=np.float32)
    rows: list[np.ndarray] = []
    for row_idx, (i, j) in enumerate(pairs):
        a = records[i]
        b = records[j]
        ai = a.info
        bi = b.info
        p = float(prob_base[i, j])
        root_j = jaccard(a.root_tags, b.root_tags)
        anchor_j = jaccard(set(a.anchor.anchor_tags), set(b.anchor.anchor_tags))
        values = [
            p,
            logit(p),
            abs(p - 0.5),
            _same_nonempty(a.primary_signature, b.primary_signature),
            _same_nonempty(a.primary_type, b.primary_type),
            _conflict(a.primary_type, b.primary_type),
            _same_nonempty(a.mismatch_type, b.mismatch_type),
            _conflict(a.mismatch_type, b.mismatch_type),
            _same_nonempty(a.test_name, b.test_name),
            _conflict(a.test_name, b.test_name),
            _same_nonempty(str(ai.get("fatal_file", "")), str(bi.get("fatal_file", ""))),
            _conflict(str(ai.get("fatal_file", "")), str(bi.get("fatal_file", ""))),
            _same_nonempty(str(ai.get("register_name", "")), str(bi.get("register_name", ""))),
            root_j,
            float(len(set(a.root_tags) & set(b.root_tags))),
            _same_nonempty(a.anchor.source, b.anchor.source),
            anchor_j,
            _same_nonempty(a.anchor.dut_pc, b.anchor.dut_pc),
            _same_nonempty(a.anchor.mismatch_register, b.anchor.mismatch_register),
        ]
        blocks = [np.asarray(values, dtype=np.float32)]
        if include_graph:
            blocks.extend([
                graph[i],
                graph[j],
                np.abs(graph[i] - graph[j]),
            ])
        if include_anchor:
            if anchor_pair_matrix is None:
                raise ValueError("include_anchor=True requires anchor_pair_matrix")
            blocks.append(anchor_pair_matrix[row_idx].astype(np.float32, copy=False))
        rows.append(np.concatenate(blocks).astype(np.float32, copy=False))
    if not rows:
        return np.zeros((0, len(pair_feature_names(include_graph, include_anchor))), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def pair_labels(records: Sequence[OfficialCaseRecord], pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.asarray([1.0 if records[i].gold and records[i].gold == records[j].gold else 0.0 for i, j in pairs], dtype=np.float32)

