"""Bounded sim/regr evidence for experimental large-scale selection.

This module never reads labels, metadata, or trace logs.  Plain logs use a
bounded head/tail sample; gzip logs use a bounded decompressed head so runtime
does not depend on scanning the complete compressed stream.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import pairwise_llm_features as plf
import regr_fail_bucketing as rfb


@dataclass
class BoundedEvidenceFeature:
    case_id: str
    vector: np.ndarray
    info: dict
    tokens: list[str]
    primary_tokens: set[str]
    sim_tokens: set[str]
    regr_tokens: set[str]
    sim_status: str
    regr_status: str


def read_bounded_sample(path: Path | None, max_bytes: int = 64 * 1024) -> tuple[str, str]:
    if path is None:
        return "", "missing_path"
    try:
        size = path.stat().st_size
    except OSError:
        return "", "missing_file"
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
                text = handle.read(max_bytes + 1)
            if len(text) > max_bytes:
                return text[:max_bytes], "ok_head_truncated"
            return text, "ok"
        with path.open("rb") as handle:
            if size <= max_bytes:
                data = handle.read(max_bytes)
                status = "ok"
            else:
                half = max_bytes // 2
                head = handle.read(half)
                handle.seek(max(0, size - half))
                data = head + b"\n... <BOUNDED_HEAD_TAIL> ...\n" + handle.read(half)
                status = "ok_head_tail_truncated"
        return data.decode("utf-8", "ignore"), status
    except OSError:
        return "", "read_error"


def _add_hash(vector: np.ndarray, token: str, weight: float) -> None:
    digest = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
    index = int.from_bytes(digest, "little") % len(vector)
    vector[index] += np.float32(weight)


def _evidence_vector(
    sim_lines: list[str],
    regr_lines: list[str],
    primary_tokens: list[str],
    info: dict,
    dim: int,
) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in primary_tokens:
        _add_hash(vector, f"primary:{token}", 4.0)
    for field in (
        "primary_type", "mismatch_type", "op_pair", "fatal_file",
        "register_name", "ibex_opcode", "spike_opcode", "pc_region",
    ):
        value = str(info.get(field, "") or "").strip().lower()
        if value:
            _add_hash(vector, f"{field}:{value}", 3.0)
    for prefix, lines in (("sim", sim_lines), ("regr", regr_lines)):
        previous = ""
        for line in lines[:96]:
            normalized = rfb.normalize_log_line_semantic(line, preserve_basenames=True)
            words = re.findall(r"[a-z_][a-z0-9_.:+-]{1,48}", normalized.lower())[:24]
            for word in words:
                _add_hash(vector, f"{prefix}:tok:{word}", 1.0)
                if previous:
                    _add_hash(vector, f"{prefix}:bi:{previous}|{word}", 0.5)
                previous = word
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= np.float32(norm)
    return vector


def _selected_tokens(lines: list[str]) -> set[str]:
    output: set[str] = set()
    for line in lines[:96]:
        normalized = rfb.normalize_log_line_semantic(line, preserve_basenames=True)
        output.update(re.findall(r"[a-z_][a-z0-9_.:+-]{1,48}", normalized.lower())[:24])
    return output


def build_bounded_evidence(
    input_csv: Path,
    max_bytes: int = 64 * 1024,
    dim: int = 256,
) -> list[BoundedEvidenceFeature]:
    input_csv = Path(input_csv).resolve()
    rows, fields = rfb.read_csv_rows(input_csv)
    sim_col = rfb.pick_column(fields, "sim")
    regr_col = rfb.pick_column(fields, "regr")
    used_cols = [column for column in (sim_col, regr_col) if column]
    output: list[BoundedEvidenceFeature] = []
    for index, row in enumerate(rows):
        case_id = rfb.case_id_from_row(row, index, used_cols)
        texts: dict[str, str] = {}
        statuses: dict[str, str] = {}
        selected: dict[str, list[str]] = {}
        paths: dict[str, Path | None] = {}
        for prefix, column in (("sim", sim_col), ("regr", regr_col)):
            path = rfb.resolve_log_path(input_csv, row.get(column) if column else None)
            text, status = read_bounded_sample(path, max_bytes=max_bytes)
            paths[prefix] = path
            texts[prefix] = text
            statuses[prefix] = status
            selected[prefix] = rfb.select_lines(text)
        primary_tokens = rfb.extract_primary_signature(
            {"path": str(paths["sim"] or ""), "status": statuses["sim"]},
            {"path": str(paths["regr"] or ""), "status": statuses["regr"]},
            selected["sim"], selected["regr"],
        )
        info = plf._extract_rich_case_info(
            selected["sim"], selected["regr"], primary_tokens
        )
        sim_tokens = _selected_tokens(selected["sim"])
        regr_tokens = _selected_tokens(selected["regr"])
        tokens = sorted(sim_tokens | regr_tokens | set(primary_tokens))
        output.append(BoundedEvidenceFeature(
            case_id=case_id,
            vector=_evidence_vector(
                selected["sim"], selected["regr"], primary_tokens, info, dim
            ),
            info=info,
            tokens=tokens,
            primary_tokens=set(primary_tokens),
            sim_tokens=sim_tokens,
            regr_tokens=regr_tokens,
            sim_status=statuses["sim"],
            regr_status=statuses["regr"],
        ))
    return output
