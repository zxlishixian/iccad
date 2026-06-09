#!/usr/bin/env python3
"""Experimental pairwise same-bug features with LLM embedding augmentation.

Extends pairwise_features.py with:
- LLM embedding vectors attached to each case
- Pairwise features combining deterministic + LLM cosine similarities
- Three model backends: logistic regression, gradient boosting, small MLP
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import pairwise_features as pf
import regr_fail_bucketing as rfb


@dataclass
class LLMCaseFeature:
    case_id: str
    det_vec: np.ndarray
    llm_vec: np.ndarray
    llm_vec_reduced: np.ndarray | None
    llm_summary_vec: np.ndarray
    llm_summary_vec_reduced: np.ndarray | None
    trace_vec: np.ndarray          # 128d raw trace embedding
    trace_vec_reduced: np.ndarray | None
    tokens: list[str]
    token_set: set[str]
    primary_tokens: set[str]
    sim_tokens: set[str]
    regr_tokens: set[str]
    info: dict
    trace_structured: Any = None   # TraceStructuredSummary | None
    completion_feature: Any = None  # CompletionCaseFeature | None

    @property
    def has_llm(self) -> bool:
        return self.llm_vec.size > 0

    @property
    def effective_llm_vec(self) -> np.ndarray:
        if self.llm_vec_reduced is not None:
            return self.llm_vec_reduced
        return self.llm_vec

    @property
    def has_llm_summary(self) -> bool:
        return self.llm_summary_vec.size > 0

    @property
    def effective_llm_summary_vec(self) -> np.ndarray:
        if self.llm_summary_vec_reduced is not None:
            return self.llm_summary_vec_reduced
        return self.llm_summary_vec

    @property
    def has_trace(self) -> bool:
        return self.trace_vec.size > 0 and np.any(self.trace_vec != 0)

    @property
    def effective_trace_vec(self) -> np.ndarray:
        if self.trace_vec_reduced is not None:
            return self.trace_vec_reduced
        return self.trace_vec


def _make_llm_args(
    llm_mode: str = "embedding",
    llm_fusion: str = "concat",
    llm_weight: float = 4.0,
    llm_alpha: float = 0.75,
    llm_doc_style: str = "features",
    llm_doc_max_features: int = 80,
    llm_cache_dir: Path = Path("/tmp/regr_fail_llm_cache"),
    llm_batch_size: int = 64,
    llm_timeout_sec: float = 20.0,
    svd_dim: int = 64,
    llm_dual: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        llm_mode=llm_mode,
        llm_fusion=llm_fusion,
        llm_weight=llm_weight,
        llm_alpha=llm_alpha,
        llm_doc_style=llm_doc_style,
        llm_doc_max_features=llm_doc_max_features,
        llm_cache_dir=Path(llm_cache_dir),
        llm_batch_size=llm_batch_size,
        llm_timeout_sec=llm_timeout_sec,
        svd_dim=svd_dim,
        parser="drain",
        llm_dual=llm_dual,
    )


def fetch_llm_embeddings_for_counters(
    feature_counters: Sequence[Counter],
    llm_args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    docs = rfb.build_llm_case_documents(
        feature_counters,
        max_features=llm_args.llm_doc_max_features,
        doc_style=llm_args.llm_doc_style,
    )
    embeddings, model_name = rfb.fetch_llm_embeddings(docs, llm_args)
    llm_mat = np.asarray(embeddings, dtype="float32")
    if llm_mat.ndim != 2 or llm_mat.shape[0] != len(feature_counters):
        raise RuntimeError(f"unexpected embedding matrix shape: {llm_mat.shape}")
    from sklearn.preprocessing import Normalizer
    llm_mat = Normalizer(copy=False).fit_transform(llm_mat)
    return llm_mat, model_name



UVM_COMPONENT_RE = re.compile(r"\[([^\]]*(?:uvm_test_top|env|agent|scoreboard|driver|monitor)[^\]]*)\]", re.IGNORECASE)
PC_VALUE_RE = re.compile(r"\bpc\[(0x[0-9a-fA-F]+|[0-9a-fA-F]{6,})\]", re.IGNORECASE)
REG_TARGET_RE = re.compile(r"register write data mismatch to\s+(x(?:[0-2]?\d|3[01]))", re.IGNORECASE)


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _pc_region(value: str) -> str:
    value = value.lower().strip()
    if not value:
        return ""
    if value.startswith("0x"):
        hex_digits = value[2:]
    else:
        hex_digits = value
    if len(hex_digits) < 3:
        return value
    return "0x" + hex_digits[: max(3, min(5, len(hex_digits)))]


def _extract_rich_case_info(sim_lines: Sequence[str], regr_lines: Sequence[str], primary_tokens: Sequence[str]) -> dict:
    joined_sim = "\n".join(sim_lines)
    joined_regr = "\n".join(regr_lines)
    joined_all = joined_sim + "\n" + joined_regr
    lower_all = joined_all.lower()

    error_source_file = ""
    fatal_file = ""
    uvm_component = ""
    failed_reason = ""
    for line in list(sim_lines) + list(regr_lines):
        lower = line.lower()
        if not error_source_file and re.search(r"uvm_(?:fatal|error|warning)|mismatch|failed|timeout", lower):
            error_source_file = rfb.extract_sv_basename(line)
        if not fatal_file and re.search(r"uvm_(?:fatal|error)", lower):
            fatal_file = rfb.extract_sv_basename(line)
            failed_reason = rfb.sanitize_primary_part(rfb.message_after_uvm_context(line))
        if not uvm_component:
            match = UVM_COMPONENT_RE.search(line)
            if match:
                uvm_component = rfb.sanitize_feature_value(match.group(1), max_len=120)
    if not failed_reason:
        for line in list(regr_lines) + list(sim_lines):
            if "failed" in line.lower():
                failed_reason = rfb.sanitize_primary_part(line)
                break

    ibex_opcode = ""
    spike_opcode = ""
    op_pair = ""
    for line in list(regr_lines) + list(sim_lines):
        for side, op in rfb.PC_OP_RE.findall(line):
            side_l = side.lower()
            op_l = op.lower()
            if side_l in {"ibex", "dut", "rtl"} and not ibex_opcode:
                ibex_opcode = op_l
            elif side_l in {"spike", "iss"} and not spike_opcode:
                spike_opcode = op_l
    if ibex_opcode and spike_opcode:
        op_pair = rfb.sanitize_primary_token("PRIMARY_REGR_OPPAIR", ibex_opcode, spike_opcode)

    register_name = ""
    reg_match = REG_TARGET_RE.search(joined_all)
    if reg_match:
        register_name = reg_match.group(1).lower()
    else:
        reg_match = rfb.REG_RE.search(joined_all)
        if reg_match:
            register_name = reg_match.group(0).lower()

    pc_region = ""
    pc_match = PC_VALUE_RE.search(joined_all)
    if pc_match:
        pc_region = _pc_region(pc_match.group(1))

    mismatch_type = ""
    mismatch_patterns = [
        ("pc_mismatch", r"\bpc mismatch\b"),
        ("register_write_data_mismatch", r"register write data mismatch"),
        ("sync_trap_mismatch", r"synchronous trap"),
        ("memory_mismatch", r"memory mismatch|\bstore\b.*mismatch|\bload\b.*mismatch"),
        ("instruction_mismatch", r"instruction mismatch|retired"),
        ("cosim_mismatch", r"cosim mismatch|co-sim"),
    ]
    for name, pattern in mismatch_patterns:
        if re.search(pattern, lower_all):
            mismatch_type = name
            break
    if not mismatch_type and "mismatch" in lower_all:
        mismatch_type = "generic_mismatch"

    primary_signature = primary_tokens[0] if primary_tokens else "PRIMARY_UNKNOWN_FAILURE"
    return {
        "has_uvm_fatal": "UVM_FATAL" in joined_sim.upper(),
        "has_uvm_error": "UVM_ERROR" in joined_sim.upper(),
        "has_regr_mismatch": "mismatch" in joined_regr.lower(),
        "primary_type": _primary_type_from_tokens(primary_tokens),
        "primary_signature": primary_signature,
        "mismatch_type": mismatch_type,
        "op_pair": _first_nonempty(op_pair, _op_pair_from_tokens(primary_tokens)),
        "fatal_file": fatal_file,
        "error_source_file": error_source_file,
        "uvm_component": uvm_component,
        "failed_reason": failed_reason,
        "uvm_testname": _extract_test_name_from_lines(sim_lines),
        "failed_test_name": _extract_failed_test_name_from_lines(regr_lines),
        "ibex_opcode": ibex_opcode,
        "spike_opcode": spike_opcode,
        "register_name": register_name,
        "pc_region": pc_region,
    }


def _extract_test_name_from_lines(lines: Sequence[str]) -> str:
    patterns = [
        r"\btest(?:name)?\s*[=:]\s*([A-Za-z0-9_.:+-]+)",
        r"\brunning\s+test\s+([A-Za-z0-9_.:+-]+)",
        r"\bTEST\s+([A-Za-z0-9_.:+-]+)",
    ]
    joined = "\n".join(lines)
    for pattern in patterns:
        match = re.search(pattern, joined, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _extract_failed_test_name_from_lines(lines: Sequence[str]) -> str:
    for line in lines:
        if "failed" not in line.lower():
            continue
        match = re.search(r"([A-Za-z0-9_.:+-]+)\s*(?:\[[^\]]+\])?\s*(?:FAILED|failed)", line)
        if match:
            return match.group(1)
    return ""


def _primary_type_from_tokens(primary_tokens: Sequence[str]) -> str:
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


def _op_pair_from_tokens(primary_tokens: Sequence[str]) -> str:
    for token in primary_tokens:
        if token.startswith("PRIMARY_REGR_OPPAIR_"):
            return token
    return ""


def collect_rich_infos(input_csvs: Sequence[Path], parser: str) -> list[dict]:
    del parser
    rich_infos: list[dict] = []
    for input_csv in input_csvs:
        rows, fields = rfb.read_csv_rows(input_csv)
        sim_col = rfb.pick_column(fields, "sim")
        regr_col = rfb.pick_column(fields, "regr")
        for row in rows:
            selected_by_prefix: dict[str, list[str]] = {"sim": [], "regr": []}
            info_by_prefix: dict[str, dict] = {"sim": {}, "regr": {}}
            for prefix, col in (("sim", sim_col), ("regr", regr_col)):
                path = rfb.resolve_log_path(input_csv, row.get(col) if col else None)
                text, status = rfb.read_log_sample(path)
                selected_by_prefix[prefix] = rfb.select_lines(text)
                info_by_prefix[prefix] = {"path": str(path) if path else "", "status": status}
            primary_tokens = rfb.extract_primary_signature(
                info_by_prefix["sim"],
                info_by_prefix["regr"],
                selected_by_prefix["sim"],
                selected_by_prefix["regr"],
            )
            rich_infos.append(
                _extract_rich_case_info(
                    selected_by_prefix["sim"],
                    selected_by_prefix["regr"],
                    primary_tokens,
                )
            )
    return rich_infos

def build_llm_case_features_for_inputs(
    input_csvs: Sequence[str | Path],
    parser: str = "drain",
    svd_dim: int = 64,
    llm_args: argparse.Namespace | None = None,
) -> tuple[list[LLMCaseFeature], pf.VectorizerBundle]:
    """Build deterministic features for one or more inputs and attach LLM embeddings.

    Multiple training splits are vectorized together so deterministic SVD vectors
    live in one basis for pair sampling. Inference can still build one input at a
    time because the learned pair model consumes relations, not absolute case ids.
    """
    resolved_inputs = [Path(p).resolve() for p in input_csvs]
    det_features, bundle = pf.build_case_features_for_inputs(
        resolved_inputs,
        parser=parser,
        svd_dim=svd_dim,
    )
    rich_infos = collect_rich_infos(resolved_inputs, parser)

    llm_vecs: list[np.ndarray] = []
    llm_summary_vecs: list[np.ndarray] = []
    model_name = ""
    if llm_args is not None and rfb.load_llm_embedding_config() is not None:
        # Re-extract feature counters to build LLM documents
        base_features = []
        normalized_lines = []
        for input_csv in resolved_inputs:
            _case_ids, base, lines, _infos = pf._collect_case_inputs(input_csv, parser)
            base_features.extend(base)
            normalized_lines.extend(lines)
        parser_args = pf._make_parser_args(parser)
        feature_counters, _template_count = rfb.build_feature_counters(
            parser_args,
            base_features,
            normalized_lines,
            token_weights=None,
            token_weight_mode="none",
        )
        try:
            if getattr(llm_args, "llm_dual", False):
                features_args = argparse.Namespace(**vars(llm_args))
                features_args.llm_doc_style = "features"
                summary_args = argparse.Namespace(**vars(llm_args))
                summary_args.llm_doc_style = "summary"
                llm_mat, model_name = fetch_llm_embeddings_for_counters(feature_counters, features_args)
                summary_mat, summary_model_name = fetch_llm_embeddings_for_counters(feature_counters, summary_args)
                for i in range(len(det_features)):
                    llm_vecs.append(llm_mat[i].astype(np.float32, copy=False))
                    llm_summary_vecs.append(summary_mat[i].astype(np.float32, copy=False))
                print(
                    f"[llm_features] model={model_name} embedding_dim={llm_mat.shape[1]} "
                    f"docs={len(feature_counters)} doc_style=features",
                    file=sys.stderr,
                )
                print(
                    f"[llm_summary] model={summary_model_name} embedding_dim={summary_mat.shape[1]} "
                    f"docs={len(feature_counters)} doc_style=summary",
                    file=sys.stderr,
                )
            else:
                llm_mat, model_name = fetch_llm_embeddings_for_counters(feature_counters, llm_args)
                for i in range(len(det_features)):
                    llm_vecs.append(llm_mat[i].astype(np.float32, copy=False))
                    llm_summary_vecs.append(np.zeros(0, dtype=np.float32))
                print(
                    f"[llm_features] model={model_name} embedding_dim={llm_mat.shape[1]} "
                    f"docs={len(feature_counters)} doc_style={llm_args.llm_doc_style}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"[llm_features] LLM fetch failed ({exc}); using zero llm_vec", file=sys.stderr)
            llm_vecs = [np.zeros(0, dtype=np.float32) for _ in det_features]
            llm_summary_vecs = [np.zeros(0, dtype=np.float32) for _ in det_features]
    else:
        print("[llm_features] LLM disabled; using zero llm_vec", file=sys.stderr)
        llm_vecs = [np.zeros(0, dtype=np.float32) for _ in det_features]
        llm_summary_vecs = [np.zeros(0, dtype=np.float32) for _ in det_features]

    result: list[LLMCaseFeature] = []
    for det_feat, llm_vec, llm_summary_vec, rich_info in zip(det_features, llm_vecs, llm_summary_vecs, rich_infos):
        info = dict(det_feat.info)
        for key, value in rich_info.items():
            if value or key not in info:
                info[key] = value
        result.append(
            LLMCaseFeature(
                case_id=det_feat.case_id,
                det_vec=det_feat.dense_vec.astype(np.float32, copy=False),
                llm_vec=llm_vec,
                llm_vec_reduced=None,
                llm_summary_vec=llm_summary_vec,
                llm_summary_vec_reduced=None,
                trace_vec=np.zeros(0, dtype=np.float32),
                trace_vec_reduced=None,
                tokens=list(det_feat.tokens),
                token_set=det_feat.token_set,
                primary_tokens=det_feat.primary_tokens,
                sim_tokens=det_feat.sim_tokens,
                regr_tokens=det_feat.regr_tokens,
                info=info,
                trace_structured=None,
                completion_feature=None,
            )
        )
    return result, bundle


def build_llm_case_features(
    input_csv: str | Path,
    parser: str = "drain",
    svd_dim: int = 64,
    llm_args: argparse.Namespace | None = None,
) -> tuple[list[LLMCaseFeature], pf.VectorizerBundle]:
    """Build deterministic features (via pairwise_features) and attach LLM embeddings."""
    return build_llm_case_features_for_inputs(
        [input_csv],
        parser=parser,
        svd_dim=svd_dim,
        llm_args=llm_args,
    )


def _pad_or_trim(matrix: np.ndarray, dim: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[1] < dim:
        pad = np.zeros((matrix.shape[0], dim - matrix.shape[1]), dtype=np.float32)
        return np.hstack([matrix, pad])
    if matrix.shape[1] > dim:
        return matrix[:, :dim]
    return matrix


def _fit_reducer_for_matrix(
    matrix: np.ndarray,
    reduce_dim: int,
    random_state: int = 0,
) -> tuple[Any, np.ndarray]:
    if reduce_dim <= 0 or matrix.size == 0:
        return None, np.zeros((matrix.shape[0], 0), dtype=np.float32)
    n_components = min(int(reduce_dim), matrix.shape[0] - 1, matrix.shape[1] - 1)
    if n_components < 2:
        transformed = matrix[:, : min(matrix.shape[1], max(1, int(reduce_dim)))]
        reducer = None
    else:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer

        reducer = Pipeline(
            [
                ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
                ("norm", Normalizer(copy=False)),
            ]
        )
        transformed = reducer.fit_transform(matrix)
    return reducer, _pad_or_trim(transformed, int(reduce_dim))


def _apply_reducer_to_matrix(matrix: np.ndarray, reducer: Any, reduce_dim: int) -> np.ndarray:
    if reduce_dim <= 0 or matrix.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float32)
    if reducer is None:
        transformed = matrix[:, : min(matrix.shape[1], int(reduce_dim))]
    else:
        transformed = reducer.transform(matrix)
    return _pad_or_trim(transformed, int(reduce_dim))


def fit_llm_reducer(
    features: list[LLMCaseFeature],
    reduce_dim: int,
    random_state: int = 0,
) -> Any:
    """Fit a train-only dimensionality reducer for feature-style LLM vectors."""
    if reduce_dim <= 0 or not features or not features[0].has_llm:
        for feat in features:
            feat.llm_vec_reduced = np.zeros(0, dtype=np.float32)
        return None
    llm = np.vstack([feat.llm_vec for feat in features]).astype(np.float32)
    reducer, transformed = _fit_reducer_for_matrix(llm, reduce_dim, random_state)
    for feat, vec in zip(features, transformed):
        feat.llm_vec_reduced = vec.astype(np.float32, copy=False)
    return reducer


def fit_llm_summary_reducer(
    features: list[LLMCaseFeature],
    reduce_dim: int,
    random_state: int = 0,
) -> Any:
    """Fit a train-only reducer for summary-style LLM vectors."""
    if reduce_dim <= 0 or not features or not features[0].has_llm_summary:
        for feat in features:
            feat.llm_summary_vec_reduced = np.zeros(0, dtype=np.float32)
        return None
    llm = np.vstack([feat.llm_summary_vec for feat in features]).astype(np.float32)
    reducer, transformed = _fit_reducer_for_matrix(llm, reduce_dim, random_state)
    for feat, vec in zip(features, transformed):
        feat.llm_summary_vec_reduced = vec.astype(np.float32, copy=False)
    return reducer


def apply_llm_reducer(
    features: list[LLMCaseFeature],
    reducer: Any,
    reduce_dim: int,
) -> None:
    if reduce_dim <= 0 or not features or not features[0].has_llm:
        for feat in features:
            feat.llm_vec_reduced = np.zeros(0, dtype=np.float32)
        return
    llm = np.vstack([feat.llm_vec for feat in features]).astype(np.float32)
    transformed = _apply_reducer_to_matrix(llm, reducer, int(reduce_dim))
    for feat, vec in zip(features, transformed):
        feat.llm_vec_reduced = vec.astype(np.float32, copy=False)


def apply_llm_summary_reducer(
    features: list[LLMCaseFeature],
    reducer: Any,
    reduce_dim: int,
) -> None:
    if reduce_dim <= 0 or not features or not features[0].has_llm_summary:
        for feat in features:
            feat.llm_summary_vec_reduced = np.zeros(0, dtype=np.float32)
        return
    llm = np.vstack([feat.llm_summary_vec for feat in features]).astype(np.float32)
    transformed = _apply_reducer_to_matrix(llm, reducer, int(reduce_dim))
    for feat, vec in zip(features, transformed):
        feat.llm_summary_vec_reduced = vec.astype(np.float32, copy=False)


def normalize_trace_vectors(features: list[LLMCaseFeature]) -> int:
    """Ensure all features have trace_vec of the same dimension.

    Features without a trace (size 0) get a zero vector of embed_dim.
    Returns the embed_dim, or 0 if no features have traces.
    """
    embed_dim = 0
    for f in features:
        if f.trace_vec.size > 0:
            embed_dim = f.trace_vec.size
            break
    if embed_dim == 0:
        return 0
    for f in features:
        if f.trace_vec.size != embed_dim:
            f.trace_vec = np.zeros(embed_dim, dtype=np.float32)
    return embed_dim


def fit_trace_reducer(
    features: list[LLMCaseFeature],
    reduce_dim: int,
    random_state: int = 0,
) -> Any:
    """Fit a train-only SVD reducer for trace embeddings."""
    if reduce_dim <= 0 or not features:
        for feat in features:
            feat.trace_vec_reduced = np.zeros(0, dtype=np.float32)
        return None
    # Determine the embedding dimension from features that have trace
    embed_dim = 0
    for f in features:
        if f.trace_vec.size > 0:
            embed_dim = f.trace_vec.size
            break
    if embed_dim == 0:
        for feat in features:
            feat.trace_vec_reduced = np.zeros(0, dtype=np.float32)
        return None
    # Normalize: zero-vectors for missing traces get the right shape
    trace_mat = np.vstack([
        feat.trace_vec if feat.trace_vec.size == embed_dim
        else np.zeros(embed_dim, dtype=np.float32)
        for feat in features
    ]).astype(np.float32)
    reducer, transformed = _fit_reducer_for_matrix(trace_mat, reduce_dim, random_state)
    for feat, vec in zip(features, transformed):
        feat.trace_vec_reduced = vec.astype(np.float32, copy=False)
    return reducer


def apply_trace_reducer(
    features: list[LLMCaseFeature],
    reducer: Any,
    reduce_dim: int,
) -> None:
    """Apply a pre-fit trace reducer to features."""
    if reduce_dim <= 0 or not features:
        for feat in features:
            feat.trace_vec_reduced = np.zeros(0, dtype=np.float32)
        return
    embed_dim = 0
    for f in features:
        if f.trace_vec.size > 0:
            embed_dim = f.trace_vec.size
            break
    if embed_dim == 0:
        for feat in features:
            feat.trace_vec_reduced = np.zeros(0, dtype=np.float32)
        return
    trace_mat = np.vstack([
        feat.trace_vec if feat.trace_vec.size == embed_dim
        else np.zeros(embed_dim, dtype=np.float32)
        for feat in features
    ]).astype(np.float32)
    transformed = _apply_reducer_to_matrix(trace_mat, reducer, int(reduce_dim))
    for feat, vec in zip(features, transformed):
        feat.trace_vec_reduced = vec.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Pairwise feature vector (fixed-dimension, as specified in the task)
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return dot / max(na * nb, 1e-12)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _same_nonempty(a: str, b: str) -> float:
    return 1.0 if a and b and a == b else 0.0


def _same_bool(di: dict, dj: dict, key: str) -> float:
    return 1.0 if bool(di.get(key)) == bool(dj.get(key)) else 0.0


def llm_pair_feature_dim() -> int:
    """Return fixed dimension of pairwise feature vectors (computed from spec)."""
    return 21


def build_llm_pair_feature_vector(
    a: LLMCaseFeature,
    b: LLMCaseFeature,
    include_det: bool = True,
    include_llm: bool = True,
) -> np.ndarray:
    det_cosine = _cosine(a.det_vec, b.det_vec) if include_det else 0.0
    det_euclidean = _euclidean(a.det_vec, b.det_vec) if include_det else 0.0

    has_llm = include_llm and a.has_llm and b.has_llm
    llm_cosine = _cosine(a.llm_vec, b.llm_vec) if has_llm else 0.0
    llm_euclidean = _euclidean(a.llm_vec, b.llm_vec) if has_llm else 0.0
    abs_det_llm_diff = abs(det_cosine - llm_cosine)

    token_jaccard = _jaccard(a.token_set, b.token_set)
    primary_token_jaccard = _jaccard(a.primary_tokens, b.primary_tokens)
    sim_token_jaccard = _jaccard(a.sim_tokens, b.sim_tokens)
    regr_token_jaccard = _jaccard(a.regr_tokens, b.regr_tokens)

    ai = a.info
    bi = b.info

    same_primary_signature = _same_nonempty(
        str(ai.get("primary_signature", "")), str(bi.get("primary_signature", ""))
    )
    same_primary_type = _same_nonempty(
        str(ai.get("primary_type", "")), str(bi.get("primary_type", ""))
    )
    same_mismatch_type = _same_nonempty(
        str(ai.get("mismatch_type", "")), str(bi.get("mismatch_type", ""))
    )
    same_op_pair = _same_nonempty(
        str(ai.get("op_pair", "")), str(bi.get("op_pair", ""))
    )
    same_fatal_file = _same_nonempty(
        str(ai.get("fatal_file", "")), str(bi.get("fatal_file", ""))
    )
    same_failed_reason = _same_nonempty(
        str(ai.get("failed_reason", "")), str(bi.get("failed_reason", ""))
    )
    same_has_uvm_fatal = _same_bool(ai, bi, "has_uvm_fatal")
    same_has_uvm_error = _same_bool(ai, bi, "has_uvm_error")
    same_has_regr_mismatch = _same_bool(ai, bi, "has_regr_mismatch")

    na = int(ai.get("num_tokens", 0))
    nb = int(bi.get("num_tokens", 0))
    abs_num_tokens_diff_log = math.log1p(abs(na - nb))
    min_num_tokens_log = math.log1p(min(na, nb))
    max_num_tokens_log = math.log1p(max(na, nb))

    return np.asarray(
        [
            det_cosine,
            llm_cosine,
            abs_det_llm_diff,
            det_euclidean,
            llm_euclidean,
            token_jaccard,
            primary_token_jaccard,
            sim_token_jaccard,
            regr_token_jaccard,
            same_primary_signature,
            same_primary_type,
            same_mismatch_type,
            same_op_pair,
            same_fatal_file,
            same_failed_reason,
            same_has_uvm_fatal,
            same_has_uvm_error,
            same_has_regr_mismatch,
            abs_num_tokens_diff_log,
            min_num_tokens_log,
            max_num_tokens_log,
        ],
        dtype=np.float32,
    )


def build_llm_pair_feature_matrix(
    features: list[LLMCaseFeature],
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    if not pairs:
        return np.zeros((0, llm_pair_feature_dim()), dtype=np.float32)
    sample = build_llm_pair_feature_vector(features[pairs[0][0]], features[pairs[0][1]])
    dim = len(sample)
    matrix = np.empty((len(pairs), dim), dtype=np.float32)
    for idx, (i, j) in enumerate(pairs):
        matrix[idx] = build_llm_pair_feature_vector(features[i], features[j])
    return matrix


FEATURE_MODES = {
    "summary21",
    "rich",
    "rich_no_llm",
    "rich_no_det",
    "llm_dual",
    "llm_dual_struct",
    "llm_dual_struct_det_summary",
    "llm_dual_struct_det_summary_cross",
    "llm_dual_struct_det_summary_trace",
    "llm_dual_struct_det_summary_trace_struct",
    "llm_dual_struct_det_summary_completion",
}
DUAL_FEATURE_MODES = {
    "llm_dual",
    "llm_dual_struct",
    "llm_dual_struct_det_summary",
    "llm_dual_struct_det_summary_cross",
    "llm_dual_struct_det_summary_trace",
    "llm_dual_struct_det_summary_trace_struct",
    "llm_dual_struct_det_summary_completion",
}


def _relation_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([np.abs(a - b), a * b]).astype(np.float32, copy=False)


def _nonempty_conflict(a: str, b: str) -> float:
    return 1.0 if a and b and a != b else 0.0


def _both_true(ai: dict, bi: dict, key: str) -> float:
    return 1.0 if bool(ai.get(key)) and bool(bi.get(key)) else 0.0


def build_structured_pair_feature_vector(a: LLMCaseFeature, b: LLMCaseFeature) -> np.ndarray:
    ai = a.info
    bi = b.info
    mismatch_a = str(ai.get("mismatch_type", ""))
    mismatch_b = str(bi.get("mismatch_type", ""))
    primary_a = str(ai.get("primary_type", ""))
    primary_b = str(bi.get("primary_type", ""))
    return np.asarray(
        [
            _same_nonempty(str(ai.get("error_source_file", "")), str(bi.get("error_source_file", ""))),
            _same_nonempty(str(ai.get("fatal_file", "")), str(bi.get("fatal_file", ""))),
            _same_nonempty(str(ai.get("uvm_component", "")), str(bi.get("uvm_component", ""))),
            _same_nonempty(str(ai.get("uvm_testname", "")), str(bi.get("uvm_testname", ""))),
            _same_nonempty(str(ai.get("failed_test_name", "")), str(bi.get("failed_test_name", ""))),
            _same_nonempty(mismatch_a, mismatch_b),
            _same_nonempty(str(ai.get("op_pair", "")), str(bi.get("op_pair", ""))),
            _same_nonempty(str(ai.get("ibex_opcode", "")), str(bi.get("ibex_opcode", ""))),
            _same_nonempty(str(ai.get("spike_opcode", "")), str(bi.get("spike_opcode", ""))),
            _same_nonempty(str(ai.get("register_name", "")), str(bi.get("register_name", ""))),
            _same_nonempty(str(ai.get("pc_region", "")), str(bi.get("pc_region", ""))),
            _same_nonempty(str(ai.get("primary_signature", "")), str(bi.get("primary_signature", ""))),
            _same_nonempty(primary_a, primary_b),
            _both_true(ai, bi, "has_uvm_fatal"),
            _both_true(ai, bi, "has_uvm_error"),
            _both_true(ai, bi, "has_regr_mismatch"),
            _nonempty_conflict(mismatch_a, mismatch_b),
            _nonempty_conflict(primary_a, primary_b),
            _jaccard(a.primary_tokens, b.primary_tokens),
            _jaccard(a.sim_tokens, b.sim_tokens),
            _jaccard(a.regr_tokens, b.regr_tokens),
            math.log1p(len(a.token_set & b.token_set)),
            math.log1p(len(a.primary_tokens & b.primary_tokens)),
        ],
        dtype=np.float32,
    )


def build_det_scalar_summary_vector(a: LLMCaseFeature, b: LLMCaseFeature) -> np.ndarray:
    det_cosine = _cosine(a.det_vec, b.det_vec)
    det_euclidean = _euclidean(a.det_vec, b.det_vec)
    ai = a.info
    bi = b.info
    return np.asarray(
        [
            det_cosine,
            det_euclidean,
            _jaccard(a.token_set, b.token_set),
            _jaccard(a.primary_tokens, b.primary_tokens),
            _jaccard(a.sim_tokens, b.sim_tokens),
            _jaccard(a.regr_tokens, b.regr_tokens),
            _same_nonempty(str(ai.get("primary_signature", "")), str(bi.get("primary_signature", ""))),
            _same_nonempty(str(ai.get("primary_type", "")), str(bi.get("primary_type", ""))),
            _same_nonempty(str(ai.get("mismatch_type", "")), str(bi.get("mismatch_type", ""))),
            _same_nonempty(str(ai.get("op_pair", "")), str(bi.get("op_pair", ""))),
        ],
        dtype=np.float32,
    )


def _dual_scalar_features(a: LLMCaseFeature, b: LLMCaseFeature) -> np.ndarray:
    fv_a = a.effective_llm_vec
    fv_b = b.effective_llm_vec
    sv_a = a.effective_llm_summary_vec
    sv_b = b.effective_llm_summary_vec
    features_cos = _cosine(fv_a, fv_b) if fv_a.size and fv_b.size else 0.0
    summary_cos = _cosine(sv_a, sv_b) if sv_a.size and sv_b.size else 0.0
    features_euc = _euclidean(fv_a, fv_b) if fv_a.size and fv_b.size else 0.0
    summary_euc = _euclidean(sv_a, sv_b) if sv_a.size and sv_b.size else 0.0
    return np.asarray(
        [
            features_cos,
            summary_cos,
            abs(features_cos - summary_cos),
            features_euc,
            summary_euc,
        ],
        dtype=np.float32,
    )


def _trace_scalars(a: LLMCaseFeature, b: LLMCaseFeature) -> np.ndarray:
    """Pairwise scalar features for trace embeddings (8 dimensions)."""
    ta = a.effective_trace_vec
    tb = b.effective_trace_vec
    has_a = a.has_trace
    has_b = b.has_trace

    cos_sim = _safe_cosine(ta, tb) if has_a and has_b else 0.0
    euc = _safe_euclidean(ta, tb) if has_a and has_b else 0.0
    l1 = float(np.sum(np.abs(ta - tb))) if has_a and has_b and ta.size else 0.0
    dot = float(np.dot(ta, tb)) if has_a and has_b and ta.size else 0.0

    return np.asarray(
        [
            cos_sim,
            euc,
            l1,
            dot,
            float(has_a and has_b),
            float(has_a != has_b),  # one_missing
            float(not has_a and not has_b),  # both_missing
            float(bool(has_a) ^ bool(has_b)),  # xor_missing
        ],
        dtype=np.float32,
    )


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return _cosine(a, b) if a.size and b.size else 0.0


def _safe_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return _euclidean(a, b) if a.size and b.size else 0.0


def build_dual_cross_scalar_features(a: LLMCaseFeature, b: LLMCaseFeature) -> np.ndarray:
    fv_a = a.effective_llm_vec
    fv_b = b.effective_llm_vec
    sv_a = a.effective_llm_summary_vec
    sv_b = b.effective_llm_summary_vec
    features_cos = _safe_cosine(fv_a, fv_b)
    summary_cos = _safe_cosine(sv_a, sv_b)
    cross_ab = _safe_cosine(fv_a, sv_b)
    cross_ba = _safe_cosine(sv_a, fv_b)
    features_euc = _safe_euclidean(fv_a, fv_b)
    summary_euc = _safe_euclidean(sv_a, sv_b)
    if fv_a.size and fv_b.size:
        fdiff = np.abs(fv_a - fv_b)
        features_l1_mean = float(np.mean(fdiff))
        features_linf = float(np.max(fdiff))
    else:
        features_l1_mean = 0.0
        features_linf = 0.0
    if sv_a.size and sv_b.size:
        sdiff = np.abs(sv_a - sv_b)
        summary_l1_mean = float(np.mean(sdiff))
        summary_linf = float(np.max(sdiff))
    else:
        summary_l1_mean = 0.0
        summary_linf = 0.0
    return np.asarray(
        [
            cross_ab,
            cross_ba,
            0.5 * (cross_ab + cross_ba),
            abs(cross_ab - cross_ba),
            max(features_cos, summary_cos),
            min(features_cos, summary_cos),
            features_cos * summary_cos,
            abs(features_euc - summary_euc),
            features_l1_mean,
            summary_l1_mean,
            abs(features_l1_mean - summary_l1_mean),
            features_linf,
            summary_linf,
            abs(features_linf - summary_linf),
        ],
        dtype=np.float32,
    )


def build_dual_pair_feature_vector(
    a: LLMCaseFeature,
    b: LLMCaseFeature,
    feature_mode: str,
) -> np.ndarray:
    blocks = [
        _relation_block(a.effective_llm_vec, b.effective_llm_vec),
        _relation_block(a.effective_llm_summary_vec, b.effective_llm_summary_vec),
        _dual_scalar_features(a, b),
    ]
    if feature_mode in {"llm_dual_struct", "llm_dual_struct_det_summary", "llm_dual_struct_det_summary_cross", "llm_dual_struct_det_summary_trace", "llm_dual_struct_det_summary_trace_struct", "llm_dual_struct_det_summary_completion"}:
        blocks.append(build_structured_pair_feature_vector(a, b))
    if feature_mode in {"llm_dual_struct_det_summary", "llm_dual_struct_det_summary_cross", "llm_dual_struct_det_summary_trace", "llm_dual_struct_det_summary_trace_struct", "llm_dual_struct_det_summary_completion"}:
        blocks.append(build_det_scalar_summary_vector(a, b))
    if feature_mode == "llm_dual_struct_det_summary_cross":
        blocks.append(build_dual_cross_scalar_features(a, b))
    if feature_mode == "llm_dual_struct_det_summary_trace":
        blocks.append(_relation_block(a.effective_trace_vec, b.effective_trace_vec))
        blocks.append(_trace_scalars(a, b))
    if feature_mode == "llm_dual_struct_det_summary_trace_struct":
        from trace_structured_features import build_trace_structured_pair_features
        blocks.append(_relation_block(a.effective_trace_vec, b.effective_trace_vec))
        blocks.append(_trace_scalars(a, b))
        blocks.append(build_trace_structured_pair_features(
            a.trace_structured, b.trace_structured, trace_mode="tail_anchor",
        ))
    if feature_mode == "llm_dual_struct_det_summary_completion":
        from completion_case_features import build_completion_pair_feature_vector, CompletionCaseFeature
        ca = a.completion_feature if a.completion_feature is not None else CompletionCaseFeature(a.case_id, "missing")
        cb = b.completion_feature if b.completion_feature is not None else CompletionCaseFeature(b.case_id, "missing")
        blocks.append(build_completion_pair_feature_vector(ca, cb))
    return np.concatenate(blocks).astype(np.float32, copy=False)


def build_rich_pair_feature_vector(
    a: LLMCaseFeature,
    b: LLMCaseFeature,
    feature_mode: str = "summary21",
) -> np.ndarray:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    if feature_mode in DUAL_FEATURE_MODES:
        return build_dual_pair_feature_vector(a, b, feature_mode)

    include_llm = feature_mode != "rich_no_llm"
    include_det = feature_mode != "rich_no_det"
    summary = build_llm_pair_feature_vector(
        a, b, include_det=include_det, include_llm=include_llm
    )
    if feature_mode == "summary21":
        return summary

    blocks: list[np.ndarray] = []
    if feature_mode in {"rich", "rich_no_llm"}:
        blocks.append(_relation_block(a.det_vec, b.det_vec))
    if feature_mode in {"rich", "rich_no_det"}:
        blocks.append(_relation_block(a.effective_llm_vec, b.effective_llm_vec))
    blocks.append(summary)
    return np.concatenate(blocks).astype(np.float32, copy=False)


def build_rich_pair_feature_matrix(
    features: list[LLMCaseFeature],
    pairs: list[tuple[int, int]],
    feature_mode: str = "summary21",
) -> np.ndarray:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    if not pairs:
        if not features:
            return np.zeros((0, llm_pair_feature_dim()), dtype=np.float32)
        sample = build_rich_pair_feature_vector(features[0], features[0], feature_mode)
        return np.zeros((0, len(sample)), dtype=np.float32)
    sample = build_rich_pair_feature_vector(features[pairs[0][0]], features[pairs[0][1]], feature_mode)
    matrix = np.empty((len(pairs), len(sample)), dtype=np.float32)
    for idx, (i, j) in enumerate(pairs):
        matrix[idx] = build_rich_pair_feature_vector(features[i], features[j], feature_mode)
    return matrix


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

def _default_hidden_dims(arch: str) -> list[int]:
    if arch == "deep":
        return [512, 512, 256, 256, 128]
    if arch == "residual":
        return [512, 512, 512, 256, 256, 128]
    return [128, 64]


def _make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int] | None = None,
    dropout: float = 0.15,
    arch: str = "shallow",
    layernorm: bool = True,
    batchnorm: bool = False,
):
    import torch
    from torch import nn

    dims = [int(h) for h in (hidden_dims if hidden_dims else _default_hidden_dims(arch))]

    def norm_layer(dim: int) -> nn.Module | None:
        if batchnorm:
            return nn.BatchNorm1d(dim)
        if layernorm:
            return nn.LayerNorm(dim)
        return None

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.Linear(dim, dim)]
            norm = norm_layer(dim)
            if norm is not None:
                layers.append(norm)
            layers.extend([nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(dim, dim)])
            norm = norm_layer(dim)
            if norm is not None:
                layers.append(norm)
            self.net = nn.Sequential(*layers)
            self.out = nn.GELU()

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.out(x + self.net(x))

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            for h in dims:
                layers.append(nn.Linear(prev, h))
                norm = norm_layer(h)
                if norm is not None:
                    layers.append(norm)
                layers.append(nn.GELU())
                layers.append(nn.Dropout(float(dropout)))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x).squeeze(-1)

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            first = dims[0] if dims else 512
            layers: list[nn.Module] = [nn.Linear(input_dim, first)]
            norm = norm_layer(first)
            if norm is not None:
                layers.append(norm)
            layers.extend([nn.GELU(), nn.Dropout(float(dropout))])
            prev = first
            for h in dims[1:]:
                h = int(h)
                if h == prev:
                    layers.append(ResidualBlock(h))
                else:
                    layers.append(nn.Linear(prev, h))
                    norm = norm_layer(h)
                    if norm is not None:
                        layers.append(norm)
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(float(dropout)))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x).squeeze(-1)

    if arch == "residual":
        return ResidualMLP()
    return MLP()


def train_logistic_model(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_scaled, y)
    return {"model": model, "scaler": scaler, "model_type": "logistic"}


def train_gbdt_model(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        early_stopping=False,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X, y)
    return {"model": model, "scaler": None, "model_type": "gbdt"}


def train_mlp_model(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    hidden_dims: Sequence[int] | None = None,
    dropout: float = 0.15,
    batch_size: int = 4096,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    random_state: int = 0,
    mlp_arch: str = "shallow",
    loss: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float | str = "auto",
    early_stop_patience: int = 8,
    layernorm: bool = True,
    batchnorm: bool = False,
    model_arch: str = "auto",
    gate_reg: float = 1e-4,
    ft_d_token: int = 64,
    ft_layers: int = 2,
    ft_heads: int = 4,
    ft_dropout: float = 0.1,
    ft_attention_dropout: float = 0.1,
    ft_ffn_mult: int = 2,
    ft_max_tokens: int = 0,
    validation_clusters: Sequence[dict] | None = None,
) -> Any:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(random_state)
    device = pf.resolve_torch_device(device)

    indices = np.arange(len(y))
    if len(np.unique(y)) == 2 and len(y) >= 20:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=random_state)
        train_idx, val_idx = next(splitter.split(X, y))
    else:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        cut = max(1, int(round(len(indices) * 0.88)))
        train_idx, val_idx = indices[:cut], indices[cut:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    y_train = y[train_idx].astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32) if len(val_idx) else X_train
    y_val = y[val_idx].astype(np.float32) if len(val_idx) else y_train

    from pairwise_neural_models import build_pairwise_neural_model, default_hidden_dims

    effective_arch = "res_mlp" if model_arch == "residual" else model_arch
    hidden = list(hidden_dims) if hidden_dims else (
        _default_hidden_dims(mlp_arch)
        if effective_arch in {"auto", "res_mlp"}
        else default_hidden_dims(effective_arch)
    )
    model_config = {
        "gate_reg": float(gate_reg),
        "ft_d_token": int(ft_d_token),
        "ft_layers": int(ft_layers),
        "ft_heads": int(ft_heads),
        "ft_dropout": float(ft_dropout),
        "ft_attention_dropout": float(ft_attention_dropout),
        "ft_ffn_mult": int(ft_ffn_mult),
        "ft_max_tokens": int(ft_max_tokens),
    }
    if effective_arch == "auto":
        model = _make_mlp(
            input_dim, hidden, dropout,
            arch=mlp_arch, layernorm=layernorm, batchnorm=batchnorm,
        ).to(device)
    else:
        model = build_pairwise_neural_model(
            input_dim, effective_arch, hidden, dropout,
            layernorm=layernorm, batchnorm=batchnorm, **model_config,
        ).to(device)
    pos = float((y_train == 1.0).sum())
    neg = float((y_train == 0.0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    if focal_alpha == "auto":
        alpha_value = neg / max(pos + neg, 1.0)
    else:
        alpha_value = float(focal_alpha)
    alpha_value = max(0.0, min(1.0, float(alpha_value)))

    def compute_loss(logits, target):  # type: ignore[no-untyped-def]
        raw = bce_loss(logits, target)
        if loss == "focal":
            prob = torch.sigmoid(logits)
            pt = torch.where(target > 0.5, prob, 1.0 - prob)
            alpha_t = torch.where(
                target > 0.5,
                torch.full_like(target, alpha_value),
                torch.full_like(target, 1.0 - alpha_value),
            )
            raw = alpha_t * torch.pow(1.0 - pt, float(focal_gamma)) * raw
        return raw.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    micro_batch_size = min(batch_size, 128) if effective_arch == "ft_transformer" else batch_size
    accumulation_steps = max(1, int(math.ceil(batch_size / micro_batch_size)))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=micro_batch_size,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    prepared_cluster_validation = []
    for item in validation_clusters or []:
        prepared_cluster_validation.append({
            **item,
            "X_scaled": scaler.transform(item["X"]).astype(np.float32),
        })

    def validation_cluster_ba() -> float:
        if not prepared_cluster_validation:
            return float("nan")
        from run_experiments import pairwise_scores
        scores = []
        model.eval()
        with torch.no_grad():
            for item in prepared_cluster_validation:
                matrix = item["X_scaled"]
                step = 128 if effective_arch == "ft_transformer" else max(1, len(matrix))
                values = []
                for start in range(0, len(matrix), step):
                    logits = model(torch.from_numpy(matrix[start:start + step]).to(device))
                    values.append(torch.sigmoid(logits).detach().cpu().numpy())
                flat = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
                prob = np.eye(int(item["n"]), dtype=np.float32)
                for (i, j), value in zip(item["pairs"], flat):
                    prob[i, j] = prob[j, i] = float(value)
                pred = cluster_from_probability(prob, int(item["k"]))
                ba, _, _ = pairwise_scores(item["gold"], [f"bucket_{x:03d}" for x in pred])
                scores.append(ba)
        return float(np.mean(scores)) if scores else float("nan")

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    best_val_ba = -1.0
    bad_epochs = 0
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            batch_loss = compute_loss(model(xb), yb)
            if hasattr(model, "regularization_loss"):
                batch_loss = batch_loss + model.regularization_loss()
            (batch_loss / accumulation_steps).backward()
            is_last = batch_index + 1 == len(train_loader)
            if (batch_index + 1) % accumulation_steps == 0 or is_last:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            val_loss = float(compute_loss(model(X_val_t), y_val_t).detach().cpu())
        cluster_ba = validation_cluster_ba()
        if prepared_cluster_validation:
            improved = cluster_ba > best_val_ba + 1e-9 or (
                abs(cluster_ba - best_val_ba) <= 1e-9 and val_loss + 1e-6 < best_val
            )
        else:
            improved = val_loss + 1e-6 < best_val
        if improved:
            best_val = val_loss
            best_val_ba = cluster_ba if prepared_cluster_validation else best_val_ba
            best_epoch = _epoch + 1
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {
        "model": model,
        "scaler": scaler,
        "state_dict": state_dict,
        "input_dim": input_dim,
        "hidden_dims": list(hidden),
        "dropout": dropout,
        "mlp_arch": mlp_arch,
        "model_arch": "legacy_mlp" if effective_arch == "auto" else effective_arch,
        "model_config": model_config,
        "loss": loss,
        "focal_gamma": focal_gamma,
        "focal_alpha": alpha_value,
        "layernorm": layernorm,
        "batchnorm": batchnorm,
        "best_val_loss": best_val,
        "best_val_BA": best_val_ba,
        "best_epoch": best_epoch,
        "model_type": "mlp",
        "device": device,
    }


# ---------------------------------------------------------------------------
# Pair probability prediction (batched)
# ---------------------------------------------------------------------------

def prepare_features_for_model(model_pkg: dict, features: list[LLMCaseFeature]) -> None:
    reduce_dim = int(model_pkg.get("llm_reduce_dim", 0) or 0)
    if reduce_dim > 0:
        apply_llm_reducer(features, model_pkg.get("llm_reducer"), reduce_dim)
        if str(model_pkg.get("feature_mode", "")) in DUAL_FEATURE_MODES:
            apply_llm_summary_reducer(features, model_pkg.get("llm_summary_reducer"), reduce_dim)
    trace_dim = int(model_pkg.get("trace_reduce_dim", 0) or 0)
    if trace_dim > 0:
        apply_trace_reducer(features, model_pkg.get("trace_reducer"), trace_dim)


def _build_model_pair_matrix(
    model_pkg: dict,
    features: list[LLMCaseFeature],
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    feature_mode = str(model_pkg.get("feature_mode", "summary21"))
    return build_rich_pair_feature_matrix(features, pairs, feature_mode=feature_mode)


def predict_probability_matrix_sklearn(
    model_pkg: dict,
    features: list[LLMCaseFeature],
    batch_size: int = 100000,
) -> np.ndarray:
    prepare_features_for_model(model_pkg, features)
    n = len(features)
    probs = np.eye(n, dtype=np.float32)
    if n <= 1:
        return probs

    pairs: list[tuple[int, int]] = []

    def flush() -> None:
        if not pairs:
            return
        X = _build_model_pair_matrix(model_pkg, features, pairs)
        model = model_pkg["model"]
        scaler = model_pkg.get("scaler")
        if scaler is not None:
            X = scaler.transform(X)
        model_type = model_pkg.get("model_type", "")
        if model_type == "mlp":
            import torch
            device = model_pkg.get("device", "cpu")
            model.to(device)
            model.eval()
            neural_batch = 128 if model_pkg.get("model_arch") == "ft_transformer" else len(X)
            chunks = []
            with torch.no_grad():
                for start in range(0, len(X), neural_batch):
                    logits = model(torch.from_numpy(X[start:start + neural_batch].astype(np.float32)).to(device)).detach().cpu()
                    chunks.append(torch.sigmoid(logits).numpy().astype(np.float32))
            batch_probs = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        elif hasattr(model, "predict_proba"):
            batch_probs = model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            batch_probs = np.clip(model.predict(X).astype(np.float32), 1e-6, 1.0 - 1e-6)
        for (i, j), prob in zip(pairs, batch_probs):
            probs[i, j] = float(prob)
            probs[j, i] = float(prob)
        pairs.clear()

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            if len(pairs) >= batch_size:
                flush()
    flush()
    return probs


def predict_probability_matrix_ensemble(
    model_pkgs: list[dict],
    weights: list[float],
    features: list[LLMCaseFeature],
    ensemble_mode: str = "prob_average",
    batch_size: int = 100000,
) -> np.ndarray:
    """Soft voting ensemble of multiple pairwise models.

    ensemble_mode:
      - prob_average: P_ens = sum(w_i * P_i)
      - logit_average: P_ens = sigmoid(sum(w_i * logit(P_i)))
    """
    n = len(features)
    if n <= 1:
        return np.eye(n, dtype=np.float32)
    if len(model_pkgs) != len(weights):
        raise ValueError(
            f"model_pkgs ({len(model_pkgs)}) and weights ({len(weights)}) length mismatch"
        )
    if abs(sum(weights) - 1.0) > 1e-4:
        raise ValueError(f"weights must sum to 1, got {sum(weights)}")

    prob_matrices = []
    for model_pkg in model_pkgs:
        prob_matrices.append(
            predict_probability_matrix_sklearn(model_pkg, features, batch_size=batch_size)
        )

    if ensemble_mode == "prob_average":
        fused = np.zeros((n, n), dtype=np.float64)
        for w, P in zip(weights, prob_matrices):
            fused += w * P.astype(np.float64)
    elif ensemble_mode == "logit_average":
        fused = np.zeros((n, n), dtype=np.float64)
        eps = 1e-9
        for w, P in zip(weights, prob_matrices):
            P_clipped = np.clip(P.astype(np.float64), eps, 1.0 - eps)
            fused += w * np.log(P_clipped / (1.0 - P_clipped))
        fused = 1.0 / (1.0 + np.exp(-fused))
    else:
        raise ValueError(f"unknown ensemble_mode: {ensemble_mode}")

    return fused.astype(np.float32)


def cluster_from_probability(prob: np.ndarray, k: int) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    n = prob.shape[0]
    if n == 0:
        return []
    k = max(1, min(k, n))
    if k == n:
        return list(range(n))
    distance = 1.0 - prob
    np.fill_diagonal(distance, 0.0)
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return model.fit_predict(distance).tolist()


def save_model_pkg(model_pkg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_type = model_pkg.get("model_type", "")
    common = {
        "feature_mode": model_pkg.get("feature_mode", "summary21"),
        "llm_reduce_dim": int(model_pkg.get("llm_reduce_dim", 0) or 0),
        "svd_dim": int(model_pkg.get("svd_dim", 64) or 64),
        "llm_dual": str(model_pkg.get("feature_mode", "")) in DUAL_FEATURE_MODES,
        "trace_reduce_dim": int(model_pkg.get("trace_reduce_dim", 0) or 0),
        "trace_encoder_dir": str(model_pkg.get("trace_encoder_dir", "")),
    }
    if model_type == "mlp":
        import torch
        torch.save(
            {
                "state_dict": model_pkg["state_dict"],
                "input_dim": model_pkg["input_dim"],
                "hidden_dims": model_pkg["hidden_dims"],
                "dropout": model_pkg["dropout"],
                "mlp_arch": model_pkg.get("mlp_arch", "shallow"),
                "model_arch": model_pkg.get(
                    "model_arch",
                    "res_mlp" if model_pkg.get("mlp_arch") == "residual" else "legacy_mlp",
                ),
                "model_config": dict(model_pkg.get("model_config", {})),
                "best_val_loss": float(model_pkg.get("best_val_loss", 0.0)),
                "best_val_BA": float(model_pkg.get("best_val_BA", -1.0)),
                "best_epoch": int(model_pkg.get("best_epoch", 0)),
                "feature_schema_version": int(model_pkg.get("feature_schema_version", 1)),
                "layernorm": bool(model_pkg.get("layernorm", True)),
                "batchnorm": bool(model_pkg.get("batchnorm", False)),
                "model_type": "mlp",
                **common,
            },
            path,
        )
        preproc = {
            "scaler": model_pkg.get("scaler"),
            "llm_reducer": model_pkg.get("llm_reducer"),
            "llm_summary_reducer": model_pkg.get("llm_summary_reducer"),
            "trace_reducer": model_pkg.get("trace_reducer"),
            **common,
        }
        import pickle
        with open(path.with_suffix(".preproc.pkl"), "wb") as f:
            pickle.dump(preproc, f)
        # Compatibility with older loaders. The torch checkpoint stays sklearn-free.
        if model_pkg.get("scaler") is not None:
            with open(path.with_suffix(".scaler.pkl"), "wb") as f:
                pickle.dump(model_pkg.get("scaler"), f)
    else:
        import pickle
        save_obj = {
            "model": model_pkg["model"],
            "scaler": model_pkg.get("scaler"),
            "model_type": model_type,
            "llm_reducer": model_pkg.get("llm_reducer"),
            "llm_summary_reducer": model_pkg.get("llm_summary_reducer"),
            "trace_reducer": model_pkg.get("trace_reducer"),
            **common,
        }
        with open(path, "wb") as f:
            pickle.dump(save_obj, f)


def load_model_pkg(path: Path) -> dict:
    if path.suffix in (".pt", ".pth"):
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_type") == "mlp":
            from pairwise_neural_models import build_pairwise_neural_model
            legacy_mlp_arch = checkpoint.get("mlp_arch", "shallow")
            model_arch = checkpoint.get("model_arch")
            if not model_arch:
                if legacy_mlp_arch == "residual":
                    model_arch = "res_mlp"
                else:
                    model = _make_mlp(
                        int(checkpoint["input_dim"]),
                        hidden_dims=checkpoint.get("hidden_dims", (128, 64)),
                        dropout=float(checkpoint.get("dropout", 0.15)),
                        arch=legacy_mlp_arch,
                        layernorm=bool(checkpoint.get("layernorm", True)),
                        batchnorm=bool(checkpoint.get("batchnorm", False)),
                    )
            if model_arch == "legacy_mlp":
                model = _make_mlp(
                    int(checkpoint["input_dim"]),
                    hidden_dims=checkpoint.get("hidden_dims", (128, 64)),
                    dropout=float(checkpoint.get("dropout", 0.15)),
                    arch=legacy_mlp_arch,
                    layernorm=bool(checkpoint.get("layernorm", True)),
                    batchnorm=bool(checkpoint.get("batchnorm", False)),
                )
            elif model_arch:
                model = build_pairwise_neural_model(
                    int(checkpoint["input_dim"]), model_arch,
                    hidden_dims=checkpoint.get("hidden_dims", (128, 64)),
                    dropout=float(checkpoint.get("dropout", 0.15)),
                    layernorm=bool(checkpoint.get("layernorm", True)),
                    batchnorm=bool(checkpoint.get("batchnorm", False)),
                    **dict(checkpoint.get("model_config", {})),
                )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            preproc = {}
            preproc_path = path.with_suffix(".preproc.pkl")
            if preproc_path.exists():
                import pickle
                with open(preproc_path, "rb") as f:
                    preproc = pickle.load(f)
            else:
                scaler_path = path.with_suffix(".scaler.pkl")
                if scaler_path.exists():
                    import pickle
                    with open(scaler_path, "rb") as f:
                        preproc["scaler"] = pickle.load(f)
            return {
                "model": model,
                "scaler": preproc.get("scaler"),
                "llm_reducer": preproc.get("llm_reducer"),
                "llm_summary_reducer": preproc.get("llm_summary_reducer"),
                "trace_reducer": preproc.get("trace_reducer"),
                "state_dict": checkpoint["state_dict"],
                "input_dim": int(checkpoint["input_dim"]),
                "hidden_dims": checkpoint.get("hidden_dims", (128, 64)),
                "dropout": float(checkpoint.get("dropout", 0.15)),
                "mlp_arch": checkpoint.get("mlp_arch", "shallow"),
                "model_arch": checkpoint.get("model_arch", "res_mlp" if checkpoint.get("mlp_arch") == "residual" else "legacy_mlp"),
                "model_config": dict(checkpoint.get("model_config", {})),
                "best_val_loss": float(checkpoint.get("best_val_loss", 0.0)),
                "best_val_BA": float(checkpoint.get("best_val_BA", -1.0)),
                "best_epoch": int(checkpoint.get("best_epoch", 0)),
                "feature_schema_version": int(checkpoint.get("feature_schema_version", 1)),
                "layernorm": bool(checkpoint.get("layernorm", True)),
                "batchnorm": bool(checkpoint.get("batchnorm", False)),
                "feature_mode": checkpoint.get("feature_mode", preproc.get("feature_mode", "summary21")),
                "llm_reduce_dim": int(checkpoint.get("llm_reduce_dim", preproc.get("llm_reduce_dim", 0)) or 0),
                "svd_dim": int(checkpoint.get("svd_dim", preproc.get("svd_dim", 64)) or 64),
                "llm_dual": bool(checkpoint.get("llm_dual", preproc.get("llm_dual", False))),
                "trace_reduce_dim": int(checkpoint.get("trace_reduce_dim", preproc.get("trace_reduce_dim", 0)) or 0),
                "trace_encoder_dir": str(checkpoint.get("trace_encoder_dir", preproc.get("trace_encoder_dir", ""))),
                "model_type": "mlp",
                "device": "cpu",
            }
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
