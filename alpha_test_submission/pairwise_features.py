#!/usr/bin/env python3
"""Shared features for the experimental pairwise same-bug MLP backend.

This module intentionally reads only input.csv and the referenced sim/regr logs.
It never reads gold.csv or meta.csv; training scripts provide labels separately.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import regr_fail_bucketing as rfb

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
except ImportError:  # pragma: no cover - pairwise MLP requires sklearn.
    TruncatedSVD = None
    Normalizer = None


FEATURE_SCHEMA_VERSION = 2
DEFAULT_DRAIN_DEPTH = 4
DEFAULT_DRAIN_ST = 0.45
DEFAULT_DRAIN_MAX_CHILDREN = 100


@dataclass
class CaseFeature:
    case_id: str
    tokens: list[str]
    token_set: set[str]
    primary_tokens: set[str]
    sim_tokens: set[str]
    regr_tokens: set[str]
    dense_vec: np.ndarray
    info: dict


@dataclass
class VectorizerBundle:
    svd_dim: int
    dense_dim: int
    sparse_shape: tuple[int, int]
    template_count: int
    svd: Any = None
    normalizer: Any = None


def _make_parser_args(parser: str) -> argparse.Namespace:
    return argparse.Namespace(
        parser=parser,
        drain_depth=DEFAULT_DRAIN_DEPTH,
        drain_st=DEFAULT_DRAIN_ST,
        drain_max_children=DEFAULT_DRAIN_MAX_CHILDREN,
    )


def _extract_test_name(lines: Sequence[str]) -> str:
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


def _extract_failed_test_name(lines: Sequence[str]) -> str:
    for line in lines:
        if "failed" not in line.lower():
            continue
        match = re.search(r"([A-Za-z0-9_.:+-]+)\s*(?:\[[^\]]+\])?\s*(?:FAILED|failed)", line)
        if match:
            return match.group(1)
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


def _mismatch_type(primary_tokens: Sequence[str], regr_lines: Sequence[str]) -> str:
    for token in primary_tokens:
        if token in {
            "PRIMARY_REGR_PC_MISMATCH",
            "PRIMARY_REGR_REGISTER_WRITE_DATA_MISMATCH",
            "PRIMARY_REGR_COSIM_MISMATCH",
            "PRIMARY_REGR_MEMORY_MISMATCH",
            "PRIMARY_REGR_INSTRUCTION_MISMATCH",
            "PRIMARY_REGR_MISMATCH_GENERIC",
        }:
            return token
    lower = "\n".join(regr_lines).lower()
    if "mismatch" in lower:
        return "PRIMARY_REGR_MISMATCH_GENERIC"
    return ""


def _op_pair(primary_tokens: Sequence[str]) -> str:
    for token in primary_tokens:
        if token.startswith("PRIMARY_REGR_OPPAIR_"):
            return token
    return ""


def _case_info(sim_lines: Sequence[str], regr_lines: Sequence[str], primary_tokens: Sequence[str]) -> dict:
    joined_sim = "\n".join(sim_lines)
    joined_regr = "\n".join(regr_lines)
    lower_regr = joined_regr.lower()
    failed_reason = ""
    fatal_file = ""
    for line in sim_lines:
        upper = line.upper()
        if "UVM_FATAL" in upper or "UVM_ERROR" in upper:
            fatal_file = rfb.extract_sv_basename(line)
            failed_reason = rfb.sanitize_primary_part(rfb.message_after_uvm_context(line))
            break
    if not failed_reason:
        for line in list(regr_lines) + list(sim_lines):
            if "failed" in line.lower():
                failed_reason = rfb.sanitize_primary_part(line)
                break

    primary_signature = primary_tokens[0] if primary_tokens else "PRIMARY_UNKNOWN_FAILURE"
    return {
        "has_uvm_fatal": "UVM_FATAL" in joined_sim.upper(),
        "has_uvm_error": "UVM_ERROR" in joined_sim.upper(),
        "has_regr_mismatch": "mismatch" in lower_regr,
        "primary_type": _primary_type(primary_tokens),
        "primary_signature": primary_signature,
        "mismatch_type": _mismatch_type(primary_tokens, regr_lines),
        "op_pair": _op_pair(primary_tokens),
        "fatal_file": fatal_file,
        "failed_reason": failed_reason,
        "uvm_testname": _extract_test_name(sim_lines),
        "failed_test_name": _extract_failed_test_name(regr_lines),
        "num_tokens": 0,
    }


def _collect_case_inputs(input_csv: Path, parser: str) -> tuple[list[str], list[Counter], list[list[tuple[str, str]]], list[dict]]:
    rows, fields = rfb.read_csv_rows(input_csv)
    sim_col = rfb.pick_column(fields, "sim")
    regr_col = rfb.pick_column(fields, "regr")
    if sim_col is None and regr_col is None:
        raise ValueError(f"{input_csv}: no sim/regr log columns")

    preserve_basenames = parser == "drain"
    case_ids: list[str] = []
    base_features: list[Counter] = []
    normalized_lines: list[list[tuple[str, str]]] = []
    infos: list[dict] = []

    for idx, row in enumerate(rows):
        feats: Counter = Counter()
        lines_for_case: list[tuple[str, str]] = []
        selected_by_prefix: dict[str, list[str]] = {"sim": [], "regr": []}
        info_by_prefix: dict[str, dict] = {"sim": {}, "regr": {}}
        used_cols = [c for c in (sim_col, regr_col) if c]
        case_id = rfb.case_id_from_row(row, idx, used_cols)
        case_ids.append(case_id)
        feats[f"case_shape:{re.sub(r'\\d', '0', case_id)}"] += 1

        for prefix, col in (("sim", sim_col), ("regr", regr_col)):
            path = rfb.resolve_log_path(input_csv, row.get(col) if col else None)
            text, status = rfb.read_log_sample(path)
            selected = rfb.select_lines(text)
            selected_by_prefix[prefix] = selected
            info_by_prefix[prefix] = {"path": str(path) if path else "", "status": status}
            feats[f"{prefix}:file_status:{status}"] += 3
            if path is not None:
                feats[f"{prefix}:basename:{path.name.lower()}"] += 1
            rfb.extract_status_features(prefix, text, feats)
            for line in selected:
                normalized = rfb.normalize_log_line(line, preserve_basenames=preserve_basenames)
                if normalized:
                    lines_for_case.append((prefix, normalized))

        primary_tokens = rfb.extract_primary_signature(
            info_by_prefix["sim"],
            info_by_prefix["regr"],
            selected_by_prefix["sim"],
            selected_by_prefix["regr"],
        )
        for primary in primary_tokens:
            feats[primary] += 1

        base_features.append(feats)
        normalized_lines.append(lines_for_case)
        infos.append(_case_info(selected_by_prefix["sim"], selected_by_prefix["regr"], primary_tokens))

    return case_ids, base_features, normalized_lines, infos


def _dense_matrix(feature_counters: Sequence[Counter], svd_dim: int) -> tuple[np.ndarray, VectorizerBundle]:
    if not rfb.SKLEARN_AVAILABLE or TruncatedSVD is None or Normalizer is None:
        raise RuntimeError("pairwise_mlp requires numpy and scikit-learn for TF-IDF/SVD features")
    sparse, sparse_shape, sklearn_input = rfb.vectorize_features(feature_counters)
    if not sklearn_input:
        raise RuntimeError("pairwise_mlp requires sklearn sparse TF-IDF features")

    n_samples, n_features = sparse.shape
    dense_dim = max(1, svd_dim)
    n_components = min(dense_dim, n_samples - 1, n_features - 1)
    svd = None
    normalizer = None
    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=rfb.RANDOM_SEED)
        dense = svd.fit_transform(sparse)
        normalizer = Normalizer(copy=False)
        dense = normalizer.fit_transform(dense)
    else:
        dense = sparse.toarray() if hasattr(sparse, "toarray") else np.asarray(sparse)
        norm = np.linalg.norm(dense, axis=1, keepdims=True)
        dense = dense / np.maximum(norm, 1e-12)
    dense = np.asarray(dense, dtype=np.float32)
    if dense.shape[1] < dense_dim:
        pad = np.zeros((dense.shape[0], dense_dim - dense.shape[1]), dtype=np.float32)
        dense = np.hstack([dense, pad])
    elif dense.shape[1] > dense_dim:
        dense = dense[:, :dense_dim]
    bundle = VectorizerBundle(
        svd_dim=svd_dim,
        dense_dim=dense_dim,
        sparse_shape=tuple(sparse_shape),
        template_count=0,
        svd=svd,
        normalizer=normalizer,
    )
    return dense.astype(np.float32, copy=False), bundle


def build_case_features_for_inputs(
    input_csvs: Sequence[str | Path],
    parser: str = "drain",
    svd_dim: int = 128,
    token_weights: str | Path | None = None,
    token_weight_mode: str = "none",
) -> tuple[list[CaseFeature], VectorizerBundle]:
    parser_args = _make_parser_args(parser)
    all_case_ids: list[str] = []
    all_base: list[Counter] = []
    all_lines: list[list[tuple[str, str]]] = []
    all_infos: list[dict] = []
    for input_csv in input_csvs:
        case_ids, base, lines, infos = _collect_case_inputs(Path(input_csv).resolve(), parser)
        all_case_ids.extend(case_ids)
        all_base.extend(base)
        all_lines.extend(lines)
        all_infos.extend(infos)

    weights = rfb.load_token_weights(token_weights) if token_weights and token_weight_mode != "none" else {}
    counters, template_count = rfb.build_feature_counters(
        parser_args,
        all_base,
        all_lines,
        token_weights=weights,
        token_weight_mode=token_weight_mode,
    )
    dense, bundle = _dense_matrix(counters, svd_dim)
    bundle.template_count = template_count

    features: list[CaseFeature] = []
    for case_id, counter, info, vec in zip(all_case_ids, counters, all_infos, dense):
        tokens = list(counter.elements())
        token_set = set(counter.keys())
        primary = {token for token in token_set if token.startswith("PRIMARY_")}
        sim_tokens = {token for token in token_set if token.startswith("sim:")}
        regr_tokens = {token for token in token_set if token.startswith("regr:")}
        info = dict(info)
        info["num_tokens"] = int(sum(counter.values()))
        features.append(
            CaseFeature(
                case_id=case_id,
                tokens=tokens,
                token_set=token_set,
                primary_tokens=primary,
                sim_tokens=sim_tokens,
                regr_tokens=regr_tokens,
                dense_vec=vec.astype(np.float32, copy=False),
                info=info,
            )
        )
    return features, bundle


def build_case_features(
    input_csv: str | Path,
    parser: str = "drain",
    svd_dim: int = 128,
    token_weights: str | Path | None = None,
    token_weight_mode: str = "none",
) -> tuple[list[CaseFeature], VectorizerBundle]:
    return build_case_features_for_inputs(
        [input_csv],
        parser=parser,
        svd_dim=svd_dim,
        token_weights=token_weights,
        token_weight_mode=token_weight_mode,
    )


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _dice(a: set[str], b: set[str]) -> float:
    denom = len(a) + len(b)
    return (2.0 * len(a & b) / denom) if denom else 1.0


def _containment(a: set[str], b: set[str]) -> tuple[float, float]:
    inter = len(a & b)
    min_size = min(len(a), len(b))
    max_size = max(len(a), len(b))
    return (
        inter / min_size if min_size else 1.0,
        inter / max_size if max_size else 1.0,
    )


def _same_nonempty(a: str, b: str) -> float:
    return 1.0 if a and b and a == b else 0.0


def _same_bool(a: dict, b: dict, key: str) -> float:
    return 1.0 if bool(a.get(key)) == bool(b.get(key)) else 0.0


def _nonempty_conflict(a: str, b: str) -> float:
    return 1.0 if a and b and a != b else 0.0


def build_pair_feature_vector(a: CaseFeature, b: CaseFeature) -> np.ndarray:
    av = a.dense_vec.astype(np.float32, copy=False)
    bv = b.dense_vec.astype(np.float32, copy=False)
    abs_diff = np.abs(av - bv)
    product = av * bv
    dot = float(np.dot(av, bv))
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    cosine = dot / max(an * bn, 1e-12)
    euclidean = float(np.linalg.norm(av - bv))
    l1_mean = float(np.mean(np.abs(av - bv)))
    linf = float(np.max(np.abs(av - bv))) if av.size else 0.0

    ai = a.info
    bi = b.info
    all_contain_min, all_contain_max = _containment(a.token_set, b.token_set)
    sim_contain_min, sim_contain_max = _containment(a.sim_tokens, b.sim_tokens)
    regr_contain_min, regr_contain_max = _containment(a.regr_tokens, b.regr_tokens)
    primary_conflict = 1.0 if a.primary_tokens and b.primary_tokens and not (a.primary_tokens & b.primary_tokens) else 0.0
    scalar = np.asarray(
        [
            cosine,
            euclidean,
            l1_mean,
            linf,
            _jaccard(a.token_set, b.token_set),
            _dice(a.token_set, b.token_set),
            all_contain_min,
            all_contain_max,
            _jaccard(a.primary_tokens, b.primary_tokens),
            _dice(a.primary_tokens, b.primary_tokens),
            primary_conflict,
            _jaccard(a.sim_tokens, b.sim_tokens),
            _dice(a.sim_tokens, b.sim_tokens),
            sim_contain_min,
            sim_contain_max,
            _jaccard(a.regr_tokens, b.regr_tokens),
            _dice(a.regr_tokens, b.regr_tokens),
            regr_contain_min,
            regr_contain_max,
            math.log1p(len(a.token_set & b.token_set)),
            math.log1p(len(a.primary_tokens & b.primary_tokens)),
            _same_nonempty(ai.get("primary_signature", ""), bi.get("primary_signature", "")),
            _same_nonempty(ai.get("primary_type", ""), bi.get("primary_type", "")),
            _same_nonempty(ai.get("mismatch_type", ""), bi.get("mismatch_type", "")),
            _same_nonempty(ai.get("op_pair", ""), bi.get("op_pair", "")),
            _same_nonempty(ai.get("fatal_file", ""), bi.get("fatal_file", "")),
            _same_nonempty(ai.get("failed_reason", ""), bi.get("failed_reason", "")),
            _same_nonempty(ai.get("uvm_testname", ""), bi.get("uvm_testname", "")),
            _same_nonempty(ai.get("failed_test_name", ""), bi.get("failed_test_name", "")),
            _nonempty_conflict(ai.get("primary_signature", ""), bi.get("primary_signature", "")),
            _nonempty_conflict(ai.get("mismatch_type", ""), bi.get("mismatch_type", "")),
            _same_bool(ai, bi, "has_uvm_fatal"),
            _same_bool(ai, bi, "has_uvm_error"),
            _same_bool(ai, bi, "has_regr_mismatch"),
            1.0 if bool(ai.get("has_uvm_fatal")) and bool(bi.get("has_uvm_fatal")) else 0.0,
            1.0 if bool(ai.get("has_uvm_error")) and bool(bi.get("has_uvm_error")) else 0.0,
            1.0 if bool(ai.get("has_regr_mismatch")) and bool(bi.get("has_regr_mismatch")) else 0.0,
            math.log1p(abs(int(ai.get("num_tokens", 0)) - int(bi.get("num_tokens", 0)))),
            math.log1p(min(int(ai.get("num_tokens", 0)), int(bi.get("num_tokens", 0)))),
            math.log1p(max(int(ai.get("num_tokens", 0)), int(bi.get("num_tokens", 0)))),
        ],
        dtype=np.float32,
    )
    return np.concatenate([abs_diff, product, scalar]).astype(np.float32, copy=False)


def pair_feature_dim(dense_dim: int) -> int:
    return 2 * dense_dim + 40


def build_pair_feature_matrix(
    features: list[CaseFeature],
    pairs: list[tuple[int, int]],
    batch_size: int | None = None,
) -> np.ndarray:
    del batch_size
    if not pairs:
        dense_dim = len(features[0].dense_vec) if features else 1
        return np.zeros((0, pair_feature_dim(dense_dim)), dtype=np.float32)
    matrix = np.empty((len(pairs), len(build_pair_feature_vector(features[pairs[0][0]], features[pairs[0][1]]))), dtype=np.float32)
    for idx, (i, j) in enumerate(pairs):
        matrix[idx] = build_pair_feature_vector(features[i], features[j])
    return matrix


def build_pairwise_mlp_model(
    input_dim: int,
    hidden_dims: Sequence[int] = (256, 128),
    dropout: float = 0.2,
    architecture: str = "plain",
):
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            self.out = nn.GELU()

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.out(x + self.net(x))

    class PairwiseMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            for hidden in hidden_dims:
                layers.append(nn.Linear(prev, int(hidden)))
                if architecture in {"layernorm", "residual"}:
                    layers.append(nn.LayerNorm(int(hidden)))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(dropout)))
                prev = int(hidden)
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x).squeeze(-1)

    class ResidualPairwiseMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = [int(dim) for dim in hidden_dims] or [256, 256, 128]
            layers: list[nn.Module] = [
                nn.Linear(input_dim, dims[0]),
                nn.LayerNorm(dims[0]),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
            prev = dims[0]
            for hidden in dims[1:]:
                hidden = int(hidden)
                if hidden == prev:
                    layers.append(ResidualBlock(hidden))
                else:
                    layers.extend(
                        [
                            nn.Linear(prev, hidden),
                            nn.LayerNorm(hidden),
                            nn.GELU(),
                            nn.Dropout(float(dropout)),
                        ]
                    )
                prev = hidden
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x).squeeze(-1)

    if architecture == "residual":
        return ResidualPairwiseMLP()
    return PairwiseMLP()


def resolve_torch_device(device: str) -> str:
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def predict_probability_matrix(
    model,
    features: list[CaseFeature],
    device: str = "cpu",
    batch_size: int = 100000,
    prob_bias: float = 0.0,
    prob_temperature: float = 1.0,
) -> np.ndarray:
    import torch

    n = len(features)
    probs = np.eye(n, dtype=np.float32)
    if n <= 1:
        return probs
    device = resolve_torch_device(device)
    model.to(device)
    model.eval()
    pairs: list[tuple[int, int]] = []

    def flush() -> None:
        if not pairs:
            return
        X = build_pair_feature_matrix(features, pairs)
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device)).detach().cpu()
            batch_probs = torch.sigmoid(logits).numpy().astype(np.float32)
        clipped = np.clip(batch_probs, 1e-6, 1.0 - 1e-6)
        logit = np.log(clipped / (1.0 - clipped))
        adjusted = 1.0 / (1.0 + np.exp(-((logit - prob_bias) / max(prob_temperature, 1e-6))))
        for (i, j), prob in zip(pairs, adjusted.astype(np.float32)):
            probs[i, j] = prob
            probs[j, i] = prob
        pairs.clear()

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            if len(pairs) >= batch_size:
                flush()
    flush()
    return probs


def _useful_primary(signature: str) -> bool:
    return bool(signature) and signature != "PRIMARY_UNKNOWN_FAILURE"


def calibrate_probability_matrix(
    probs: np.ndarray,
    features: list[CaseFeature],
    primary_floor: float = 0.70,
    op_pair_floor: float = 0.65,
    mismatch_floor: float = 0.55,
    conflict_penalty: float = 0.05,
    cosine_gate: float = 0.20,
) -> np.ndarray:
    """Lightweight deterministic calibration for the experimental pairwise backend.

    The learned MLP was under-merging on larger datasets. These floors only apply
    to stable, label-free signatures extracted from sim/regr logs and are kept
    configurable so validation can disable or tune them.
    """
    out = probs.astype(np.float32, copy=True)
    n = len(features)
    for i in range(n):
        ai = features[i].info
        av = features[i].dense_vec
        for j in range(i + 1, n):
            bj = features[j].info
            bv = features[j].dense_vec
            p = float(out[i, j])
            a_sig = str(ai.get("primary_signature", ""))
            b_sig = str(bj.get("primary_signature", ""))
            if primary_floor > 0 and _useful_primary(a_sig) and a_sig == b_sig:
                p = max(p, primary_floor)
            a_op = str(ai.get("op_pair", ""))
            b_op = str(bj.get("op_pair", ""))
            if op_pair_floor > 0 and a_op and a_op == b_op:
                p = max(p, op_pair_floor)
            a_mismatch = str(ai.get("mismatch_type", ""))
            b_mismatch = str(bj.get("mismatch_type", ""))
            if mismatch_floor > 0 and a_mismatch and a_mismatch == b_mismatch:
                denom = max(float(np.linalg.norm(av) * np.linalg.norm(bv)), 1e-12)
                cosine = float(np.dot(av, bv) / denom)
                if cosine >= cosine_gate:
                    p = max(p, mismatch_floor)
            if conflict_penalty > 0 and _useful_primary(a_sig) and _useful_primary(b_sig) and a_sig != b_sig:
                p = max(0.0, p - conflict_penalty)
            out[i, j] = p
            out[j, i] = p
    return out
