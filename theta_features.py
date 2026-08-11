#!/usr/bin/env python3
"""Compact multi-granular evidence features for the experimental Theta route.

Theta reads only input.csv and its referenced sim/regr logs at inference time.
Gold labels are deliberately absent from this module.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import multigranular_features as mgf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb


@dataclass
class ThetaCaseFeature:
    case_id: str
    base: plf.LLMCaseFeature
    evidence: mgf.CaseEvidence
    evidence_doc: str
    context_doc: str
    fingerprint: str
    evidence_vec: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    context_vec: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    evidence_reduced: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    context_reduced: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


def _event_sequence(evidence: mgf.CaseEvidence) -> str:
    return " > ".join(
        f"{event.source}:{event.event_type}:{event.object_type}"
        for event in evidence.events[:24]
    ) or "none"


def build_evidence_document(evidence: mgf.CaseEvidence) -> str:
    """Build the compact semantic view without case-specific identifiers."""
    counts: dict[str, int] = {}
    for event in evidence.events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    count_text = ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"
    return "\n".join(
        [
            "REGRESSION_FAILURE_EVIDENCE",
            f"event_sequence: {_event_sequence(evidence)}",
            f"event_counts: {count_text}",
            f"states: {','.join(evidence.states) or 'none'}",
            f"pc_regions: {','.join(evidence.pc_regions) or 'none'}",
            f"opcodes: {','.join(evidence.opcodes) or 'none'}",
            f"registers: {','.join(evidence.registers) or 'none'}",
            f"csrs: {','.join(evidence.csrs) or 'none'}",
            f"sim_status: {evidence.sim_status}",
            f"regr_status: {evidence.regr_status}",
        ]
    )


def build_context_document(evidence: mgf.CaseEvidence, max_events: int = 5) -> str:
    """Keep only the strongest unique failure windows from sim and regr."""
    selected = sorted(
        evidence.events,
        key=lambda event: (-event.severity, event.source, event.position),
    )
    seen: set[str] = set()
    lines = ["REGRESSION_FAILURE_LOCAL_CONTEXT"]
    for event in selected:
        context = event.context.strip()
        if not context or context in seen:
            continue
        seen.add(context)
        lines.append(
            f"event={event.event_type} source={event.source} "
            f"object={event.object_type} severity={event.severity} "
            f"position={event.relative_position:.3f}"
        )
        lines.append(f"context: {context}")
        if len(seen) >= max(1, int(max_events)):
            break
    if not seen:
        lines.append("context: none")
    return "\n".join(lines)


def _visible_fingerprints(input_csv: Path) -> list[str]:
    rows, fields = rfb.read_csv_rows(input_csv)
    sim_col = rfb.pick_column(fields, "sim")
    regr_col = rfb.pick_column(fields, "regr")
    values: list[str] = []
    for row in rows:
        sim_path = rfb.resolve_log_path(input_csv, row.get(sim_col) if sim_col else None)
        regr_path = rfb.resolve_log_path(input_csv, row.get(regr_col) if regr_col else None)
        sim_text, _ = rfb.read_log_sample(sim_path)
        regr_text, _ = rfb.read_log_sample(regr_path)
        visible = sim_text + "\n<REGR_LOG>\n" + regr_text
        values.append(hashlib.sha256(visible.encode("utf-8", errors="replace")).hexdigest())
    return values


def build_theta_case_features(
    input_csv: str | Path,
    parser: str = "drain",
    svd_dim: int = 64,
    context_radius: int = 2,
    context_events: int = 5,
) -> tuple[list[ThetaCaseFeature], list[dict]]:
    input_path = Path(input_csv).resolve()
    base, _ = plf.build_llm_case_features_for_inputs(
        [input_path], parser=parser, svd_dim=svd_dim, llm_args=None,
        log_llm_disabled=False,
    )
    evidence, debug = mgf.build_case_evidence([input_path], context_radius=context_radius)
    fingerprints = _visible_fingerprints(input_path)
    if not (len(base) == len(evidence) == len(fingerprints)):
        raise RuntimeError(
            f"Theta feature alignment failed for {input_path}: "
            f"base={len(base)} evidence={len(evidence)} fingerprints={len(fingerprints)}"
        )
    output: list[ThetaCaseFeature] = []
    for base_item, evidence_item, fingerprint in zip(base, evidence, fingerprints):
        output.append(
            ThetaCaseFeature(
                case_id=base_item.case_id,
                base=base_item,
                evidence=evidence_item,
                evidence_doc=build_evidence_document(evidence_item),
                context_doc=build_context_document(evidence_item, context_events),
                fingerprint=fingerprint,
            )
        )
    return output, debug


def fetch_theta_embeddings(
    cases: Sequence[ThetaCaseFeature],
    cache_dir: str | Path,
    batch_size: int = 128,
    timeout_sec: float = 120.0,
) -> tuple[str, int, dict[str, int]]:
    docs = [case.evidence_doc for case in cases] + [case.context_doc for case in cases]
    unique_docs = list(dict.fromkeys(docs))
    index = {doc: idx for idx, doc in enumerate(unique_docs)}
    inverse = np.fromiter((index[doc] for doc in docs), dtype=np.int64, count=len(docs))
    args = argparse.Namespace(
        llm_cache_dir=Path(cache_dir),
        llm_batch_size=int(batch_size),
        llm_timeout_sec=float(timeout_sec),
    )
    vectors, model_name = rfb.fetch_llm_embeddings(unique_docs, args)
    unique = np.asarray(vectors, dtype=np.float32)
    if unique.ndim != 2 or unique.shape[0] != len(unique_docs):
        raise RuntimeError(f"unexpected Theta embedding shape: {unique.shape}")
    unique /= np.maximum(np.linalg.norm(unique, axis=1, keepdims=True), np.float32(1e-12))
    matrix = unique[inverse]
    count = len(cases)
    for idx, case in enumerate(cases):
        case.evidence_vec = matrix[idx]
        case.context_vec = matrix[count + idx]
    return model_name, int(unique.shape[1]), {
        "documents": len(docs),
        "unique_documents": len(unique_docs),
    }


def fit_theta_reducers(
    cases: Sequence[ThetaCaseFeature],
    dim: int,
    random_state: int,
) -> dict[str, Any]:
    if not cases or not cases[0].evidence_vec.size:
        for case in cases:
            case.evidence_reduced = np.zeros(0, dtype=np.float32)
            case.context_reduced = np.zeros(0, dtype=np.float32)
        return {"evidence": None, "context": None, "dim": int(dim)}
    evidence = np.vstack([case.evidence_vec for case in cases]).astype(np.float32)
    context = np.vstack([case.context_vec for case in cases]).astype(np.float32)
    evidence_reducer, evidence_reduced = plf._fit_reducer_for_matrix(
        evidence, int(dim), random_state
    )
    context_reducer, context_reduced = plf._fit_reducer_for_matrix(
        context, int(dim), random_state + 17
    )
    for case, left, right in zip(cases, evidence_reduced, context_reduced):
        case.evidence_reduced = left.astype(np.float32, copy=False)
        case.context_reduced = right.astype(np.float32, copy=False)
    return {"evidence": evidence_reducer, "context": context_reducer, "dim": int(dim)}


def apply_theta_reducers(cases: Sequence[ThetaCaseFeature], reducers: dict[str, Any]) -> None:
    dim = int(reducers.get("dim", 0))
    if not cases or not cases[0].evidence_vec.size or dim <= 0:
        for case in cases:
            case.evidence_reduced = np.zeros(0, dtype=np.float32)
            case.context_reduced = np.zeros(0, dtype=np.float32)
        return
    evidence = np.vstack([case.evidence_vec for case in cases]).astype(np.float32)
    context = np.vstack([case.context_vec for case in cases]).astype(np.float32)
    left = plf._apply_reducer_to_matrix(evidence, reducers.get("evidence"), dim)
    right = plf._apply_reducer_to_matrix(context, reducers.get("context"), dim)
    for case, evidence_vec, context_vec in zip(cases, left, right):
        case.evidence_reduced = evidence_vec.astype(np.float32, copy=False)
        case.context_reduced = context_vec.astype(np.float32, copy=False)


def _relation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not a.size or a.shape != b.shape:
        return np.zeros(0, dtype=np.float32)
    diff = a - b
    cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    euclidean = float(np.linalg.norm(diff))
    return np.concatenate(
        [np.abs(diff), a * b, np.asarray([cosine, euclidean], dtype=np.float32)]
    ).astype(np.float32, copy=False)


def build_theta_pair_feature_vector(a: ThetaCaseFeature, b: ThetaCaseFeature) -> np.ndarray:
    return np.concatenate(
        [
            _relation(a.evidence_reduced, b.evidence_reduced),
            _relation(a.context_reduced, b.context_reduced),
            mgf.build_multigranular_pair_feature_vector(
                a.evidence, b.evidence,
                include_event_order=True,
                include_local_embeddings=False,
            ),
            plf.build_structured_pair_feature_vector(a.base, b.base),
            plf.build_det_scalar_summary_vector(a.base, b.base),
        ]
    ).astype(np.float32, copy=False)


def build_theta_pair_feature_matrix(
    cases: Sequence[ThetaCaseFeature], pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    if not pairs:
        if not cases:
            return np.zeros((0, 53), dtype=np.float32)
        sample = build_theta_pair_feature_vector(cases[0], cases[0])
        return np.zeros((0, len(sample)), dtype=np.float32)
    sample = build_theta_pair_feature_vector(cases[pairs[0][0]], cases[pairs[0][1]])
    output = np.empty((len(pairs), len(sample)), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        output[row] = build_theta_pair_feature_vector(cases[i], cases[j])
    return output


def theta_case_matrix(cases: Sequence[ThetaCaseFeature]) -> np.ndarray:
    """Return normalized case vectors used only for candidate retrieval."""
    rows: list[np.ndarray] = []
    for case in cases:
        det = np.asarray(case.base.det_vec, dtype=np.float32)
        if det.size:
            det = det / max(float(np.linalg.norm(det)), 1e-12)
        rows.append(np.concatenate([case.evidence_reduced, case.context_reduced, det]))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    width = max(len(row) for row in rows)
    matrix = np.zeros((len(rows), width), dtype=np.float32)
    for idx, row in enumerate(rows):
        matrix[idx, : len(row)] = row
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), np.float32(1e-12))
    return matrix


def theta_view_matrices(cases: Sequence[ThetaCaseFeature]) -> dict[str, np.ndarray]:
    """Return separately normalized views for union-of-neighbors retrieval."""
    raw: dict[str, list[np.ndarray]] = {"evidence": [], "context": [], "deterministic": []}
    for case in cases:
        raw["evidence"].append(np.asarray(case.evidence_reduced, dtype=np.float32))
        raw["context"].append(np.asarray(case.context_reduced, dtype=np.float32))
        raw["deterministic"].append(np.asarray(case.base.det_vec, dtype=np.float32))
    output: dict[str, np.ndarray] = {}
    for name, rows in raw.items():
        width = max((len(row) for row in rows), default=0)
        matrix = np.zeros((len(rows), width), dtype=np.float32)
        for idx, row in enumerate(rows):
            matrix[idx, : len(row)] = row
        if width:
            matrix /= np.maximum(
                np.linalg.norm(matrix, axis=1, keepdims=True), np.float32(1e-12)
            )
        output[name] = matrix
    return output


def _add_nearest_neighbors(
    keep: set[tuple[int, int]],
    matrix: np.ndarray,
    neighbors: int,
    block_size: int,
) -> int:
    n = len(matrix)
    if not matrix.size or neighbors <= 0:
        return 0
    before = len(keep)
    take = min(max(1, int(neighbors)), n - 1)
    for start in range(0, n, max(1, int(block_size))):
        stop = min(n, start + max(1, int(block_size)))
        similarity = matrix[start:stop] @ matrix.T
        for local, row in enumerate(similarity):
            i = start + local
            row[i] = -np.inf
            indices = np.argpartition(row, -take)[-take:]
            for j in indices:
                j = int(j)
                if i != j:
                    keep.add((i, j) if i < j else (j, i))
    return len(keep) - before


def _add_prototype_anchor_edges(
    keep: set[tuple[int, int]],
    matrix: np.ndarray,
    reference_k: int,
    anchors_per_cluster: int,
    anchor_cluster_count: int,
    random_state: int,
) -> dict:
    from sklearn.cluster import KMeans

    n = len(matrix)
    k = max(1, min(int(reference_k), n))
    if k <= 1 or not matrix.size:
        return {"prototype_edges": 0, "prototype_clusters": k, "prototype_anchors": 0}
    model = KMeans(
        n_clusters=k, n_init=3, max_iter=100, random_state=int(random_state)
    ).fit(matrix)
    labels = model.labels_.astype(np.int32)
    centers = model.cluster_centers_.astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), np.float32(1e-12))
    anchors: dict[int, list[int]] = {}
    before = len(keep)
    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        if not len(members):
            continue
        scores = matrix[members] @ centers[cluster]
        take = min(max(1, int(anchors_per_cluster)), len(members))
        chosen = members[np.argpartition(scores, -take)[-take:]].tolist()
        anchors[cluster] = [int(value) for value in chosen]
        # Small prototype regions receive dense support without making the
        # global graph dense.
        if len(members) <= 128:
            values = members.tolist()
            for pos, i in enumerate(values):
                for j in values[pos + 1:]:
                    keep.add((int(i), int(j)))
    centroid_similarity = matrix @ centers.T
    cluster_take = min(max(1, int(anchor_cluster_count)), k)
    for node, row in enumerate(centroid_similarity):
        clusters = np.argpartition(row, -cluster_take)[-cluster_take:]
        for cluster in clusters:
            for anchor in anchors.get(int(cluster), []):
                if node != anchor:
                    keep.add((node, anchor) if node < anchor else (anchor, node))
    return {
        "prototype_edges": len(keep) - before,
        "prototype_clusters": k,
        "prototype_anchors": sum(map(len, anchors.values())),
    }


def candidate_pairs(
    cases: Sequence[ThetaCaseFeature],
    top_l: int = 48,
    full_pair_limit: int = 300,
    block_size: int = 512,
    mode: str = "concat",
    reference_k: int | None = None,
    anchors_per_cluster: int = 2,
    anchor_cluster_count: int = 8,
    random_state: int = 0,
) -> tuple[list[tuple[int, int]], dict]:
    n = len(cases)
    if n <= 1:
        return [], {"mode": "empty", "pairs": 0, "all_pairs": 0}
    all_count = n * (n - 1) // 2
    if n <= int(full_pair_limit):
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return pairs, {"mode": "full", "pairs": len(pairs), "all_pairs": all_count}

    keep: set[tuple[int, int]] = set()
    neighbors = min(max(1, int(top_l)), n - 1)
    source_counts: dict[str, int] = {}
    if mode == "concat":
        source_counts["concat_neighbors"] = _add_nearest_neighbors(
            keep, theta_case_matrix(cases), neighbors, block_size
        )
    elif mode == "multiview_anchor":
        for name, matrix in theta_view_matrices(cases).items():
            source_counts[f"{name}_neighbors"] = _add_nearest_neighbors(
                keep, matrix, neighbors, block_size
            )
        if reference_k is not None:
            source_counts.update(_add_prototype_anchor_edges(
                keep,
                theta_case_matrix(cases),
                reference_k,
                anchors_per_cluster,
                anchor_cluster_count,
                random_state,
            ))
    else:
        raise ValueError(f"unknown Theta candidate mode: {mode}")

    for key in ("primary_signature", "op_pair", "mismatch_type", "fatal_file"):
        groups: dict[str, list[int]] = {}
        for idx, case in enumerate(cases):
            value = str(case.base.info.get(key, "")).strip()
            if value:
                groups.setdefault(value, []).append(idx)
        for members in groups.values():
            if len(members) > 128:
                continue
            for pos, i in enumerate(members):
                for j in members[pos + 1:]:
                    keep.add((i, j))

    pairs = sorted(keep)
    return pairs, {
        "mode": mode,
        "pairs": len(pairs),
        "all_pairs": all_count,
        "density": len(pairs) / max(1, all_count),
        "top_l": neighbors,
        **source_counts,
    }


def conflict_values(
    cases: Sequence[ThetaCaseFeature], pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    return np.asarray(
        [mgf.pair_conflict_score(cases[i].evidence, cases[j].evidence) for i, j in pairs],
        dtype=np.float32,
    )
