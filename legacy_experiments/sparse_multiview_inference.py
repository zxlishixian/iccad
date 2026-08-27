#!/usr/bin/env python3
"""Experimental deterministic-first sparse multiview refinement.

The public output receives a validated deterministic baseline before any API
request. Only difficult cases and a few anchors per baseline bucket receive
five-view embeddings. Any expert failure preserves the baseline output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import joblib
import numpy as np

import beta_multiview_inference as bmi
from bounded_evidence import BoundedEvidenceFeature, build_bounded_evidence
import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
from run_selective_multiview_experiments import (
    case_difficulty,
    deterministic_proxy_stack,
    select_expert_cases,
)
from sparse_refinement import (
    centroid_sparse_plan,
    choose_cluster_anchors,
    normalized_det_vectors,
    sparse_refine_from_edges,
    sparse_refine_labels,
)
from selective_multiview_inference import _embed_documents


ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse five-view refinement")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-backend", type=Path,
        default=Path("beta_test_submission_v2/fast/regr_fail_bucketing_fast/regr_fail_bucketing_fast"),
    )
    parser.add_argument(
        "--baseline-cluster", choices=("auto", "agglomerative", "kmeans"),
        default="auto",
    )
    parser.add_argument("--baseline-pred", type=Path, help="Reuse an already validated baseline output instead of rerunning the backend.")
    parser.add_argument("--selection-fraction", type=float, default=0.25)
    parser.add_argument("--max-selected", type=int, default=40)
    parser.add_argument(
        "--selection-mode", choices=("auto", "matrix", "centroid", "evidence"),
        default="auto",
    )
    parser.add_argument(
        "--expert-views", choices=("dual", "five"), default="five",
        help="Embed only features/summary for dual, or all five document views.",
    )
    parser.add_argument("--matrix-selection-max-cases", type=int, default=400)
    parser.add_argument("--evidence-max-bytes", type=int, default=64 * 1024)
    parser.add_argument("--evidence-dim", type=int, default=256)
    parser.add_argument(
        "--active-feature-source", choices=("evidence", "full"), default="evidence",
    )
    parser.add_argument("--anchors-per-cluster", type=int, default=2)
    parser.add_argument("--expert-weight", type=float, default=0.75)
    parser.add_argument("--min-probability", type=float, default=0.55)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--support-probability", type=float, default=0.0)
    parser.add_argument("--max-conflict-ratio", type=float, default=1.0)
    parser.add_argument("--min-det-margin", type=float, default=-1.0)
    parser.add_argument("--require-structured-agreement", action="store_true")
    parser.add_argument("--diagnostics", type=Path)
    return parser.parse_args(argv)


def read_baseline_labels(path: Path, expected_cases: Sequence[str]) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Case", "bucket"]:
            raise ValueError("baseline output header is not Case,bucket")
        rows = list(reader)
    if [str(row["Case"]) for row in rows] != [str(case) for case in expected_cases]:
        raise ValueError("baseline output cases do not match input")
    mapping: dict[str, int] = {}
    return np.asarray([
        mapping.setdefault(str(row["bucket"]), len(mapping)) for row in rows
    ], dtype=np.int32)


def write_labels(path: Path, cases: Sequence[str], labels: Sequence[int]) -> None:
    bmi.write_output(path, cases, labels)


def run_baseline(args: argparse.Namespace, cases: Sequence[str]) -> np.ndarray:
    if args.baseline_pred is not None:
        source = args.baseline_pred.resolve()
        labels = read_baseline_labels(source, cases)
        if source != args.output.resolve():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, args.output.resolve())
        return labels
    backend = args.baseline_backend
    if not backend.is_absolute():
        backend = (ROOT / backend).resolve()
    cluster = args.baseline_cluster
    if cluster == "auto":
        cluster = "agglomerative" if len(cases) <= 300 else "kmeans"
    command = [
        str(backend), "--input", str(args.input.resolve()),
        "--output", str(args.output.resolve()), "--k", str(args.k),
        "--llm-mode", "none", "--cluster", cluster,
    ]
    if cluster == "kmeans":
        command.extend(("--cluster-factor", "1.0"))
    subprocess.run(command, cwd=ROOT, check=True)
    return read_baseline_labels(args.output.resolve(), cases)


def build_active_documents(
    input_csv: Path,
    active_indices: np.ndarray,
    parser: str,
    max_features: int,
    expert_views: str,
) -> dict[str, list[str]]:
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    selected_rows = []
    for idx in active_indices:
        row = dict(rows[int(idx)])
        for field, value in list(row.items()):
            normalized = field.lower().replace("_", " ").strip()
            if not value or ("log" not in normalized and normalized not in {"trace", "sim", "regr"}):
                continue
            value_path = Path(value)
            if not value_path.is_absolute():
                value_path = (input_csv.parent / value_path).resolve()
            row[field] = str(value_path)
        selected_rows.append(row)
    with tempfile.TemporaryDirectory(prefix="iccad_sparse_active_") as temp_name:
        temp_input = Path(temp_name) / "input.csv"
        with temp_input.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_rows)
        feature_docs, summary_docs = bmi.build_feature_documents(temp_input, parser, max_features)
        custom_docs = (
            gm.build_all_view_documents(temp_input.parent)
            if expert_views == "five" else {}
        )
    docs = {
        "features": feature_docs,
        "summary": summary_docs,
    }
    if expert_views == "five":
        docs.update({view: custom_docs[view] for view in ("event", "object", "context")})
    return docs


def build_active_case_features(
    input_csv: Path,
    active_indices: np.ndarray,
    parser: str,
    svd_dim: int,
) -> list[plf.LLMCaseFeature]:
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    selected_rows = []
    for idx in active_indices:
        row = dict(rows[int(idx)])
        for field, value in list(row.items()):
            normalized = field.lower().replace("_", " ").strip()
            if not value or ("log" not in normalized and normalized not in {"trace", "sim", "regr"}):
                continue
            value_path = Path(value)
            if not value_path.is_absolute():
                value_path = (input_csv.parent / value_path).resolve()
            row[field] = str(value_path)
        selected_rows.append(row)
    with tempfile.TemporaryDirectory(prefix="iccad_sparse_features_") as temp_name:
        temp_input = Path(temp_name) / "input.csv"
        with temp_input.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_rows)
        features, _ = plf.build_llm_case_features_for_inputs(
            [temp_input], parser=parser, svd_dim=svd_dim,
            llm_args=None, log_llm_disabled=False,
        )
    return features


def build_evidence_case_features(
    evidence: Sequence[BoundedEvidenceFeature],
    active_indices: np.ndarray,
) -> list[plf.LLMCaseFeature]:
    output = []
    for idx in active_indices:
        item = evidence[int(idx)]
        output.append(plf.LLMCaseFeature(
            case_id=item.case_id,
            det_vec=item.vector.astype(np.float32, copy=False),
            llm_vec=np.zeros(0, dtype=np.float32),
            llm_vec_reduced=None,
            llm_summary_vec=np.zeros(0, dtype=np.float32),
            llm_summary_vec_reduced=None,
            trace_vec=np.zeros(0, dtype=np.float32),
            trace_vec_reduced=None,
            tokens=list(item.tokens),
            token_set=set(item.tokens),
            primary_tokens=set(item.primary_tokens),
            sim_tokens=set(item.sim_tokens),
            regr_tokens=set(item.regr_tokens),
            info=dict(item.info),
            trace_structured=None,
            completion_feature=None,
        ))
    return output


def build_active_pairs(
    selected: np.ndarray,
    anchors: dict[int, list[int]],
    global_to_local: dict[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    local_pairs: list[tuple[int, int]] = []
    global_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    anchor_indices = sorted({idx for values in anchors.values() for idx in values})
    for case_idx in np.flatnonzero(selected):
        for anchor_idx in anchor_indices:
            if int(case_idx) == anchor_idx:
                continue
            edge = (min(int(case_idx), anchor_idx), max(int(case_idx), anchor_idx))
            if edge in seen:
                continue
            seen.add(edge)
            global_pairs.append((int(case_idx), anchor_idx))
            local_pairs.append((global_to_local[int(case_idx)], global_to_local[anchor_idx]))
    return local_pairs, global_pairs


def expert_probabilities(
    subset_features: list[plf.LLMCaseFeature],
    raw_views: dict[str, np.ndarray],
    local_pairs: Sequence[tuple[int, int]],
    model_dir: Path,
    manifest: dict,
    expert_views: str,
) -> np.ndarray:
    seeds = [int(seed) for seed in manifest["seeds"]]
    view_dim = int(manifest["view_dim"])
    branch_weight = float(manifest["primary_beta"])
    seed_values = []
    for seed in seeds:
        preproc = joblib.load(model_dir / f"preprocess_seed{seed}.pkl")
        reduced = {
            "features": plf._apply_reducer_to_matrix(raw_views["features"], preproc["feature_reducer"], view_dim).astype(np.float32),
            "summary": plf._apply_reducer_to_matrix(raw_views["summary"], preproc["summary_reducer"], view_dim).astype(np.float32),
        }
        if expert_views == "five":
            reduced.update({
                view: plf._apply_reducer_to_matrix(
                    raw_views[view], preproc["custom_reducers"][view], view_dim
                ).astype(np.float32)
                for view in ("event", "object", "context")
            })
        for idx, feature in enumerate(subset_features):
            feature.llm_vec_reduced = reduced["features"][idx]
            feature.llm_summary_vec_reduced = reduced["summary"][idx]
        dual_x = gm.build_multiview_pair_feature_matrix(
            subset_features, {}, ["features", "summary"], local_pairs
        )
        dual_pkg = joblib.load(model_dir / f"model_dual_seed{seed}.pkl")
        dual_prob = gm.predict_view_probabilities_flat(dual_pkg, dual_x)
        if expert_views == "dual":
            seed_values.append(dual_prob)
            continue
        five_x = gm.build_multiview_pair_feature_matrix(
            subset_features,
            {view: reduced[view] for view in ("event", "object", "context")},
            ["features", "summary", "event", "object", "context"],
            local_pairs,
        )
        five_pkg = joblib.load(model_dir / f"model_quad_event_object_context_seed{seed}.pkl")
        five_prob = gm.predict_view_probabilities_flat(five_pkg, five_x)
        seed_values.append((1.0 - branch_weight) * dual_prob + branch_weight * five_prob)
    return np.mean(np.stack(seed_values), axis=0).astype(np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    bmi.install_log_sample_cache()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    model_dir = args.model_dir.resolve()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    cases = [str(case) for case in osf.read_cases(args.input)]
    if len(cases) <= 1:
        write_labels(args.output, cases, [0] * len(cases))
        return 0

    try:
        base_labels = run_baseline(args, cases)
    except Exception as exc:
        print(f"[sparse-expert] baseline failed: {exc}", file=sys.stderr)
        write_labels(args.output, cases, range(len(cases)))
        return 0
    baseline_at = time.perf_counter()

    diagnostics: dict[str, object] = {
        "cases": len(cases), "k": int(args.k), "selected": "baseline",
        "timing": {"baseline": baseline_at - started},
    }
    try:
        selection_mode = args.selection_mode
        if selection_mode == "auto":
            selection_mode = (
                "matrix" if len(cases) <= args.matrix_selection_max_cases else "evidence"
            )
        features: list[plf.LLMCaseFeature] | None = None
        evidence: list[BoundedEvidenceFeature] | None = None
        structured_infos: list[dict] | None = None
        selection_stats: dict[str, object] = {}
        det_similarity = None
        if selection_mode == "evidence":
            evidence = build_bounded_evidence(
                args.input,
                max_bytes=args.evidence_max_bytes,
                dim=args.evidence_dim,
            )
            if len(evidence) != len(cases):
                raise RuntimeError(
                    f"bounded evidence count mismatch: {len(evidence)} != {len(cases)}"
                )
            det_vectors = np.vstack([item.vector for item in evidence]).astype(np.float32)
            structured_infos = [item.info for item in evidence]
            selected, anchors, centroid_stats = centroid_sparse_plan(
                base_labels, det_vectors, args.selection_fraction,
                args.max_selected, args.anchors_per_cluster,
            )
            selection_stats.update(centroid_stats)
            selection_stats["sim_status"] = dict(sorted({
                status: sum(item.sim_status == status for item in evidence)
                for status in {item.sim_status for item in evidence}
            }.items()))
            selection_stats["regr_status"] = dict(sorted({
                status: sum(item.regr_status == status for item in evidence)
                for status in {item.regr_status for item in evidence}
            }.items()))
        else:
            features, _ = plf.build_llm_case_features_for_inputs(
                [args.input], parser=str(manifest["parser"]),
                svd_dim=int(manifest["svd_dim"]),
                llm_args=None, log_llm_disabled=False,
            )
            det_vectors = normalized_det_vectors(features)
            if selection_mode == "matrix":
                det_stack = deterministic_proxy_stack(features)
                difficulty, _components, det_similarity = case_difficulty(
                    det_stack, int(args.k), str(manifest["source_clusterer"])
                )
                selected = select_expert_cases(
                    difficulty, args.selection_fraction, 2, args.max_selected
                )
                anchors = choose_cluster_anchors(
                    base_labels, det_similarity, selected, args.anchors_per_cluster
                )
            else:
                selected, anchors, selection_stats = centroid_sparse_plan(
                    base_labels, det_vectors, args.selection_fraction,
                    args.max_selected, args.anchors_per_cluster,
                )
        active_indices = np.asarray(sorted(
            set(np.flatnonzero(selected).tolist())
            | {idx for values in anchors.values() for idx in values}
        ), dtype=np.int64)
        global_to_local = {int(value): idx for idx, value in enumerate(active_indices)}
        local_pairs, global_pairs = build_active_pairs(selected, anchors, global_to_local)
        if not local_pairs:
            raise RuntimeError("no sparse expert pairs were selected")

        docs = build_active_documents(
            args.input, active_indices, str(manifest["parser"]), 80,
            args.expert_views,
        )
        runtime_args = SimpleNamespace(
            llm_doc_max_features=80,
            llm_cache_dir=Path(os.environ.get("BETA_LLM_CACHE_DIR", "/tmp/iccad_sparse_multiview_cache")),
            llm_batch_size=int(os.environ.get("BETA_LLM_BATCH_SIZE", "128")),
            llm_timeout_sec=float(os.environ.get("BETA_LLM_TIMEOUT_SEC", "120")),
            svd_dim=int(manifest["svd_dim"]),
        )
        raw_views, model_name, docs_total, docs_unique = _embed_documents(
            docs, runtime_args, bool(manifest.get("canonicalized_docs", False))
        )
        expected_dim = int(manifest["embedding_dim"])
        if any(matrix.shape != (len(active_indices), expected_dim) for matrix in raw_views.values()):
            raise RuntimeError(f"embedding fallback/dimension mismatch: {[m.shape for m in raw_views.values()]}")
        embedded_at = time.perf_counter()

        if features is not None:
            subset_features = [features[idx] for idx in active_indices]
        elif evidence is not None and args.active_feature_source == "evidence":
            subset_features = build_evidence_case_features(evidence, active_indices)
        else:
            subset_features = build_active_case_features(
                args.input, active_indices, str(manifest["parser"]),
                int(manifest["svd_dim"]),
            )
        flat_prob = expert_probabilities(
            subset_features, raw_views, local_pairs, model_dir, manifest,
            args.expert_views,
        )
        if selection_mode == "matrix":
            expert_matrix = np.eye(len(cases), dtype=np.float32)
            for (left, right), value in zip(global_pairs, flat_prob):
                expert_matrix[left, right] = expert_matrix[right, left] = float(value)
            labels, stats = sparse_refine_labels(
                base_labels, det_similarity, expert_matrix, selected,
                args.anchors_per_cluster, args.expert_weight,
                args.min_probability, args.margin,
            )
        else:
            edge_values = {
                (min(left, right), max(left, right)): float(value)
                for (left, right), value in zip(global_pairs, flat_prob)
            }
            labels, stats = sparse_refine_from_edges(
                base_labels, det_vectors, edge_values, selected, anchors,
                args.expert_weight, args.min_probability, args.margin,
                structured_infos=structured_infos,
                min_support=args.min_support,
                support_probability=args.support_probability,
                max_conflict_ratio=args.max_conflict_ratio,
                min_det_margin=args.min_det_margin,
                require_structured_agreement=args.require_structured_agreement,
            )
        write_labels(args.output, cases, labels)
        finished = time.perf_counter()
        diagnostics.update({
            "selected": "sparse_expert",
            "selected_cases": int(np.sum(selected)),
            "active_cases": int(len(active_indices)),
            "selection_mode": selection_mode,
            "expert_views": args.expert_views,
            "selection_stats": selection_stats,
            "embedding_docs": docs_total,
            "embedding_unique_docs": docs_unique,
            "embedding_model": model_name,
            **stats,
            "timing": {
                "baseline": baseline_at - started,
                "selection_and_docs": embedded_at - baseline_at,
                "model_and_refinement": finished - embedded_at,
                "total": finished - started,
            },
        })
    except Exception as exc:
        diagnostics["expert_error"] = str(exc)
        diagnostics["timing"]["total"] = time.perf_counter() - started
        print(f"[sparse-expert] expert failed; baseline preserved: {exc}", file=sys.stderr)

    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[sparse-expert] selected={diagnostics['selected']} cases={len(cases)} "
        f"total={diagnostics['timing']['total']:.3f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
