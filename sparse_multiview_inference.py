#!/usr/bin/env python3
"""Experimental deterministic-first sparse five-view refinement.

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
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import joblib
import numpy as np

import beta_multiview_inference as bmi
import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
from run_selective_multiview_experiments import (
    case_difficulty,
    deterministic_proxy_stack,
    select_expert_cases,
)
from run_sparse_anchor_refinement_experiments import (
    choose_cluster_anchors,
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
    parser.add_argument("--baseline-pred", type=Path, help="Reuse an already validated baseline output instead of rerunning the backend.")
    parser.add_argument("--selection-fraction", type=float, default=0.25)
    parser.add_argument("--max-selected", type=int, default=40)
    parser.add_argument("--anchors-per-cluster", type=int, default=2)
    parser.add_argument("--expert-weight", type=float, default=0.75)
    parser.add_argument("--min-probability", type=float, default=0.55)
    parser.add_argument("--margin", type=float, default=0.10)
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
    command = [
        str(backend), "--input", str(args.input.resolve()),
        "--output", str(args.output.resolve()), "--k", str(args.k),
        "--llm-mode", "none", "--cluster", "agglomerative",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return read_baseline_labels(args.output.resolve(), cases)


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
            **{
                view: plf._apply_reducer_to_matrix(
                    raw_views[view], preproc["custom_reducers"][view], view_dim
                ).astype(np.float32)
                for view in ("event", "object", "context")
            },
        }
        for idx, feature in enumerate(subset_features):
            feature.llm_vec_reduced = reduced["features"][idx]
            feature.llm_summary_vec_reduced = reduced["summary"][idx]
        dual_x = gm.build_multiview_pair_feature_matrix(
            subset_features, {}, ["features", "summary"], local_pairs
        )
        five_x = gm.build_multiview_pair_feature_matrix(
            subset_features,
            {view: reduced[view] for view in ("event", "object", "context")},
            ["features", "summary", "event", "object", "context"],
            local_pairs,
        )
        dual_pkg = joblib.load(model_dir / f"model_dual_seed{seed}.pkl")
        five_pkg = joblib.load(model_dir / f"model_quad_event_object_context_seed{seed}.pkl")
        dual_prob = gm.predict_view_probabilities_flat(dual_pkg, dual_x)
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
        features, _ = plf.build_llm_case_features_for_inputs(
            [args.input], parser=str(manifest["parser"]), svd_dim=int(manifest["svd_dim"]),
            llm_args=None, log_llm_disabled=False,
        )
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
        active_indices = np.asarray(sorted(
            set(np.flatnonzero(selected).tolist())
            | {idx for values in anchors.values() for idx in values}
        ), dtype=np.int64)
        global_to_local = {int(value): idx for idx, value in enumerate(active_indices)}
        local_pairs, global_pairs = build_active_pairs(selected, anchors, global_to_local)
        if not local_pairs:
            raise RuntimeError("no sparse expert pairs were selected")

        feature_docs, summary_docs = bmi.build_feature_documents(
            args.input, str(manifest["parser"]), 80
        )
        custom_docs = gm.build_all_view_documents(args.input.parent)
        docs = {
            "features": [feature_docs[idx] for idx in active_indices],
            "summary": [summary_docs[idx] for idx in active_indices],
            **{
                view: [custom_docs[view][idx] for idx in active_indices]
                for view in ("event", "object", "context")
            },
        }
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

        subset_features = [features[idx] for idx in active_indices]
        flat_prob = expert_probabilities(
            subset_features, raw_views, local_pairs, model_dir, manifest
        )
        expert_matrix = np.eye(len(cases), dtype=np.float32)
        for (left, right), value in zip(global_pairs, flat_prob):
            expert_matrix[left, right] = expert_matrix[right, left] = float(value)
        labels, stats = sparse_refine_labels(
            base_labels, det_similarity, expert_matrix, selected,
            args.anchors_per_cluster, args.expert_weight,
            args.min_probability, args.margin,
        )
        write_labels(args.output, cases, labels)
        finished = time.perf_counter()
        diagnostics.update({
            "selected": "sparse_expert",
            "selected_cases": int(np.sum(selected)),
            "active_cases": int(len(active_indices)),
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
