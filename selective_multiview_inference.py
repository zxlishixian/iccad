#!/usr/bin/env python3
"""Experimental dual-first selective five-view inference.

The base stage embeds features/summary for every case. It selects difficult
cases without labels, embeds event/object/context only for the selected set,
and applies the five-view model only to pairs whose endpoints are selected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import joblib
import numpy as np

import beta_multiview_inference as bmi
import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
import run_graph_multiview_experiments as gm
from run_selective_multiview_experiments import (
    case_difficulty,
    deterministic_proxy_stack,
    expand_with_neighbors,
    select_expert_cases,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental selective five-view inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--selector", choices=["dual", "deterministic"], default="dual")
    parser.add_argument("--selection-fraction", type=float, default=0.50)
    parser.add_argument("--min-selected", type=int, default=2)
    parser.add_argument("--max-selected", type=int, default=20)
    parser.add_argument("--neighbors-per-case", type=int, default=2)
    parser.add_argument("--max-expanded-selected", type=int, default=40)
    parser.add_argument("--diagnostics", type=Path)
    return parser.parse_args(argv)


def _embed_documents(
    docs_by_view: dict[str, list[str]],
    runtime_args: SimpleNamespace,
    canonicalize: bool,
) -> tuple[dict[str, np.ndarray], str, int, int]:
    ordered_names = list(docs_by_view)
    canonical = {
        name: [bmi.canonicalize_document(name, doc) if canonicalize else doc for doc in docs]
        for name, docs in docs_by_view.items()
    }
    counts = {name: len(canonical[name]) for name in ordered_names}
    combined = [doc for name in ordered_names for doc in canonical[name]]
    if not combined:
        return {name: np.zeros((0, 0), dtype=np.float32) for name in ordered_names}, "none", 0, 0
    unique = list(dict.fromkeys(combined))
    unique_index = {doc: idx for idx, doc in enumerate(unique)}
    inverse = np.fromiter((unique_index[doc] for doc in combined), dtype=np.int64, count=len(combined))
    llm_args = gm.make_embedding_args(runtime_args)
    llm_args.llm_dual = False
    embeddings, model_name = rfb.fetch_llm_embeddings(unique, llm_args)
    unique_matrix = np.asarray(embeddings, dtype=np.float32)
    if unique_matrix.ndim != 2 or unique_matrix.shape[0] != len(unique):
        raise RuntimeError(f"unexpected embedding shape: {unique_matrix.shape}")
    matrix = unique_matrix[inverse]
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), np.float32(1e-12))
    views = {}
    offset = 0
    for name in ordered_names:
        count = counts[name]
        views[name] = matrix[offset : offset + count]
        offset += count
    return views, model_name, len(combined), len(unique)


def _dual_seed_probabilities(
    features: list[plf.LLMCaseFeature],
    pairs: Sequence[tuple[int, int]],
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    pair_tail: np.ndarray,
    model_dir: Path,
    seeds: Sequence[int],
    view_dim: int,
) -> list[np.ndarray]:
    n = len(features)
    probabilities = []
    for seed in seeds:
        preproc = joblib.load(model_dir / f"preprocess_seed{seed}.pkl")
        plf.apply_llm_reducer(features, preproc["feature_reducer"], view_dim)
        plf.apply_llm_summary_reducer(features, preproc["summary_reducer"], view_dim)
        feature_matrix = np.vstack([feature.effective_llm_vec for feature in features]).astype(np.float32)
        summary_matrix = np.vstack([feature.effective_llm_summary_vec for feature in features]).astype(np.float32)
        X = np.concatenate((
            bmi.relation_matrix(feature_matrix, pair_left, pair_right),
            bmi.relation_matrix(summary_matrix, pair_left, pair_right),
            pair_tail,
        ), axis=1)
        model_pkg = joblib.load(model_dir / f"model_dual_seed{seed}.pkl")
        probabilities.append(gm.predict_view_probabilities(model_pkg, X, pairs, n))
    return probabilities


def _selective_seed_probabilities(
    features: list[plf.LLMCaseFeature],
    selected_pairs: Sequence[tuple[int, int]],
    raw_selected: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    dual_probs: Sequence[np.ndarray],
    model_dir: Path,
    seeds: Sequence[int],
    view_dim: int,
    beta: float,
) -> list[np.ndarray]:
    n = len(features)
    output = []
    for seed, base_prob in zip(seeds, dual_probs):
        preproc = joblib.load(model_dir / f"preprocess_seed{seed}.pkl")
        plf.apply_llm_reducer(features, preproc["feature_reducer"], view_dim)
        plf.apply_llm_summary_reducer(features, preproc["summary_reducer"], view_dim)
        reduced_custom = {}
        for view in ("event", "object", "context"):
            selected_matrix = plf._apply_reducer_to_matrix(
                raw_selected[view], preproc["custom_reducers"][view], view_dim
            ).astype(np.float32)
            full_matrix = np.zeros((n, view_dim), dtype=np.float32)
            full_matrix[selected_indices] = selected_matrix
            reduced_custom[view] = full_matrix
        X = gm.build_multiview_pair_feature_matrix(
            features,
            reduced_custom,
            ["features", "summary", "event", "object", "context"],
            selected_pairs,
        )
        model_pkg = joblib.load(model_dir / f"model_quad_event_object_context_seed{seed}.pkl")
        expert_flat = gm.predict_view_probabilities_flat(model_pkg, X)
        prob = base_prob.copy()
        for (i, j), expert_value in zip(selected_pairs, expert_flat):
            value = (1.0 - beta) * float(base_prob[i, j]) + beta * float(expert_value)
            prob[i, j] = prob[j, i] = value
        output.append(prob)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    bmi.install_log_sample_cache()
    model_dir = args.model_dir.resolve()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in manifest["seeds"]]
    view_dim = int(manifest["view_dim"])
    expected_dim = int(manifest["embedding_dim"])
    beta = float(manifest["primary_beta"])
    parser = str(manifest["parser"])
    canonicalize = bool(manifest.get("canonicalized_docs", False))

    input_csv = args.input.resolve()
    cases = osf.read_cases(input_csv)
    n = len(cases)
    if n <= 1:
        bmi.write_output(args.output, cases, [0] * n)
        return 0
    k = max(1, min(int(args.k), n))
    runtime_args = SimpleNamespace(
        llm_doc_max_features=80,
        llm_cache_dir=Path(os.environ.get("BETA_LLM_CACHE_DIR", "/tmp/iccad_selective_cache")),
        llm_batch_size=int(os.environ.get("BETA_LLM_BATCH_SIZE", "128")),
        llm_timeout_sec=float(os.environ.get("BETA_LLM_TIMEOUT_SEC", "120")),
        svd_dim=int(manifest["svd_dim"]),
    )

    features, _ = plf.build_llm_case_features_for_inputs(
        [input_csv], parser=parser, svd_dim=int(manifest["svd_dim"]),
        llm_args=None, log_llm_disabled=False,
    )
    parsed_at = time.perf_counter()
    feature_docs, summary_docs = bmi.build_feature_documents(
        input_csv, parser, int(runtime_args.llm_doc_max_features)
    )
    selected = np.zeros(n, dtype=bool)
    selected_indices = np.zeros(0, dtype=np.int64)
    selected_pairs: list[tuple[int, int]] = []
    raw_selected: dict[str, np.ndarray] = {}
    difficulty = np.zeros(n, dtype=np.float32)
    components: dict[str, np.ndarray] = {}
    embedding_requests = 1
    docs_by_view = {"features": feature_docs, "summary": summary_docs}
    if args.selector == "deterministic" and args.selection_fraction > 0.0:
        proxy_stack = deterministic_proxy_stack(features)
        difficulty, components, proxy_similarity = case_difficulty(
            proxy_stack, k, str(manifest["source_clusterer"])
        )
        selected = select_expert_cases(
            difficulty, args.selection_fraction, args.min_selected, args.max_selected
        )
        selected = expand_with_neighbors(
            selected, proxy_similarity, args.neighbors_per_case, args.max_expanded_selected
        )
        selected_indices = np.flatnonzero(selected)
        custom_docs = gm.build_all_view_documents(input_csv.parent)
        docs_by_view.update({
            view: [custom_docs[view][idx] for idx in selected_indices]
            for view in ("event", "object", "context")
        })

    embedded_views, model_name, embedded_docs, embedded_unique = _embed_documents(
        docs_by_view, runtime_args, canonicalize
    )
    dual_views = {view: embedded_views[view] for view in ("features", "summary")}
    if args.selector == "deterministic" and len(selected_indices):
        raw_selected = {
            view: embedded_views[view] for view in ("event", "object", "context")
        }
    if any(matrix.shape != (n, expected_dim) for matrix in dual_views.values()):
        raise RuntimeError(f"dual embedding fallback detected: {[matrix.shape for matrix in dual_views.values()]}")
    for idx, feature in enumerate(features):
        feature.llm_vec = dual_views["features"][idx]
        feature.llm_summary_vec = dual_views["summary"][idx]
    dual_embedded_at = time.perf_counter()

    pairs = osf.all_pairs(n)
    pair_left = np.fromiter((i for i, _ in pairs), dtype=np.int64, count=len(pairs))
    pair_right = np.fromiter((j for _, j in pairs), dtype=np.int64, count=len(pairs))
    pair_tail = bmi.build_pair_tail(features, pairs)
    dual_probs = _dual_seed_probabilities(
        features, pairs, pair_left, pair_right, pair_tail, model_dir, seeds, view_dim
    )
    dual_predicted_at = time.perf_counter()

    if args.selector == "dual":
        difficulty, components, mean_dual = case_difficulty(
            np.stack(dual_probs), k, str(manifest["source_clusterer"])
        )
        selected = select_expert_cases(
            difficulty, args.selection_fraction, args.min_selected, args.max_selected
        )
        selected = expand_with_neighbors(
            selected, mean_dual, args.neighbors_per_case, args.max_expanded_selected
        )
        selected_indices = np.flatnonzero(selected)
    selected_pairs = [
        (int(selected_indices[i]), int(selected_indices[j]))
        for i in range(len(selected_indices))
        for j in range(i + 1, len(selected_indices))
    ]

    dual_docs = n * 2
    extra_docs = len(selected_indices) * 3
    extra_unique = 0
    selective_probs = dual_probs
    if selected_pairs and args.selector == "dual":
        custom_docs = gm.build_all_view_documents(input_csv.parent)
        selected_docs = {
            view: [custom_docs[view][idx] for idx in selected_indices]
            for view in ("event", "object", "context")
        }
        raw_selected, extra_model, extra_docs, extra_unique = _embed_documents(
            selected_docs, runtime_args, canonicalize
        )
        embedding_requests += 1
        if extra_model != model_name:
            raise RuntimeError(f"embedding model changed between stages: {model_name} vs {extra_model}")
        if any(matrix.shape != (len(selected_indices), expected_dim) for matrix in raw_selected.values()):
            raise RuntimeError(f"extra embedding fallback detected: {[matrix.shape for matrix in raw_selected.values()]}")
    if selected_pairs:
        selective_probs = _selective_seed_probabilities(
            features, selected_pairs, raw_selected, selected_indices, dual_probs,
            model_dir, seeds, view_dim, beta,
        )
    expert_at = time.perf_counter()

    mean_prob = np.mean(np.stack(selective_probs), axis=0).astype(np.float32)
    consensus_weight = float(manifest["consensus_weight"])
    if consensus_weight > 0.0:
        coassoc = np.zeros_like(mean_prob)
        for prob in selective_probs:
            labels = np.asarray(
                gc.cluster_probability_graph(prob, k, str(manifest["source_clusterer"])).labels
            )
            coassoc += (labels[:, None] == labels[None, :]).astype(np.float32)
        coassoc /= float(len(selective_probs))
        mean_prob = (1.0 - consensus_weight) * mean_prob + consensus_weight * coassoc
    mean_prob = (mean_prob + mean_prob.T) * 0.5
    np.fill_diagonal(mean_prob, 1.0)
    labels = gc.cluster_probability_graph(
        mean_prob, k, str(manifest["final_clusterer"])
    ).labels
    bmi.write_output(args.output, cases, labels)
    finished = time.perf_counter()

    diagnostics = {
        "cases": n,
        "k": k,
        "seeds": len(seeds),
        "embedding_model": model_name,
        "embedding_dim": expected_dim,
        "selector": args.selector,
        "embedding_requests": embedding_requests,
        "selection_fraction_requested": args.selection_fraction,
        "selected_cases": int(len(selected_indices)),
        "selected_fraction": float(len(selected_indices) / n),
        "selected_pairs": len(selected_pairs),
        "selected_pair_fraction": len(selected_pairs) / max(1, len(pairs)),
        "dual_docs": dual_docs,
        "embedded_docs_total": embedded_docs + (extra_docs if args.selector == "dual" else 0),
        "embedded_unique_docs_total": embedded_unique + extra_unique,
        "dual_unique_docs": embedded_unique if args.selector == "dual" else None,
        "extra_docs": extra_docs,
        "extra_unique_docs": extra_unique,
        "timing": {
            "parse": parsed_at - started,
            "dual_embedding": dual_embedded_at - parsed_at,
            "dual_model_and_difficulty": dual_predicted_at - dual_embedded_at,
            "selection_and_expert": expert_at - dual_predicted_at,
            "final_cluster_and_output": finished - expert_at,
            "total": finished - started,
        },
        "difficulty": {
            "min": float(np.min(difficulty)),
            "mean": float(np.mean(difficulty)),
            "max": float(np.max(difficulty)),
            **{f"mean_{name}": float(np.mean(values)) for name, values in components.items()},
        },
    }
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[selective-runtime] cases={n} selector={args.selector} requests={embedding_requests} "
        f"selected={len(selected_indices)} extra_docs={extra_docs} "
        f"embedding_dim={expected_dim} clusters={len(set(labels))} total={finished-started:.3f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[selective-runtime] failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
