#!/usr/bin/env python3
"""Experimental five-seed multi-view inference for the Beta submission.

Runtime inputs are limited to input.csv plus referenced sim/regr logs and the
organizer-provided embedding endpoint. Gold, golden, meta, and trace files are
never discovered or opened by this entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import joblib
import numpy as np

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
from run_graph_multiview_experiments import (
    build_all_view_documents,
    build_multiview_pair_feature_matrix,
    fetch_view_embeddings,
    make_embedding_args,
    predict_view_probabilities,
    views_for_config,
)


def runtime_root() -> Path:
    override = os.environ.get("BETA_MULTIVIEW_MODEL_DIR", "").strip()
    if override:
        return Path(override).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "models_multiview"
    return Path(__file__).resolve().parent / "beta_multiview_models"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental Beta multi-view bucketing")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    return parser.parse_args(argv)


def write_output(path: Path, cases: Sequence[str], labels: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Case", "bucket"])
        writer.writerows((case, f"bucket_{int(label):03d}") for case, label in zip(cases, labels))


def validate_embeddings(features, raw_custom: dict[str, np.ndarray], expected_dim: int) -> None:
    if not features:
        return
    if features[0].llm_vec.size != expected_dim or features[0].llm_summary_vec.size != expected_dim:
        raise RuntimeError(
            f"embedding fallback detected: features={features[0].llm_vec.size} "
            f"summary={features[0].llm_summary_vec.size} expected={expected_dim}"
        )
    for view, matrix in raw_custom.items():
        if matrix.shape != (len(features), expected_dim):
            raise RuntimeError(f"embedding fallback detected for {view}: shape={matrix.shape}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = runtime_root()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in manifest["seeds"]]
    view_dim = int(manifest["view_dim"])
    expected_dim = int(manifest["embedding_dim"])
    beta = float(manifest["primary_beta"])
    consensus_weight = float(manifest["consensus_weight"])

    input_csv = args.input.resolve()
    cases = osf.read_cases(input_csv)
    n = len(cases)
    if n == 0:
        write_output(args.output, [], [])
        return 0
    if n == 1:
        write_output(args.output, cases, [0])
        return 0

    runtime_args = SimpleNamespace(
        llm_doc_max_features=80,
        llm_cache_dir=Path(os.environ.get("BETA_LLM_CACHE_DIR", "/tmp/iccad_beta_multiview_cache")),
        llm_batch_size=int(os.environ.get("BETA_LLM_BATCH_SIZE", "64")),
        llm_timeout_sec=float(os.environ.get("BETA_LLM_TIMEOUT_SEC", "120")),
        svd_dim=int(manifest["svd_dim"]),
    )
    features, _ = plf.build_llm_case_features_for_inputs(
        [input_csv],
        parser=str(manifest["parser"]),
        svd_dim=int(manifest["svd_dim"]),
        llm_args=make_embedding_args(runtime_args),
    )
    dataset_docs = build_all_view_documents(input_csv.parent)
    raw_custom = {
        view: fetch_view_embeddings(dataset_docs[view], runtime_args, view)
        for view in ("event", "object", "context")
    }
    validate_embeddings(features, raw_custom, expected_dim)
    pairs = osf.all_pairs(n)

    seed_probs: list[np.ndarray] = []
    for seed in seeds:
        preproc = joblib.load(model_dir / f"preprocess_seed{seed}.pkl")
        plf.apply_llm_reducer(features, preproc["feature_reducer"], view_dim)
        plf.apply_llm_summary_reducer(features, preproc["summary_reducer"], view_dim)
        reduced_custom = {
            view: plf._apply_reducer_to_matrix(raw, preproc["custom_reducers"][view], view_dim).astype(np.float32)
            for view, raw in raw_custom.items()
        }
        branch_probs = {}
        for view_config in ("dual", "quad_event_object_context"):
            X = build_multiview_pair_feature_matrix(
                features, reduced_custom, views_for_config(view_config), pairs
            )
            model_pkg = joblib.load(model_dir / f"model_{view_config}_seed{seed}.pkl")
            branch_probs[view_config] = predict_view_probabilities(model_pkg, X, pairs, n)
        prob = (
            (1.0 - beta) * branch_probs["dual"]
            + beta * branch_probs["quad_event_object_context"]
        ).astype(np.float32)
        prob = (prob + prob.T) * 0.5
        np.fill_diagonal(prob, 1.0)
        seed_probs.append(prob)

    mean_prob = np.mean(np.stack(seed_probs, axis=0), axis=0).astype(np.float32)
    coassoc = np.zeros_like(mean_prob)
    source_clusterer = str(manifest["source_clusterer"])
    k = max(1, min(int(args.k), n))
    for prob in seed_probs:
        seed_labels = np.asarray(gc.cluster_probability_graph(prob, k, source_clusterer).labels)
        coassoc += (seed_labels[:, None] == seed_labels[None, :]).astype(np.float32)
    coassoc /= float(len(seed_probs))
    final_prob = ((1.0 - consensus_weight) * mean_prob + consensus_weight * coassoc).astype(np.float32)
    final_prob = (final_prob + final_prob.T) * 0.5
    np.fill_diagonal(final_prob, 1.0)
    labels = gc.cluster_probability_graph(final_prob, k, str(manifest["final_clusterer"])).labels
    write_output(args.output, cases, labels)
    print(
        f"[beta-multiview] cases={n} seeds={len(seeds)} embedding_dim={expected_dim} "
        f"clusters={len(set(labels))}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[beta-multiview] failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
