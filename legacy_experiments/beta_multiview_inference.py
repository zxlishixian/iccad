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
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import joblib
import numpy as np

import graph_clustering as gc
import official_style_features as osf
import pairwise_features as pf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
from run_graph_multiview_experiments import (
    build_all_view_documents,
    make_embedding_args,
    predict_view_probabilities,
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



def install_log_sample_cache() -> None:
    original = rfb.read_log_sample

    @lru_cache(maxsize=None)
    def cached(path_text: str) -> tuple[str, str]:
        return original(Path(path_text))

    def wrapper(path: Path | None) -> tuple[str, str]:
        if path is None:
            return original(None)
        return cached(str(Path(path).resolve()))

    rfb.read_log_sample = wrapper


def canonicalize_document(view: str, doc: str) -> str:
    lines = doc.splitlines()
    if not lines:
        return doc
    if view == "features" and lines[0].startswith("case_index:"):
        lines[0] = "case_index:"
    elif view == "summary" and lines[0].startswith("Case "):
        lines[0] = "Regression failure summary."
    return "\n".join(lines)


def build_feature_documents(input_csv: Path, parser: str, max_features: int) -> tuple[list[str], list[str]]:
    _case_ids, base_features, normalized_lines, _infos = pf._collect_case_inputs(input_csv, parser)
    parser_args = pf._make_parser_args(parser)
    counters, _template_count = rfb.build_feature_counters(
        parser_args,
        base_features,
        normalized_lines,
        token_weights=None,
        token_weight_mode="none",
    )
    return (
        rfb.build_llm_case_documents(counters, max_features=max_features, doc_style="features"),
        rfb.build_llm_case_documents(counters, max_features=max_features, doc_style="summary"),
    )


def fetch_all_view_embeddings(
    input_csv: Path,
    features: list[plf.LLMCaseFeature],
    runtime_args: SimpleNamespace,
    parser: str,
    canonicalize: bool,
) -> dict[str, np.ndarray]:
    feature_docs, summary_docs = build_feature_documents(
        input_csv, parser, int(runtime_args.llm_doc_max_features)
    )
    custom_docs = build_all_view_documents(input_csv.parent)
    ordered = {
        "features": feature_docs,
        "summary": summary_docs,
        "event": custom_docs["event"],
        "object": custom_docs["object"],
        "context": custom_docs["context"],
    }
    canonical = {
        name: [canonicalize_document(name, doc) if canonicalize else doc for doc in docs]
        for name, docs in ordered.items()
    }
    counts = {name: len(docs) for name, docs in canonical.items()}
    combined_docs = [doc for docs in canonical.values() for doc in docs]
    unique_docs = list(dict.fromkeys(combined_docs))
    unique_index = {doc: idx for idx, doc in enumerate(unique_docs)}
    inverse = np.fromiter(
        (unique_index[doc] for doc in combined_docs), dtype=np.int64, count=len(combined_docs)
    )
    llm_args = make_embedding_args(runtime_args)
    llm_args.llm_dual = False
    embeddings, model_name = rfb.fetch_llm_embeddings(unique_docs, llm_args)
    unique_matrix = np.asarray(embeddings, dtype=np.float32)
    if unique_matrix.ndim != 2 or unique_matrix.shape[0] != len(unique_docs):
        raise RuntimeError(f"unexpected unique embedding shape: {unique_matrix.shape}")
    matrix = unique_matrix[inverse]
    print(
        f"[view] embedding_docs={len(combined_docs)} unique_docs={len(unique_docs)}",
        file=sys.stderr,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, np.float32(1e-12))

    views: dict[str, np.ndarray] = {}
    offset = 0
    for name, count in counts.items():
        views[name] = matrix[offset : offset + count]
        offset += count
        print(
            f"[view] {name} model={model_name} embedding_dim={views[name].shape[1]} docs={count}",
            file=sys.stderr,
        )
    for idx, feature in enumerate(features):
        feature.llm_vec = views["features"][idx]
        feature.llm_summary_vec = views["summary"][idx]
    return {name: views[name] for name in ("event", "object", "context")}


def relation_matrix(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = matrix[left]
    b = matrix[right]
    diff = a - b
    dot = np.einsum("ij,ij->i", a, b)
    denom = np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), np.float32(1e-12)
    )
    cosine = (dot / denom)[:, None]
    euclidean = np.linalg.norm(diff, axis=1)[:, None]
    return np.concatenate((np.abs(diff), a * b, cosine, euclidean), axis=1).astype(np.float32)


def build_pair_tail(
    features: list[plf.LLMCaseFeature], pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    if not pairs:
        return np.zeros((0, 0), dtype=np.float32)
    first_i, first_j = pairs[0]
    dim = len(plf.build_structured_pair_feature_vector(features[first_i], features[first_j]))
    dim += len(plf.build_det_scalar_summary_vector(features[first_i], features[first_j]))
    tail = np.empty((len(pairs), dim), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        tail[row] = np.concatenate((
            plf.build_structured_pair_feature_vector(features[i], features[j]),
            plf.build_det_scalar_summary_vector(features[i], features[j]),
        ))
    return tail


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    install_log_sample_cache()
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
        llm_batch_size=int(os.environ.get("BETA_LLM_BATCH_SIZE", "128")),
        llm_timeout_sec=float(os.environ.get("BETA_LLM_TIMEOUT_SEC", "120")),
        svd_dim=int(manifest["svd_dim"]),
    )
    features, _ = plf.build_llm_case_features_for_inputs(
        [input_csv],
        parser=str(manifest["parser"]),
        svd_dim=int(manifest["svd_dim"]),
        llm_args=None,
        log_llm_disabled=False,
    )
    parsed_at = time.perf_counter()
    raw_custom = fetch_all_view_embeddings(
        input_csv,
        features,
        runtime_args,
        str(manifest["parser"]),
        bool(manifest.get("canonicalized_docs", False)),
    )
    embedded_at = time.perf_counter()
    validate_embeddings(features, raw_custom, expected_dim)
    pairs = osf.all_pairs(n)
    pair_left = np.fromiter((i for i, _ in pairs), dtype=np.int64, count=len(pairs))
    pair_right = np.fromiter((j for _, j in pairs), dtype=np.int64, count=len(pairs))
    pair_tail = build_pair_tail(features, pairs)
    tail_at = time.perf_counter()

    seed_probs: list[np.ndarray] = []
    for seed in seeds:
        preproc = joblib.load(model_dir / f"preprocess_seed{seed}.pkl")
        plf.apply_llm_reducer(features, preproc["feature_reducer"], view_dim)
        plf.apply_llm_summary_reducer(features, preproc["summary_reducer"], view_dim)
        reduced_custom = {
            view: plf._apply_reducer_to_matrix(raw, preproc["custom_reducers"][view], view_dim).astype(np.float32)
            for view, raw in raw_custom.items()
        }
        view_mats = {
            "features": np.vstack([f.effective_llm_vec for f in features]).astype(np.float32),
            "summary": np.vstack([f.effective_llm_summary_vec for f in features]).astype(np.float32),
            **reduced_custom,
        }
        relations = {
            view: relation_matrix(view_mats[view], pair_left, pair_right)
            for view in ("features", "summary", "event", "object", "context")
        }
        branch_features = {
            "dual": np.concatenate((relations["features"], relations["summary"], pair_tail), axis=1),
            "quad_event_object_context": np.concatenate((
                relations["features"], relations["summary"], relations["event"],
                relations["object"], relations["context"], pair_tail,
            ), axis=1),
        }
        branch_probs = {}
        for view_config, X in branch_features.items():
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
    finished = time.perf_counter()
    print(
        f"[beta-timing] parse={parsed_at-started:.3f}s embedding={embedded_at-parsed_at:.3f}s "
        f"pair_tail={tail_at-embedded_at:.3f}s model_cluster={finished-tail_at:.3f}s "
        f"total={finished-started:.3f}s",
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
