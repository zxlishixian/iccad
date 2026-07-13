#!/usr/bin/env python3
"""LODO multi-view inference: 7-fold ensemble, no data leakage.

Each fold-model was trained on 6 datasets, never seeing the 7th.
At inference, all 7 fold-models predict independently, then their
probabilities are averaged (within each seed), then averaged across seeds.
"""

from __future__ import annotations

import argparse, csv, json, os, sys, time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

import graph_clustering as gc
import official_style_features as osf
import pairwise_features as pf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
from beta_multiview_inference import (
    build_feature_documents, canonicalize_document, install_log_sample_cache,
)
from run_graph_multiview_experiments import (
    build_all_view_documents, make_embedding_args,
    predict_view_probabilities, views_for_config,
    relation_block_with_scalars,
)

FOLD_NAMES = [
    "first_batch_dataset", "stage2_dataset_working",
    "stage3_dataset_32bugs_640cases", "official_vcs_stage1_dataset_v1",
    "directed_cross_v2", "benchmark_set_1", "benchmark_set_2",
]


def model_dir() -> Path:
    override = os.environ.get("BETA_MULTIVIEW_MODEL_DIR", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent / "multiview" / "models_multiview_lodo"


def relation_matrix(mat, left, right):
    """Build pair relation features matching training's relation_block_with_scalars format."""
    n_pairs = len(left)
    rows = np.empty((n_pairs, mat.shape[1] * 2 + 2), dtype=np.float32)
    for idx, (i, j) in enumerate(zip(left, right)):
        rows[idx] = relation_block_with_scalars(mat, i, j)
    return rows


def build_pair_tail(features, pairs):
    rows = []
    for i, j in pairs:
        s = plf.build_structured_pair_feature_vector(features[i], features[j])
        d = plf.build_det_scalar_summary_vector(features[i], features[j])
        rows.append(np.concatenate([s, d]))
    return np.array(rows, dtype=np.float32)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, required=True)
    return p.parse_args(argv)


def main():
    install_log_sample_cache()
    args = parse_args()
    started = time.perf_counter()
    cases = osf.read_cases(args.input)

    mdir = model_dir()
    manifest = json.loads((mdir / "manifest.json").read_text())
    view_dim = manifest["view_dim"]
    seeds = manifest["seeds"]
    source_clusterer = manifest["source_clusterer"]
    final_clusterer = manifest["final_clusterer"]
    consensus_weight = manifest.get("consensus_weight", 0.0)
    beta = manifest.get("primary_beta", 0.50)

    # Build features (same as original beta_multiview_inference)
    llm_args = make_embedding_args(argparse.Namespace(
        parser="drain", svd_dim=64, view_dim=view_dim,
        llm_doc_max_features=80, llm_cache_dir=Path("/tmp/regr_fail_llm_cache"),
        llm_batch_size=128, llm_timeout_sec=600.0,
        view_hard_positive_ratio=1.0,
    ))
    logger_args = argparse.Namespace(
        parser="drain", svd_dim=64, llm_mode="none",
    )

    features, _ = plf.build_llm_case_features_for_inputs(
        [args.input], parser="drain", svd_dim=64, llm_args=llm_args)
    n = len(features)
    k = max(1, min(int(args.k), n))

    feature_docs, summary_docs = build_feature_documents(
        args.input, "drain", 80)
    custom_docs = build_all_view_documents(args.input.parent)
    docs = {"features": feature_docs, "summary": summary_docs, **custom_docs}
    counts = {v: len(vals) for v, vals in docs.items()}
    combined = [d for vals in docs.values() for d in vals]
    unique_docs = list(dict.fromkeys(combined))
    unique_idx = {d: i for i, d in enumerate(unique_docs)}
    inverse = np.fromiter((unique_idx[d] for d in combined), dtype=np.int64, count=len(combined))
    embeddings, model_name = rfb.fetch_llm_embeddings(unique_docs, llm_args)
    umat = np.asarray(embeddings, dtype=np.float32)
    umat /= np.maximum(np.linalg.norm(umat, axis=1, keepdims=True), np.float32(1e-12))
    matrix = umat[inverse]
    raw_views = {}
    off = 0
    for view, count in counts.items():
        raw_views[view] = matrix[off:off + count].astype(np.float32, copy=False)
        off += count

    for idx, feat in enumerate(features):
        feat.llm_vec = raw_views["features"][idx]
        feat.llm_summary_vec = raw_views["summary"][idx]
    raw_custom = {v: raw_views[v] for v in ("event", "object", "context")}

    pairs = osf.all_pairs(n)
    pair_left = np.fromiter((i for i, _ in pairs), dtype=np.int64, count=len(pairs))
    pair_right = np.fromiter((j for _, j in pairs), dtype=np.int64, count=len(pairs))

    # Ensemble across LODO folds and seeds
    all_seed_probs = []
    for seed in seeds:
        fold_probs = []
        for fold_name in FOLD_NAMES:
            preproc_path = mdir / f"preprocess_{fold_name}_seed{seed}.pkl"
            if not preproc_path.exists():
                continue
            preproc = joblib.load(preproc_path)
            plf.apply_llm_reducer(features, preproc["feature_reducer"], view_dim)
            plf.apply_llm_summary_reducer(features, preproc["summary_reducer"], view_dim)
            reduced_custom = {}
            for view in ("event", "object", "context"):
                if view in preproc.get("custom_reducers", {}):
                    reduced_custom[view] = plf._apply_reducer_to_matrix(
                        raw_custom.get(view, np.zeros((n, 768), dtype=np.float32)),
                        preproc["custom_reducers"][view], view_dim,
                    ).astype(np.float32)

            view_mats = {
                "features": np.vstack([f.effective_llm_vec for f in features]).astype(np.float32),
                "summary": np.vstack([f.effective_llm_summary_vec for f in features]).astype(np.float32),
                **reduced_custom,
            }
            relations = {}
            for v in ("features", "summary", "event", "object", "context"):
                if v in view_mats:
                    relations[v] = relation_matrix(view_mats[v], pair_left, pair_right)

            pair_tail = build_pair_tail(features, pairs)
            branch_features = {}
            if "dual" in manifest.get("view_configs", ["dual"]):
                branch_features["dual"] = np.concatenate(
                    (relations["features"], relations["summary"], pair_tail), axis=1)
            if "quad_event_object_context" in manifest.get("view_configs", []):
                branch_features["quad_event_object_context"] = np.concatenate((
                    relations["features"], relations["summary"],
                    relations.get("event", np.zeros((len(pairs), 0), dtype=np.float32)),
                    relations.get("object", np.zeros((len(pairs), 0), dtype=np.float32)),
                    relations.get("context", np.zeros((len(pairs), 0), dtype=np.float32)),
                    pair_tail), axis=1)

            branch_probs = {}
            for vc, X in branch_features.items():
                model_path = mdir / f"model_{fold_name}_{vc}_seed{seed}.pkl"
                if model_path.exists():
                    model_pkg = joblib.load(model_path)
                    branch_probs[vc] = predict_view_probabilities(model_pkg, X, pairs, n)

            if "dual" in branch_probs and "quad_event_object_context" in branch_probs:
                prob = ((1.0 - beta) * branch_probs["dual"] + beta * branch_probs["quad_event_object_context"]).astype(np.float32)
            elif "dual" in branch_probs:
                prob = branch_probs["dual"]
            else:
                prob = list(branch_probs.values())[0]
            prob = (prob + prob.T) * 0.5
            np.fill_diagonal(prob, 1.0)
            fold_probs.append(prob)

        if not fold_probs:
            continue
        seed_prob = np.mean(np.stack(fold_probs, axis=0), axis=0).astype(np.float32)
        all_seed_probs.append(seed_prob)

    if not all_seed_probs:
        raise RuntimeError("No fold models produced valid probabilities")

    mean_prob = np.mean(np.stack(all_seed_probs, axis=0), axis=0).astype(np.float32)
    coassoc = np.zeros_like(mean_prob)
    for prob in all_seed_probs:
        seed_labels = np.asarray(gc.cluster_probability_graph(prob, k, source_clusterer).labels)
        coassoc += (seed_labels[:, None] == seed_labels[None, :]).astype(np.float32)
    coassoc /= float(len(all_seed_probs))
    final_prob = ((1.0 - consensus_weight) * mean_prob + consensus_weight * coassoc).astype(np.float32)
    final_prob = (final_prob + final_prob.T) * 0.5
    np.fill_diagonal(final_prob, 1.0)
    labels = gc.cluster_probability_graph(final_prob, k, final_clusterer).labels

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "bucket"])
        for case, label in zip(cases, labels):
            w.writerow([case, f"bucket_{label:03d}"])

    finished = time.perf_counter()
    print(f"[beta-v3-lodo] cases={n} folds={len(fold_probs)}/{len(FOLD_NAMES)} "
          f"seeds={len(all_seed_probs)} clusters={len(set(labels))} "
          f"total={finished-started:.3f}s", file=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"[beta-v3-lodo] failed: {exc}", file=sys.stderr)
        raise
