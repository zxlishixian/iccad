#!/usr/bin/env python3
"""Final-submission inference: 9-fold TriLog ensemble + correlation clustering.

Given a trained model directory (``--model-dir``) produced by
``run_final_submission_train.py``, this reconstructs the pair-feature matrix for
a single held-out ``input.csv``, runs every (seed, fold) model, averages their
pairwise probability matrices, and clusters with weighted correlation
clustering (with a degenerate fallback to agglomerative).

Feature construction mirrors the training hold-out scoring path exactly:
LLM feature/summary reducers, custom (event/object/context) view reducers, and
hierarchical trace-view reducers are all persisted and re-applied here, so the
held-out features are identical to those the model expects.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
import theta_trace_features as ttf
import theta_trilog_model as ttm

DEFAULT_CACHE = Path("/tmp/regr_fail_llm_cache")
DEFAULT_TRACE_CACHE = Path("/tmp/theta_trilog_trace_cache")


def _probability_matrix(pairs, values, case_count):
    out = np.eye(case_count, dtype=np.float32)
    for (left, right), value in zip(pairs, values):
        out[left, right] = out[right, left] = float(value)
    return out


def _embedding_namespace(manifest: dict) -> argparse.Namespace:
    """Namespace shaped like the training script's args for make_embedding_args."""
    return argparse.Namespace(
        parser=str(manifest.get("parser", "drain")),
        svd_dim=int(manifest.get("svd_dim", 64)),
        llm_doc_max_features=int(manifest.get("llm_doc_max_features", 80)),
        llm_cache_dir=DEFAULT_CACHE,
        llm_batch_size=int(manifest.get("llm_batch_size", 64)),
        llm_timeout_sec=60.0,
        embedding_expected_dim=int(manifest.get("embedding_expected_dim", 768)),
        embedding_cache_only=False,
    )


def load_models(model_dir: Path, manifest: dict):
    models, preprocessors = [], []
    for seed in manifest["seeds"]:
        for fold in manifest["folds"]:
            model_path = model_dir / "models" / f"model_{fold}_seed{seed}.pt"
            pre_path = model_dir / "models" / f"preprocess_{fold}_seed{seed}.pkl"
            if model_path.exists() and pre_path.exists():
                pkg = torch.load(model_path, map_location="cpu", weights_only=False)
                pre = joblib.load(pre_path)
                models.append((seed, fold, pkg))
                preprocessors.append((seed, fold, pre))
    return models, preprocessors


def main(argv=None):
    started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    args = p.parse_args(argv)

    cases = osf.read_cases(args.input)
    n = len(cases)
    k = max(1, min(int(args.k), n))
    if n == 0:
        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerow(["Case", "bucket"])
        return 0

    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    view_dim = int(manifest["view_dim"])
    svd_dim = int(manifest.get("svd_dim", 64))
    segment_count = int(manifest.get("trace_segment_count", 16))
    chunk_size = int(manifest.get("trace_chunk_size", 512))
    anchor_sizes = [int(x) for x in manifest.get("trace_anchor_sizes", [32, 64, 128])]

    emb_args = _embedding_namespace(manifest)
    llm_args = gm.make_embedding_args(emb_args)
    features, _ = plf.build_llm_case_features_for_inputs(
        [args.input], parser=manifest.get("parser", "drain"), svd_dim=svd_dim, llm_args=llm_args
    )
    if len(features) != n:
        raise RuntimeError(f"feature/case mismatch: {len(features)} != {n}")
    trace_features, _ = ttf.build_hierarchical_trace_features(
        args.input,
        cache_dir=DEFAULT_TRACE_CACHE,
        segment_count=segment_count,
        chunk_size=chunk_size,
        anchor_sizes=anchor_sizes,
    )
    docs = gm.build_all_view_documents(args.input.parent)
    raw_custom = {name: gm.fetch_view_embeddings(docs[name], emb_args, name) for name in ("event", "object", "context")}

    pairs = osf.all_pairs(n)
    models, preprocessors = load_models(args.model_dir, manifest)
    if not models:
        raise RuntimeError("no fold models found; deterministic fallback expected upstream")

    seed_probs: dict[int, list[np.ndarray]] = {}
    for (seed, _fold, pkg), (_, _, pre) in zip(models, preprocessors):
        plf.apply_llm_reducer(features, pre["feature_reducer"], view_dim)
        plf.apply_llm_summary_reducer(features, pre["summary_reducer"], view_dim)
        reduced_custom: dict[str, np.ndarray] = {}
        for name, reducer in pre["custom_reducers"].items():
            raw = raw_custom.get(name, np.zeros((n, 0), dtype=np.float32))
            reduced_custom[name] = (
                plf._apply_reducer_to_matrix(raw, reducer, view_dim).astype(np.float32, copy=False)
                if reducer is not None and raw.shape[1] > 0
                else np.zeros((n, 0), dtype=np.float32)
            )
        trace_matrices = ttf.apply_trace_reducers(pre["trace_bundle"], trace_features)
        base = gm.build_multiview_pair_feature_matrix(features, reduced_custom, pre["view_names"], pairs)
        trace = ttf.build_trace_pair_feature_components(trace_features, trace_matrices, pairs)["residual"]
        matrix = np.hstack([base, trace]).astype(np.float32, copy=False)
        flat = ttm.predict_trilog_pair_model(pkg, matrix)
        seed_probs.setdefault(seed, []).append(_probability_matrix(pairs, flat, n))

    # Mean probability: average across folds within a seed, then across seeds.
    mean_prob = np.mean(
        [np.mean(np.stack(mats, axis=0), axis=0) for mats in seed_probs.values()], axis=0
    ).astype(np.float32)

    # Co-association consensus across seeds (weighted in by consensus_weight).
    cw = float(manifest.get("consensus_weight", 0.0))
    if cw > 0.0:
        source_clusterer = str(manifest.get("source_clusterer", "agglomerative_avg"))
        coassoc = np.zeros((n, n), dtype=np.float32)
        for mats in seed_probs.values():
            seed_prob = np.mean(np.stack(mats, axis=0), axis=0)
            labels = np.asarray(gc.cluster_probability_graph(seed_prob, k, source_clusterer).labels)
            coassoc += (labels[:, None] == labels[None, :]).astype(np.float32)
        coassoc /= max(1, len(seed_probs))
        final_prob = ((1.0 - cw) * mean_prob + cw * coassoc).astype(np.float32)
    else:
        final_prob = mean_prob

    final_prob = (final_prob + final_prob.T) * 0.5
    np.fill_diagonal(final_prob, 1.0)

    clustered = gc.cluster_with_fallback(
        final_prob, k, cannot_link_weight=float(manifest.get("cannot_link_weight", 100.0))
    )
    labels = np.asarray(clustered.labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "bucket"])
        for case, label in zip(cases, labels):
            w.writerow([case, f"bucket_{int(label):03d}"])

    print(
        f"[final-inference] cases={n} clusters={len(set(labels.tolist()))} "
        f"models={len(models)} total={time.perf_counter() - started:.3f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
