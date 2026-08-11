#!/usr/bin/env python3
"""Experimental Theta compact-student inference.

This entry point never discovers gold/golden/meta/trace files. It is not wired
into the formal predictor or submission router yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

import graph_clustering as gc
import official_style_features as osf
import theta_clustering as tc
import theta_features as tf


ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--clusterer", choices=["theta_graph", "average"], default="theta_graph")
    parser.add_argument("--top-l", type=int, default=48)
    parser.add_argument("--full-pair-limit", type=int, default=300)
    parser.add_argument(
        "--candidate-mode",
        choices=["concat", "multiview_anchor"],
        default="concat",
    )
    parser.add_argument("--anchors-per-cluster", type=int, default=2)
    parser.add_argument("--anchor-cluster-count", type=int, default=8)
    parser.add_argument("--pair-batch-size", type=int, default=32768)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/theta_llm_cache"))
    parser.add_argument("--llm-batch-size", type=int, default=128)
    parser.add_argument("--llm-timeout-sec", type=float, default=120.0)
    parser.add_argument("--conflict-penalty", type=float, default=2.0)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--graph-max-iter", type=int, default=12)
    parser.add_argument("--diagnostics", type=Path)
    return parser.parse_args(argv)


def write_output(path: Path, cases: Sequence[str], labels: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Case", "bucket"])
        writer.writerows(
            (case, f"bucket_{int(label):03d}") for case, label in zip(cases, labels)
        )


def predict_pair_probabilities(
    model_pkg: dict,
    cases: Sequence[tf.ThetaCaseFeature],
    pairs: Sequence[tuple[int, int]],
    batch_size: int,
    expected_dim: int,
) -> np.ndarray:
    model = model_pkg["model"]
    scaler = model_pkg.get("scaler")
    chunks: list[np.ndarray] = []
    for start in range(0, len(pairs), max(1, int(batch_size))):
        batch = pairs[start : start + max(1, int(batch_size))]
        X = tf.build_theta_pair_feature_matrix(cases, batch)
        if X.shape[1] != int(expected_dim):
            raise RuntimeError(
                f"Theta feature dimension mismatch: got={X.shape[1]} expected={expected_dim}"
            )
        if scaler is not None:
            X = scaler.transform(X)
        if model_pkg.get("model_type") == "theta_gated_mlp":
            import torch

            with torch.no_grad():
                logits = model(torch.from_numpy(np.asarray(X, dtype=np.float32)))
                chunks.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
        else:
            chunks.append(model.predict_proba(X)[:, 1].astype(np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def load_theta_model(model_dir: Path) -> dict:
    model_pkg = joblib.load(model_dir / "pair_student.pkl")
    if model_pkg.get("model_type") != "theta_gated_mlp":
        return model_pkg
    import torch
    from pairwise_neural_models import build_pairwise_neural_model

    checkpoint = torch.load(
        model_dir / "pair_student.pt", map_location="cpu", weights_only=False
    )
    model = build_pairwise_neural_model(
        input_dim=int(checkpoint["input_dim"]),
        model_arch="gated_mlp",
        hidden_dims=checkpoint["hidden_dims"],
        dropout=float(checkpoint["dropout"]),
        layernorm=True,
        batchnorm=False,
        **dict(checkpoint.get("model_config", {})),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model_pkg["model"] = model.eval()
    return model_pkg


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    input_csv = args.input.resolve()
    model_dir = args.model_dir.resolve()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    reducers = joblib.load(model_dir / "reducers.pkl")
    model_pkg = load_theta_model(model_dir)
    output_cases = osf.read_cases(input_csv)
    theta_cases, evidence_debug = tf.build_theta_case_features(
        input_csv,
        parser=str(manifest["parser"]),
        svd_dim=int(manifest["svd_dim"]),
        context_radius=int(manifest["context_radius"]),
        context_events=int(manifest["context_events"]),
    )
    parsed_at = time.perf_counter()
    if len(theta_cases) != len(output_cases):
        raise RuntimeError(
            f"Theta input alignment failed: output_cases={len(output_cases)} features={len(theta_cases)}"
        )
    embedding_model = "none"
    embedding_stats = {"documents": 0, "unique_documents": 0}
    if manifest.get("embedding_mode") == "embedding":
        embedding_model, raw_dim, embedding_stats = tf.fetch_theta_embeddings(
            theta_cases,
            cache_dir=args.llm_cache_dir,
            batch_size=args.llm_batch_size,
            timeout_sec=args.llm_timeout_sec,
        )
        if raw_dim != int(manifest["raw_embedding_dim"]):
            raise RuntimeError(
                f"Theta endpoint dimension mismatch: got={raw_dim} "
                f"expected={manifest['raw_embedding_dim']}"
            )
    tf.apply_theta_reducers(theta_cases, reducers)
    embedded_at = time.perf_counter()

    n = len(theta_cases)
    if n == 0:
        write_output(args.output, [], [])
        return 0
    k = max(1, min(int(args.k), n))
    if n == 1:
        write_output(args.output, output_cases, [0])
        return 0
    pairs, candidate_debug = tf.candidate_pairs(
        theta_cases,
        top_l=args.top_l,
        full_pair_limit=args.full_pair_limit,
        mode=args.candidate_mode,
        reference_k=k,
        anchors_per_cluster=args.anchors_per_cluster,
        anchor_cluster_count=args.anchor_cluster_count,
        random_state=int(manifest["random_state"]),
    )
    probabilities = predict_pair_probabilities(
        model_pkg, theta_cases, pairs, args.pair_batch_size, int(manifest["feature_dim"])
    )
    conflicts = tf.conflict_values(theta_cases, pairs)
    case_matrix = tf.theta_case_matrix(theta_cases)
    scored_at = time.perf_counter()

    if args.clusterer == "average":
        dense = tc.dense_probability_matrix(n, pairs, probabilities)
        result = gc.agglomerative_avg(dense, k)
        labels = result.labels
        cluster_debug = {
            "method": "average",
            "iterations": 0,
            "moves": 0,
            "objective": 0.0,
            "trajectory": [],
        }
    else:
        result = tc.sparse_signed_graph_cluster(
            case_matrix,
            pairs,
            probabilities,
            k,
            conflicts=conflicts,
            conflict_penalty=args.conflict_penalty,
            balance_weight=args.balance_weight,
            max_iter=args.graph_max_iter,
            random_state=int(manifest["random_state"]),
        )
        labels = result.labels
        cluster_debug = {
            "method": result.method,
            "iterations": result.iterations,
            "moves": result.moves,
            "objective": result.objective,
            "trajectory": result.trajectory,
        }
    write_output(args.output, output_cases, labels)
    finished = time.perf_counter()
    diagnostics = {
        "model_name": manifest["model_name"],
        "cases": n,
        "requested_k": int(args.k),
        "selected_k": len(set(labels)),
        "embedding_model": embedding_model,
        "embedding_stats": embedding_stats,
        "candidate_graph": candidate_debug,
        "cluster": cluster_debug,
        "evidence_status": {
            "sim_ok": sum(row["sim_status"] == "ok" for row in evidence_debug),
            "regr_ok": sum(row["regr_status"] == "ok" for row in evidence_debug),
        },
        "timing": {
            "parse": parsed_at - started,
            "embedding": embedded_at - parsed_at,
            "pair_score": scored_at - embedded_at,
            "cluster_write": finished - scored_at,
            "total": finished - started,
        },
    }
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(
        f"[theta] cases={n} k={k} clusters={len(set(labels))} "
        f"pairs={len(pairs)}/{candidate_debug['all_pairs']} "
        f"embedding_model={embedding_model} total={finished-started:.2f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
