#!/usr/bin/env python3
"""Theta v5 bounded-correction and episode-robust LODO experiments.

Strict 7-episode leave-one-dataset-out evaluation for the v5 candidate
architecture:

    P_final = sigmoid(logit(P_no_trace_teacher) + alpha * tanh(z_trace / tau))

The trace correction is bounded (tanh) so it can never override the teacher.
A label-free OOD gate shrinks alpha on holdout episodes whose trace pair
features look out-of-distribution relative to the training episodes, which
targets the known set1 trace domain shift.

Variant ``groupdro_*`` reweights training episodes by in-sample group loss
(GroupDRO-lite) before fitting the teacher, targeting worst-episode
robustness.  Gold is read only for final scoring; no held-out gold is used to
select any hyperparameter.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
import run_graph_multiview_experiments as gm
import theta_trace_features as ttf
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred
from run_theta_trilog_lodo import (
    _binary_metrics,
    _flat_probabilities,
    _logit,
    _official_pair_mask,
    _pair_labels,
    _probability_matrix,
    resolve,
    summarize,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("dataset/fake_dataset/old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("dataset/fake_dataset/official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/directed_cross_v4"),
    Path("dataset/fake_dataset/official_format_fake_dataset/benchmark5_final"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]
VARIANTS = ("teacher", "bounded_trace", "bounded_trace_gated", "groupdro_teacher", "groupdro_bounded_trace")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--holdout-datasets", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--graph-method", default="agglomerative_avg")
    parser.add_argument("--view-dim", type=int, default=64)
    parser.add_argument("--trace-global-struct-dim", type=int, default=48)
    parser.add_argument("--trace-global-text-dim", type=int, default=48)
    parser.add_argument("--trace-anchor-struct-dim", type=int, default=32)
    parser.add_argument("--trace-anchor-text-dim", type=int, default=32)
    parser.add_argument("--trace-residual-struct-dim", type=int, default=48)
    parser.add_argument("--trace-residual-text-dim", type=int, default=48)
    parser.add_argument("--trace-segment-count", type=int, default=16)
    parser.add_argument("--trace-chunk-size", type=int, default=512)
    parser.add_argument("--trace-anchor-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--trace-cache-dir", type=Path, default=Path("/tmp/theta_trilog_trace_cache"))
    parser.add_argument("--trace-force-rebuild", action="store_true")
    parser.add_argument("--view-max-pairs-per-dataset", type=int, default=60000)
    parser.add_argument("--view-negative-ratio", type=float, default=2.0)
    parser.add_argument("--view-hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--view-hard-positive-ratio", type=float, default=1.0)
    parser.add_argument("--view-dataset-balance-power", type=float, default=1.0)
    parser.add_argument("--view-official-weight", type=float, default=4.0)
    parser.add_argument("--parser", default="drain")
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--llm-doc-max-features", type=int, default=80)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    parser.add_argument("--llm-batch-size", type=int, default=64)
    parser.add_argument("--llm-timeout-sec", type=float, default=20.0)
    parser.add_argument("--embedding-cache-only", action="store_true")
    parser.add_argument("--no-llm-views", action="store_true", help="drop all LLM embedding views; deterministic features only")
    parser.add_argument("--embedding-expected-dim", type=int, default=768)
    # Bounded-correction hyperparameters (fixed by principle, not tuned on gold).
    parser.add_argument("--trace-alpha", type=float, default=0.5)
    parser.add_argument("--trace-tau", type=float, default=1.0)
    parser.add_argument("--ood-kappa", type=float, default=0.5)
    parser.add_argument("--trace-block", choices=("full", "residual"), default="full")
    # GroupDRO-lite hyperparameters.
    parser.add_argument("--groupdro-iterations", type=int, default=5)
    parser.add_argument("--groupdro-eta", type=float, default=0.5)
    return parser.parse_args(argv)


def _gbdt_fit(X: np.ndarray, y: np.ndarray, weight: np.ndarray, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        early_stopping=False,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y, sample_weight=weight)
    return {"model": model, "scaler": None, "model_type": "gbdt"}


def _bce_group_loss(y: np.ndarray, prob: np.ndarray) -> float:
    clipped = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)))


def groupdro_teacher(X: np.ndarray, y: np.ndarray, base_weight: np.ndarray, groups: np.ndarray, group_names: Sequence[str], args: argparse.Namespace, seed: int) -> dict:
    """Iteratively reweight episode groups by their in-sample BCE loss."""
    group_ids = np.unique(groups)
    weights = np.ones(len(group_ids), dtype=np.float64)
    model_pkg = None
    history: list[dict] = []
    for iteration in range(int(args.groupdro_iterations)):
        pair_weight = base_weight.astype(np.float64) * weights[groups]
        pair_weight /= max(float(np.mean(pair_weight)), 1e-12)
        model_pkg = _gbdt_fit(X, y, pair_weight.astype(np.float32), seed + iteration * 101)
        prob = np.clip(model_pkg["model"].predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
        group_losses = {
            int(gid): _bce_group_loss(y[groups == gid], prob[groups == gid])
            for gid in group_ids
        }
        mean_loss = float(np.mean(list(group_losses.values())))
        for gid in group_ids:
            weights[gid] *= float(np.exp(args.groupdro_eta * (group_losses[gid] - mean_loss)))
        weights /= float(np.mean(weights))
        history.append({
            "iteration": iteration,
            "mean_loss": mean_loss,
            "group_losses": {str(group_names[int(k)]): round(float(v), 5) for k, v in group_losses.items()},
            "group_weights": {str(group_names[int(k)]): round(float(v), 4) for k, v in zip(group_ids, weights)},
        })
    assert model_pkg is not None
    model_pkg["groupdro_history"] = history
    model_pkg["groupdro_final_weights"] = {str(group_names[int(k)]): float(v) for k, v in zip(group_ids, weights)}
    return model_pkg


def _ood_score(train_matrix: np.ndarray, hold_matrix: np.ndarray) -> float:
    """Label-free robust out-of-distribution score for holdout pair features."""
    if train_matrix.shape[1] == 0:
        return 0.0
    median = np.median(train_matrix, axis=0)
    mad = np.median(np.abs(train_matrix - median), axis=0)
    scale = np.maximum(mad, 1e-6)
    z = np.abs(hold_matrix - median) / scale
    z = np.clip(z, 0.0, 3.0)
    trimmed = np.sort(z, axis=0)
    keep = max(1, int(0.9 * trimmed.shape[0]))
    return float(np.mean(trimmed[-keep:, :]))


def prepare(args: argparse.Namespace) -> tuple:
    datasets = [resolve(path) for path in args.datasets]
    for dataset in datasets:
        if not (dataset / "input.csv").exists():
            raise FileNotFoundError(dataset / "input.csv")
    llm_args = gm.make_embedding_args(args)
    base_features: list = []
    slices: list[dict] = []
    trace_features: list = []
    trace_debug: list[dict] = []
    custom_documents = {name: [] for name in ("event", "object", "context")}
    offset = 0
    trace_started = time.perf_counter()
    for dataset in datasets:
        episode_features, _ = plf.build_llm_case_features_for_inputs(
            [dataset / "input.csv"], parser=args.parser, svd_dim=args.svd_dim, llm_args=llm_args
        )
        labels = read_gold(osf.gold_path(dataset))
        cases = osf.read_cases(dataset / "input.csv")
        if not (len(episode_features) == len(labels) == len(cases)):
            raise ValueError(f"case/feature/label mismatch for {dataset}")
        base_features.extend(episode_features)
        slices.append({
            "name": dataset.name,
            "path": dataset,
            "start": offset,
            "stop": offset + len(labels),
            "labels": labels,
            "cases": cases,
        })
        offset += len(labels)
        documents = gm.build_all_view_documents(dataset)
        for name in custom_documents:
            custom_documents[name].extend(documents[name])
        episode_trace, episode_debug = ttf.build_hierarchical_trace_features(
            dataset / "input.csv",
            cache_dir=args.trace_cache_dir,
            segment_count=args.trace_segment_count,
            chunk_size=args.trace_chunk_size,
            anchor_sizes=args.trace_anchor_sizes,
            force_rebuild=args.trace_force_rebuild,
        )
        if len(episode_trace) != len(labels):
            raise ValueError(f"trace/label mismatch for {dataset}")
        trace_features.extend(episode_trace)
        for row in episode_debug:
            trace_debug.append({"dataset": dataset.name, **row})
    trace_parse_runtime = time.perf_counter() - trace_started
    raw_custom = {
        name: gm.fetch_view_embeddings(documents, args, name)
        for name, documents in custom_documents.items()
    }
    return base_features, slices, trace_features, raw_custom, trace_debug, trace_parse_runtime


def run_fold(args: argparse.Namespace, prepared, holdout: dict, seed: int) -> list[dict]:
    base_features, slices, trace_features, raw_custom, _, _ = prepared
    train_indices = [
        index
        for episode in slices
        if episode["name"] != holdout["name"]
        for index in range(episode["start"], episode["stop"])
    ]
    hold_indices = list(range(holdout["start"], holdout["stop"]))
    train_base_features = [base_features[index] for index in train_indices]
    hold_base_features = [base_features[index] for index in hold_indices]
    feature_reducer = plf.fit_llm_reducer(train_base_features, args.view_dim, random_state=seed)
    summary_reducer = plf.fit_llm_summary_reducer(train_base_features, args.view_dim, random_state=seed + 17)
    plf.apply_llm_reducer(hold_base_features, feature_reducer, args.view_dim)
    plf.apply_llm_summary_reducer(hold_base_features, summary_reducer, args.view_dim)
    reduced_custom = {
        name: gm.fit_apply_reduced_matrix(raw, train_indices, args.view_dim, seed + 101 + position * 13)
        for position, (name, raw) in enumerate(raw_custom.items())
    }
    _, trace_matrices = ttf.fit_transform_trace_views(
        trace_features,
        train_indices,
        seed=seed,
        global_struct_dim=args.trace_global_struct_dim,
        global_text_dim=args.trace_global_text_dim,
        anchor_struct_dim=args.trace_anchor_struct_dim,
        anchor_text_dim=args.trace_anchor_text_dim,
        residual_struct_dim=args.trace_residual_struct_dim,
        residual_text_dim=args.trace_residual_text_dim,
    )
    train_pairs, train_y, sample_weight, pair_stats = gm.sample_lodo_train_pairs(
        base_features, slices, holdout["name"], args, seed
    )
    official_pair_mask = _official_pair_mask(train_pairs, slices)
    hold_pairs = osf.all_pairs(len(hold_indices))
    hold_y = _pair_labels(holdout["labels"], hold_pairs)
    view_names = [] if args.no_llm_views else gm.views_for_config("quad_event_object_context")
    train_base = gm.build_multiview_pair_feature_matrix(base_features, reduced_custom, view_names, train_pairs)
    hold_custom = {name: matrix[hold_indices] for name, matrix in reduced_custom.items()}
    hold_base = gm.build_multiview_pair_feature_matrix(hold_base_features, hold_custom, view_names, hold_pairs)
    trace_components = ttf.build_trace_pair_feature_components(trace_features, trace_matrices, train_pairs)
    hold_trace_features = [trace_features[index] for index in hold_indices]
    hold_trace_matrices = {name: matrix[hold_indices] for name, matrix in trace_matrices.items()}
    hold_trace_components = ttf.build_trace_pair_feature_components(hold_trace_features, hold_trace_matrices, hold_pairs)
    conflict = gm.conflict_matrix_from_records(
        osf.build_case_records(holdout["name"], holdout["path"] / "input.csv", gold_csv=None)
    )
    k = len(set(holdout["labels"]))
    group_ids = np.zeros(len(train_pairs), dtype=np.int32)
    group_names: list[str] = []
    for episode in slices:
        if episode["name"] == holdout["name"]:
            continue
        group_names.append(str(episode["name"]))
        group_ids[
            np.logical_and(np.asarray(train_pairs)[:, 0] >= episode["start"], np.asarray(train_pairs)[:, 0] < episode["stop"])
        ] = len(group_names) - 1
    rows: list[dict] = []
    for variant in args.variants:
        started = time.perf_counter()
        notes: dict = {}
        use_groupdro = variant.startswith("groupdro_")
        use_trace = variant in {"bounded_trace", "bounded_trace_gated", "groupdro_bounded_trace"}
        use_gate = variant == "bounded_trace_gated"
        if use_groupdro:
            teacher = groupdro_teacher(train_base, train_y, sample_weight, group_ids, group_names, args, seed)
            notes["groupdro_history"] = teacher.pop("groupdro_history", [])
            notes["groupdro_final_weights"] = teacher.pop("groupdro_final_weights", {})
        else:
            teacher = _gbdt_fit(train_base, train_y, sample_weight, seed)
        hold_teacher = _flat_probabilities(teacher, hold_base)
        probability = _probability_matrix(hold_pairs, hold_teacher, len(hold_indices))
        if use_trace:
            block = str(args.trace_block)
            trace_matrix = trace_components[block]
            hold_trace = hold_trace_components[block]
            trace_model = _gbdt_fit(trace_matrix, train_y, sample_weight, seed + 31)
            trace_prob = np.clip(trace_model["model"].predict_proba(hold_trace)[:, 1], 1e-6, 1.0 - 1e-6)
            z_trace = _logit(trace_prob)
            alpha = float(args.trace_alpha)
            if use_gate:
                ood = _ood_score(trace_matrix, hold_trace)
                alpha = alpha * float(np.exp(-args.ood_kappa * ood))
                notes["ood_score"] = float(ood)
                notes["alpha_effective"] = float(alpha)
            correction = alpha * np.tanh(z_trace / float(args.trace_tau))
            fused_logit = _logit(hold_teacher) + correction
            fused = np.clip(1.0 / (1.0 + np.exp(-fused_logit)), 1e-6, 1.0 - 1e-6)
            probability = _probability_matrix(hold_pairs, fused, len(hold_indices))
        flat = np.asarray([probability[i, j] for i, j in hold_pairs], dtype=np.float32)
        pair_auc, pair_bce = _binary_metrics(flat, hold_y)
        clustered = gc.cluster_probability_graph(probability, k, args.graph_method, conflict_matrix=conflict)
        runtime = time.perf_counter() - started
        pred_path = args.output_dir / "preds" / f"{holdout['name']}_{variant}_{args.graph_method}_seed{seed}.csv"
        pred = write_pred(pred_path, holdout["cases"], clustered.labels)
        prob_path = args.output_dir / "probs" / f"{holdout['name']}_{variant}_{args.graph_method}_seed{seed}.npy"
        prob_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prob_path, probability)
        ba, tpr, tnr = pairwise_scores(holdout["labels"], pred)
        rows.append({
            "dataset": holdout["name"],
            "seed": seed,
            "variant": variant,
            "graph_method": args.graph_method,
            "BA": ba,
            "TPR": tpr,
            "TNR": tnr,
            "pair_AUC": pair_auc,
            "pair_BCE": pair_bce,
            "k": k,
            "cases": len(hold_indices),
            "num_clusters": len(set(pred)),
            "train_pairs": len(train_y),
            "runtime_sec": runtime,
            "notes": json.dumps(notes, sort_keys=True),
        })
        print(
            f"[v5] dataset={holdout['name']} seed={seed} variant={variant} "
            f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f} AUC={pair_auc:.6f} notes={json.dumps(notes, sort_keys=True)}",
            flush=True,
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = resolve(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.embedding_cache_only:
        rfb.fetch_llm_embeddings = gm.fetch_cached_llm_embeddings
    prepared = prepare(args)
    rows: list[dict] = []
    for seed in args.seeds:
        for holdout in prepared[1]:
            if args.holdout_datasets and holdout["name"] not in args.holdout_datasets:
                continue
            rows.extend(run_fold(args, prepared, holdout, seed))
    fields = [
        "dataset", "seed", "variant", "graph_method", "BA", "TPR", "TNR",
        "pair_AUC", "pair_BCE", "k", "cases", "num_clusters", "train_pairs",
        "runtime_sec", "notes",
    ]
    write_csv(args.output_dir / "results.csv", rows, fields)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["graph_method"])].append(row)
    summary_rows: list[dict] = []
    for (variant, graph_method), values in grouped.items():
        dataset_means: dict[str, float] = {}
        for dataset in sorted({str(row["dataset"]) for row in values}):
            dataset_means[dataset] = float(np.mean([float(row["BA"]) for row in values if row["dataset"] == dataset]))
        bas = list(dataset_means.values())
        official = [dataset_means[name] for name in ("benchmark_set_1", "benchmark_set_2") if name in dataset_means]
        summary_rows.append({
            "variant": variant,
            "graph_method": graph_method,
            "mean_BA": float(np.mean(bas)) if bas else 0.0,
            "worst_BA": float(np.min(bas)) if bas else 0.0,
            "official_mean_BA": float(np.mean(official)) if official else 0.0,
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])),
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])),
            "mean_pair_AUC": float(np.mean([float(row["pair_AUC"]) for row in values])),
            "runs": len(values),
            "dataset_means": json.dumps(dataset_means, sort_keys=True),
        })
    summary_rows.sort(key=lambda row: (row["official_mean_BA"], row["mean_BA"], row["worst_BA"]), reverse=True)
    write_csv(args.output_dir / "summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else [])
    (args.output_dir / "manifest.json").write_text(
        json.dumps({
            "datasets": [str(d) for d in args.datasets],
            "seeds": args.seeds,
            "variants": args.variants,
            "graph_method": args.graph_method,
            "bounded": {"alpha": args.trace_alpha, "tau": args.trace_tau, "ood_kappa": args.ood_kappa, "trace_block": args.trace_block},
            "no_llm_views": bool(args.no_llm_views),
            "groupdro": {"iterations": args.groupdro_iterations, "eta": args.groupdro_eta},
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n| rank | variant | official mean | macro BA | worst BA | pair AUC |")
    print("|---:|---|---:|---:|---:|---:|")
    for rank, row in enumerate(summary_rows, 1):
        print(
            f"| {rank} | {row['variant']} | {row['official_mean_BA']:.4f} | "
            f"{row['mean_BA']:.4f} | {row['worst_BA']:.4f} | {row['mean_pair_AUC']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
