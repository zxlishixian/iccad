#!/usr/bin/env python3
"""Strict LODO experiments for the experimental Theta v3 TriLog model.

The normal TriLog path always consumes sim.log, regr.log, and trace.log.  Gold
is read only by this training/evaluation runner.  The formal predictor is not
modified.
"""

from __future__ import annotations

import argparse
import json
import math
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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("official_format_fake_dataset/directed_cross_v4"),
    Path("official_format_fake_dataset/benchmark5_final"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]
CONFIGS = (
    "no_trace",
    "trace_only",
    "trilog_global",
    "trilog_anchor",
    "trilog_residual",
    "trilog_full",
    "trilog_full_residual",
    "trilog_global_staged",
    "trilog_full_staged",
)


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _binary_metrics(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return 0.5, 0.0
    from sklearn.metrics import log_loss, roc_auc_score

    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return float(roc_auc_score(labels, clipped)), float(log_loss(labels, clipped))


def _flat_probabilities(model_pkg: dict, matrix: np.ndarray) -> np.ndarray:
    if model_pkg.get("model_type") == "theta_trilog_mlp":
        import theta_trilog_model as trilog_model

        return trilog_model.predict_trilog_pair_model(model_pkg, matrix)
    model = model_pkg["model"]
    scaler = model_pkg.get("scaler")
    effective = scaler.transform(matrix) if scaler is not None else matrix
    if hasattr(model, "predict_proba"):
        return model.predict_proba(effective)[:, 1].astype(np.float32)
    return np.clip(model.predict(effective).astype(np.float32), 1e-6, 1.0 - 1e-6)


def _probability_matrix(pairs: Sequence[tuple[int, int]], values: np.ndarray, case_count: int) -> np.ndarray:
    output = np.eye(case_count, dtype=np.float32)
    for (left, right), value in zip(pairs, values):
        output[left, right] = output[right, left] = float(value)
    return output


def _pair_labels(labels: Sequence[str], pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.asarray([float(labels[left] == labels[right]) for left, right in pairs], dtype=np.float32)


def _config_matrix(
    config: str,
    base: np.ndarray,
    trace_global: np.ndarray,
    trace_anchor: np.ndarray,
    trace_residual: np.ndarray,
    trace_full: np.ndarray,
    trace_full_residual: np.ndarray,
) -> np.ndarray:
    config = config.removesuffix("_staged")
    if config == "no_trace":
        return base
    if config == "trace_only":
        return trace_full
    if config == "trilog_global":
        return np.hstack([base, trace_global]).astype(np.float32, copy=False)
    if config == "trilog_anchor":
        return np.hstack([base, trace_anchor]).astype(np.float32, copy=False)
    if config == "trilog_residual":
        return np.hstack([base, trace_residual]).astype(np.float32, copy=False)
    if config == "trilog_full":
        return np.hstack([base, trace_full]).astype(np.float32, copy=False)
    if config == "trilog_full_residual":
        return np.hstack([base, trace_full_residual]).astype(np.float32, copy=False)
    raise ValueError(f"unsupported TriLog config: {config}")


def _official_pair_mask(pairs: Sequence[tuple[int, int]], slices: Sequence[dict]) -> np.ndarray:
    official_indices: set[int] = set()
    for episode in slices:
        if episode["name"] in {"benchmark_set_1", "benchmark_set_2"}:
            official_indices.update(range(int(episode["start"]), int(episode["stop"])))
    return np.asarray(
        [left in official_indices and right in official_indices for left, right in pairs],
        dtype=bool,
    )


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def _train_staged_trilog(
    train_matrix: np.ndarray,
    hold_matrix: np.ndarray,
    train_trace: np.ndarray,
    hold_trace: np.ndarray,
    train_base: np.ndarray,
    hold_base: np.ndarray,
    train_y: np.ndarray,
    sample_weight: np.ndarray,
    official_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Fake pretraining followed by low-capacity official adaptation."""
    fake_mask = ~official_mask
    if int(np.sum(fake_mask)) == 0:
        raise ValueError("staged TriLog requires at least one fake training episode")
    fake_weights = np.asarray(sample_weight[fake_mask], dtype=np.float32)
    fake_weights /= max(float(np.mean(fake_weights)), 1e-12)
    teacher = gm.train_view_model(
        train_matrix[fake_mask],
        train_y[fake_mask],
        "gbdt",
        seed,
        sample_weight=fake_weights,
    )
    train_teacher = _flat_probabilities(teacher, train_matrix)
    hold_teacher = _flat_probabilities(teacher, hold_matrix)
    # Keep only the compact structured/deterministic tail from the no-trace
    # block; the high-dimensional sim/regr signal is already represented by
    # the frozen teacher probability.
    structured_dim = min(64, train_base.shape[1]) if len(train_y) else 0
    train_tail = train_base[:, -structured_dim:] if structured_dim else np.zeros((len(train_y), 0), dtype=np.float32)
    hold_tail = hold_base[:, -structured_dim:] if structured_dim else np.zeros((len(hold_teacher), 0), dtype=np.float32)
    adapter_train = np.hstack([_logit(train_teacher)[:, None], train_trace, train_tail]).astype(np.float32)
    adapter_hold = np.hstack([_logit(hold_teacher)[:, None], hold_trace, hold_tail]).astype(np.float32)

    from sklearn.ensemble import HistGradientBoostingClassifier

    adapter = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=3,
        learning_rate=0.05,
        l2_regularization=2.0,
        early_stopping=False,
        class_weight="balanced",
        random_state=seed + 7919,
    )
    adapter.fit(adapter_train, train_y, sample_weight=sample_weight)
    probabilities = adapter.predict_proba(adapter_hold)[:, 1].astype(np.float32)
    return probabilities, {
        "fake_pretrain_pairs": int(np.sum(fake_mask)),
        "official_adapt_pairs": int(np.sum(official_mask)),
        "adapter_feature_dim": int(adapter_train.shape[1]),
        "teacher_train_mean": float(np.mean(train_teacher)),
        "teacher_hold_mean": float(np.mean(hold_teacher)),
    }


def summarize(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["config"]), str(row["model_type"]), str(row["graph_method"]))].append(row)
    output: list[dict] = []
    for (config, model_type, graph_method), values in grouped.items():
        dataset_means: dict[str, float] = {}
        for dataset in sorted({str(value["dataset"]) for value in values}):
            selected = [float(value["BA"]) for value in values if value["dataset"] == dataset]
            dataset_means[dataset] = float(np.mean(selected))
        bas = list(dataset_means.values())
        official = [dataset_means[name] for name in ("benchmark_set_1", "benchmark_set_2") if name in dataset_means]
        output.append({
            "config": config,
            "model_type": model_type,
            "graph_method": graph_method,
            "mean_BA": float(np.mean(bas)) if bas else 0.0,
            "worst_BA": float(np.min(bas)) if bas else 0.0,
            "official_mean_BA": float(np.mean(official)) if official else 0.0,
            "mean_TPR": float(np.mean([float(value["TPR"]) for value in values])),
            "mean_TNR": float(np.mean([float(value["TNR"]) for value in values])),
            "mean_pair_AUC": float(np.mean([float(value["pair_AUC"]) for value in values])),
            "mean_pair_BCE": float(np.mean([float(value["pair_BCE"]) for value in values])),
            "runs": len(values),
            "dataset_means": json.dumps(dataset_means, sort_keys=True),
        })
    return sorted(output, key=lambda row: (row["official_mean_BA"], row["mean_BA"], row["worst_BA"]), reverse=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Theta v3 hierarchical three-log LODO experiments")
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--holdout-datasets", nargs="*", default=[])
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument(
        "--model-types", nargs="+", choices=("gbdt", "logistic", "trilog_mlp"),
        default=["gbdt"],
    )
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
    parser.add_argument("--view-connectivity-positive-fraction", type=float, default=0.0)
    parser.add_argument("--view-connectivity-positive-weight", type=float, default=1.0)
    parser.add_argument("--view-dataset-balance-power", type=float, default=1.0)
    parser.add_argument("--view-official-weight", type=float, default=4.0)
    parser.add_argument("--view-device", default="cpu")
    parser.add_argument("--view-epochs", type=int, default=24)
    parser.add_argument("--view-batch-size", type=int, default=4096)
    parser.add_argument("--view-lr", type=float, default=1e-3)
    parser.add_argument("--view-weight-decay", type=float, default=1e-4)
    parser.add_argument("--view-dropout", type=float, default=0.2)
    parser.add_argument("--view-early-stop-patience", type=int, default=6)
    parser.add_argument("--view-focal-gamma", type=float, default=2.0)
    parser.add_argument("--parser", default="drain")
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--llm-doc-max-features", type=int, default=80)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    parser.add_argument("--llm-batch-size", type=int, default=64)
    parser.add_argument("--llm-timeout-sec", type=float, default=20.0)
    parser.add_argument("--embedding-cache-only", action="store_true")
    parser.add_argument("--embedding-expected-dim", type=int, default=768)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = resolve(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.embedding_cache_only:
        rfb.fetch_llm_embeddings = gm.fetch_cached_llm_embeddings
    datasets = [resolve(path) for path in args.datasets]
    for dataset in datasets:
        if not (dataset / "input.csv").exists():
            raise FileNotFoundError(dataset / "input.csv")

    llm_args = gm.make_embedding_args(args)
    base_features: list[plf.LLMCaseFeature] = []
    slices: list[dict] = []
    trace_features: list[ttf.HierarchicalTraceFeature] = []
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
        print(
            f"[trilog-trace] dataset={dataset.name} cases={len(episode_trace)} "
            f"ok={sum(feature.has_trace for feature in episode_trace)} "
            f"cache_hits={sum(int(row['cache_hit']) for row in episode_debug)}",
            flush=True,
        )
    trace_parse_runtime = time.perf_counter() - trace_started
    write_csv(args.output_dir / "trace_debug.csv", trace_debug, sorted({key for row in trace_debug for key in row}))

    raw_custom = {
        name: gm.fetch_view_embeddings(documents, args, name)
        for name, documents in custom_documents.items()
    }
    rows: list[dict] = []
    for seed in args.seeds:
        for holdout in slices:
            if args.holdout_datasets and holdout["name"] not in args.holdout_datasets:
                continue
            fold_started = time.perf_counter()
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
            view_names = gm.views_for_config("quad_event_object_context")
            train_base = gm.build_multiview_pair_feature_matrix(base_features, reduced_custom, view_names, train_pairs)
            hold_custom = {name: matrix[hold_indices] for name, matrix in reduced_custom.items()}
            hold_base = gm.build_multiview_pair_feature_matrix(hold_base_features, hold_custom, view_names, hold_pairs)
            train_trace_components = ttf.build_trace_pair_feature_components(trace_features, trace_matrices, train_pairs)
            train_trace_global = train_trace_components["global"]
            train_trace_anchor = train_trace_components["anchor"]
            train_trace_residual = train_trace_components["residual"]
            train_trace_full = train_trace_components["full"]
            train_trace_full_residual = train_trace_components["full_residual"]
            hold_trace_features = [trace_features[index] for index in hold_indices]
            hold_trace_matrices = {name: matrix[hold_indices] for name, matrix in trace_matrices.items()}
            hold_trace_components = ttf.build_trace_pair_feature_components(hold_trace_features, hold_trace_matrices, hold_pairs)
            hold_trace_global = hold_trace_components["global"]
            hold_trace_anchor = hold_trace_components["anchor"]
            hold_trace_residual = hold_trace_components["residual"]
            hold_trace_full = hold_trace_components["full"]
            hold_trace_full_residual = hold_trace_components["full_residual"]
            conflict = gm.conflict_matrix_from_records(
                osf.build_case_records(holdout["name"], holdout["path"] / "input.csv", gold_csv=None)
            )
            k = len(set(holdout["labels"]))

            for config in args.configs:
                train_matrix = _config_matrix(
                    config,
                    train_base,
                    train_trace_global,
                    train_trace_anchor,
                    train_trace_residual,
                    train_trace_full,
                    train_trace_full_residual,
                )
                hold_matrix = _config_matrix(
                    config,
                    hold_base,
                    hold_trace_global,
                    hold_trace_anchor,
                    hold_trace_residual,
                    hold_trace_full,
                    hold_trace_full_residual,
                )
                for model_type in args.model_types:
                    model_started = time.perf_counter()
                    staged_debug: dict = {}
                    if config.endswith("_staged"):
                        if model_type != "gbdt":
                            raise ValueError("staged TriLog currently requires model_type=gbdt")
                        adapter_train_trace = train_trace_global if config.startswith("trilog_global") else train_trace_full
                        adapter_hold_trace = hold_trace_global if config.startswith("trilog_global") else hold_trace_full
                        flat, staged_debug = _train_staged_trilog(
                            train_matrix,
                            hold_matrix,
                            adapter_train_trace,
                            adapter_hold_trace,
                            train_base,
                            hold_base,
                            train_y,
                            sample_weight,
                            official_pair_mask,
                            seed,
                        )
                    elif model_type == "trilog_mlp":
                        import theta_trilog_model as trilog_model

                        if config == "trace_only":
                            base_dim = 0
                        else:
                            base_dim = int(train_base.shape[1])
                        neural_args = argparse.Namespace(
                            random_state=seed,
                            device=args.view_device,
                            epochs=args.view_epochs,
                            batch_size=args.view_batch_size,
                            lr=args.view_lr,
                            weight_decay=args.view_weight_decay,
                            dropout=args.view_dropout,
                            early_stop_patience=args.view_early_stop_patience,
                            focal_gamma=args.view_focal_gamma,
                        )
                        model_pkg = trilog_model.train_trilog_pair_model(
                            train_matrix, train_y, sample_weight, base_dim, neural_args
                        )
                        flat = _flat_probabilities(model_pkg, hold_matrix)
                        staged_debug = {
                            "neural_best_epoch": model_pkg["best_epoch"],
                            "neural_best_validation_loss": model_pkg["best_validation_loss"],
                            "neural_device": model_pkg["device_used"],
                        }
                    else:
                        model_pkg = gm.train_view_model(
                            train_matrix,
                            train_y,
                            model_type,
                            seed,
                            sample_weight=sample_weight,
                            train_args=args,
                        )
                        flat = _flat_probabilities(model_pkg, hold_matrix)
                    pair_auc, pair_bce = _binary_metrics(flat, hold_y)
                    probability = _probability_matrix(hold_pairs, flat, len(hold_indices))
                    clustered = gc.cluster_probability_graph(
                        probability,
                        k,
                        args.graph_method,
                        conflict_matrix=conflict,
                    )
                    runtime = time.perf_counter() - model_started
                    pred_path = args.output_dir / "preds" / f"{holdout['name']}_{config}_{model_type}_{args.graph_method}_seed{seed}.csv"
                    pred = write_pred(pred_path, holdout["cases"], clustered.labels)
                    prob_path = args.output_dir / "probs" / f"{holdout['name']}_{config}_{model_type}_{args.graph_method}_seed{seed}.npy"
                    prob_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(prob_path, probability)
                    ba, tpr, tnr = pairwise_scores(holdout["labels"], pred)
                    row = {
                        "dataset": holdout["name"],
                        "seed": seed,
                        "config": config,
                        "model_type": model_type,
                        "graph_method": args.graph_method,
                        "BA": ba,
                        "TPR": tpr,
                        "TNR": tnr,
                        "pair_AUC": pair_auc,
                        "pair_BCE": pair_bce,
                        "k": k,
                        "cases": len(hold_indices),
                        "num_clusters": len(set(pred)),
                        "feature_dim": int(train_matrix.shape[1]),
                        "train_pairs": len(train_y),
                        "runtime_sec": runtime,
                        "fold_prepare_runtime_sec": time.perf_counter() - fold_started,
                        "trace_parse_runtime_sec": trace_parse_runtime,
                        "pred_path": str(pred_path),
                        "prob_path": str(prob_path),
                        "notes": json.dumps({"pair_stats": pair_stats, "staged": staged_debug}, sort_keys=True),
                    }
                    rows.append(row)
                    print(
                        f"[trilog] dataset={holdout['name']} seed={seed} config={config} "
                        f"model={model_type} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f} "
                        f"AUC={pair_auc:.6f} dim={train_matrix.shape[1]}",
                        flush=True,
                    )

    fields = [
        "dataset", "seed", "config", "model_type", "graph_method", "BA", "TPR", "TNR",
        "pair_AUC", "pair_BCE", "k", "cases", "num_clusters", "feature_dim", "train_pairs",
        "runtime_sec", "fold_prepare_runtime_sec", "trace_parse_runtime_sec", "pred_path", "prob_path", "notes",
    ]
    write_csv(args.output_dir / "results.csv", rows, fields)
    summary = summarize(rows)
    summary_fields = [
        "config", "model_type", "graph_method", "mean_BA", "worst_BA", "official_mean_BA",
        "mean_TPR", "mean_TNR", "mean_pair_AUC", "mean_pair_BCE", "runs", "dataset_means",
    ]
    write_csv(args.output_dir / "summary.csv", summary, summary_fields)
    manifest = {
        "datasets": [str(dataset) for dataset in datasets],
        "seeds": args.seeds,
        "holdout_datasets": args.holdout_datasets,
        "configs": args.configs,
        "model_types": args.model_types,
        "graph_method": args.graph_method,
        "trace": {
            "segment_count": args.trace_segment_count,
            "chunk_size": args.trace_chunk_size,
            "anchor_sizes": args.trace_anchor_sizes,
            "global_struct_dim": args.trace_global_struct_dim,
            "global_text_dim": args.trace_global_text_dim,
            "anchor_struct_dim": args.trace_anchor_struct_dim,
            "anchor_text_dim": args.trace_anchor_text_dim,
            "residual_struct_dim": args.trace_residual_struct_dim,
            "residual_text_dim": args.trace_residual_text_dim,
        },
        "training": {
            "negative_ratio": args.view_negative_ratio,
            "hard_negative_ratio": args.view_hard_negative_ratio,
            "hard_positive_ratio": args.view_hard_positive_ratio,
            "connectivity_positive_fraction": args.view_connectivity_positive_fraction,
            "connectivity_positive_weight": args.view_connectivity_positive_weight,
            "dataset_balance_power": args.view_dataset_balance_power,
            "official_weight": args.view_official_weight,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n| rank | config | model | official mean | macro BA | worst BA | pair AUC |")
    print("|---:|---|---|---:|---:|---:|---:|")
    for rank, row in enumerate(summary, 1):
        print(
            f"| {rank} | {row['config']} | {row['model_type']} | {row['official_mean_BA']:.4f} | "
            f"{row['mean_BA']:.4f} | {row['worst_BA']:.4f} | {row['mean_pair_AUC']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
