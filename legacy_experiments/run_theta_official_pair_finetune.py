#!/usr/bin/env python3
"""Strict pair-probability adaptation experiments for Theta v4 TriLog.

Fake episodes pretrain the always-on sim/regr/trace two-tower. Public official
pair labels then adapt only a constrained calibration/head. Every target is a
complete held-out episode; target gold is used only for final metrics.
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
import theta_trilog_model as ttm
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred
from run_theta_trilog_lodo import (
    DEFAULT_DATASETS,
    _binary_metrics,
    _config_matrix,
    _pair_labels,
    _probability_matrix,
    resolve,
)


OFFICIAL_NAMES = {"benchmark_set_1", "benchmark_set_2"}
VARIANTS = ("fake_pretrain", "official_affine", "official_last_rank", "official_head_connect")
CLUSTERERS = ("average", "quality_selected")


def _case_episode_names(slices: Sequence[dict], case_count: int) -> list[str]:
    output = [""] * case_count
    for episode in slices:
        for index in range(int(episode["start"]), int(episode["stop"])):
            output[index] = str(episode["name"])
    if any(not name for name in output):
        raise ValueError("episode case ranges do not cover all cases")
    return output


def _pair_episode_names(
    pairs: Sequence[tuple[int, int]], case_episode: Sequence[str],
) -> list[str]:
    output: list[str] = []
    for left, right in pairs:
        left_name = case_episode[left]
        right_name = case_episode[right]
        if left_name != right_name:
            raise RuntimeError(f"cross-episode pair detected: {left_name} != {right_name}")
        output.append(left_name)
    return output


def _probability_diagnostics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss

    probabilities = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(labels, dtype=np.float32)
    positive = probabilities[labels > 0.5]
    negative = probabilities[labels < 0.5]

    def quantile(values: np.ndarray, value: float) -> float:
        return float(np.quantile(values, value)) if len(values) else float("nan")

    pair_auc, _ = _binary_metrics(probabilities, labels)
    positive_pred = probabilities >= 0.5
    return {
        "pair_AUC": pair_auc,
        "pair_BCE": float(log_loss(labels, probabilities, labels=[0.0, 1.0])),
        "pair_Brier": float(brier_score_loss(labels, probabilities)),
        "pair_threshold_TPR": float(np.mean(positive_pred[labels > 0.5])) if len(positive) else 0.0,
        "pair_threshold_TNR": float(np.mean(~positive_pred[labels < 0.5])) if len(negative) else 0.0,
        "positive_p10": quantile(positive, 0.10),
        "positive_p50": quantile(positive, 0.50),
        "positive_p90": quantile(positive, 0.90),
        "negative_p50": quantile(negative, 0.50),
        "negative_p90": quantile(negative, 0.90),
        "negative_p99": quantile(negative, 0.99),
        "probability_mean": float(np.mean(probabilities)),
        "probability_std": float(np.std(probabilities)),
    }


def _hard_pair_rows(
    dataset: str,
    seed: int,
    variant: str,
    cases: Sequence[str],
    pairs: Sequence[tuple[int, int]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    limit: int = 50,
) -> list[dict]:
    candidates: list[tuple[float, str, int]] = []
    for row, (label, probability) in enumerate(zip(labels, probabilities)):
        kind = "hard_positive" if label > 0.5 else "hard_negative"
        hardness = 1.0 - float(probability) if label > 0.5 else float(probability)
        candidates.append((hardness, kind, row))
    selected: list[tuple[float, str, int]] = []
    for kind in ("hard_positive", "hard_negative"):
        selected.extend(sorted(
            (item for item in candidates if item[1] == kind), reverse=True
        )[:limit])
    output: list[dict] = []
    for hardness, kind, row in selected:
        left, right = pairs[row]
        output.append({
            "dataset": dataset,
            "seed": seed,
            "variant": variant,
            "kind": kind,
            "case_left": cases[left],
            "case_right": cases[right],
            "gold_same": int(labels[row] > 0.5),
            "probability": float(probabilities[row]),
            "hardness": hardness,
        })
    return output


def _adaptation_args(args: argparse.Namespace, seed: int, **overrides) -> argparse.Namespace:
    values = {
        "random_state": seed,
        "device": args.device,
        "finetune_scope": "last",
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "finetune_weight_decay": args.finetune_weight_decay,
        "label_smoothing": args.label_smoothing,
        "affine_reg": args.affine_reg,
        "ranking_weight": 0.0,
        "ranking_margin": args.ranking_margin,
        "connectivity_weight": 0.0,
        "connectivity_top_m": args.connectivity_top_m,
        "transitivity_weight": 0.0,
        "replay_weight": args.replay_weight,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _summarize(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["clusterer"]))].append(row)
    output: list[dict] = []
    for (variant, clusterer), values in grouped.items():
        dataset_means = {
            dataset: float(np.mean([
                float(row["BA"]) for row in values if row["dataset"] == dataset
            ]))
            for dataset in sorted({str(row["dataset"]) for row in values})
        }
        official = [value for dataset, value in dataset_means.items() if dataset in OFFICIAL_NAMES]
        output.append({
            "variant": variant,
            "clusterer": clusterer,
            "mean_BA": float(np.mean(list(dataset_means.values()))),
            "official_mean_BA": float(np.mean(official)) if official else 0.0,
            "worst_BA": float(np.min(list(dataset_means.values()))),
            "mean_pair_AUC": float(np.mean([float(row["pair_AUC"]) for row in values])),
            "mean_pair_Brier": float(np.mean([float(row["pair_Brier"]) for row in values])),
            "runs": len(values),
            "dataset_means": json.dumps(dataset_means, sort_keys=True),
        })
    return sorted(output, key=lambda row: (row["official_mean_BA"], row["mean_BA"]), reverse=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--holdout-datasets", nargs="*", default=["benchmark_set_1", "benchmark_set_2"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--clusterers", nargs="+", choices=CLUSTERERS, default=["average"])
    parser.add_argument("--view-dim", type=int, default=64)
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--parser", default="drain")
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    parser.add_argument("--llm-batch-size", type=int, default=64)
    parser.add_argument("--llm-timeout-sec", type=float, default=20.0)
    parser.add_argument("--llm-doc-max-features", type=int, default=80)
    parser.add_argument("--embedding-cache-only", action="store_true")
    parser.add_argument("--embedding-expected-dim", type=int, default=768)
    parser.add_argument("--trace-cache-dir", type=Path, default=Path("/tmp/theta_trilog_trace_cache"))
    parser.add_argument("--trace-segment-count", type=int, default=16)
    parser.add_argument("--trace-chunk-size", type=int, default=512)
    parser.add_argument("--trace-anchor-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--trace-global-struct-dim", type=int, default=48)
    parser.add_argument("--trace-global-text-dim", type=int, default=48)
    parser.add_argument("--trace-anchor-struct-dim", type=int, default=32)
    parser.add_argument("--trace-anchor-text-dim", type=int, default=32)
    parser.add_argument("--trace-residual-struct-dim", type=int, default=48)
    parser.add_argument("--trace-residual-text-dim", type=int, default=48)
    parser.add_argument("--view-max-pairs-per-dataset", type=int, default=30000)
    parser.add_argument("--view-negative-ratio", type=float, default=2.0)
    parser.add_argument("--view-hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--view-hard-positive-ratio", type=float, default=1.0)
    parser.add_argument("--view-connectivity-positive-fraction", type=float, default=0.0)
    parser.add_argument("--view-connectivity-positive-weight", type=float, default=1.0)
    parser.add_argument("--view-dataset-balance-power", type=float, default=1.0)
    parser.add_argument("--view-official-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=4096)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrain-dropout", type=float, default=0.2)
    parser.add_argument("--pretrain-patience", type=int, default=6)
    parser.add_argument("--pretrain-focal-gamma", type=float, default=2.0)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--affine-reg", type=float, default=0.20)
    parser.add_argument("--ranking-weight", type=float, default=0.20)
    parser.add_argument("--ranking-margin", type=float, default=0.50)
    parser.add_argument("--connectivity-weight", type=float, default=0.10)
    parser.add_argument("--connectivity-top-m", type=int, default=2)
    parser.add_argument("--transitivity-weight", type=float, default=0.05)
    parser.add_argument("--replay-weight", type=float, default=0.30)
    parser.add_argument("--replay-pairs", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [resolve(path) for path in args.datasets]
    if args.embedding_cache_only:
        rfb.fetch_llm_embeddings = gm.fetch_cached_llm_embeddings
    llm_args = gm.make_embedding_args(args)

    base_features: list[plf.LLMCaseFeature] = []
    trace_features: list[ttf.HierarchicalTraceFeature] = []
    custom_documents = {name: [] for name in ("event", "object", "context")}
    slices: list[dict] = []
    offset = 0
    for dataset in datasets:
        episode_features, _ = plf.build_llm_case_features_for_inputs(
            [dataset / "input.csv"], parser=args.parser, svd_dim=args.svd_dim, llm_args=llm_args
        )
        labels = read_gold(osf.gold_path(dataset))
        cases = osf.read_cases(dataset / "input.csv")
        if not (len(episode_features) == len(labels) == len(cases)):
            raise RuntimeError(f"feature/label mismatch for {dataset}")
        base_features.extend(episode_features)
        slices.append({
            "name": dataset.name, "path": dataset, "start": offset,
            "stop": offset + len(labels), "labels": labels, "cases": cases,
        })
        offset += len(labels)
        documents = gm.build_all_view_documents(dataset)
        for name in custom_documents:
            custom_documents[name].extend(documents[name])
        episode_trace, _ = ttf.build_hierarchical_trace_features(
            dataset / "input.csv", cache_dir=args.trace_cache_dir,
            segment_count=args.trace_segment_count, chunk_size=args.trace_chunk_size,
            anchor_sizes=args.trace_anchor_sizes,
        )
        trace_features.extend(episode_trace)
        print(
            f"[theta-v4-load] dataset={dataset.name} cases={len(labels)} "
            f"trace_ok={sum(feature.has_trace for feature in episode_trace)}",
            flush=True,
        )
    raw_custom = {
        name: gm.fetch_view_embeddings(documents, args, name)
        for name, documents in custom_documents.items()
    }
    case_episode = _case_episode_names(slices, len(base_features))
    rows: list[dict] = []
    hard_rows: list[dict] = []

    for seed in args.seeds:
        for holdout in slices:
            if args.holdout_datasets and holdout["name"] not in args.holdout_datasets:
                continue
            started = time.perf_counter()
            train_indices = [
                index for episode in slices if episode["name"] != holdout["name"]
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
                name: gm.fit_apply_reduced_matrix(raw, train_indices, args.view_dim, seed + 101 + pos * 13)
                for pos, (name, raw) in enumerate(raw_custom.items())
            }
            _, trace_matrices = ttf.fit_transform_trace_views(
                trace_features, train_indices, seed=seed,
                global_struct_dim=args.trace_global_struct_dim,
                global_text_dim=args.trace_global_text_dim,
                anchor_struct_dim=args.trace_anchor_struct_dim,
                anchor_text_dim=args.trace_anchor_text_dim,
                residual_struct_dim=args.trace_residual_struct_dim,
                residual_text_dim=args.trace_residual_text_dim,
            )
            train_pairs, train_y, train_weight, pair_stats = gm.sample_lodo_train_pairs(
                base_features, slices, holdout["name"], args, seed
            )
            pair_episode = _pair_episode_names(train_pairs, case_episode)
            fake_mask = np.asarray([name not in OFFICIAL_NAMES for name in pair_episode], dtype=bool)
            official_mask = np.asarray([name in OFFICIAL_NAMES for name in pair_episode], dtype=bool)
            if not np.any(fake_mask):
                raise RuntimeError(f"no fake pretraining pairs for {holdout['name']}")
            official_sources = sorted(set(name for name in pair_episode if name in OFFICIAL_NAMES))

            hold_pairs = osf.all_pairs(len(hold_indices))
            hold_y = _pair_labels(holdout["labels"], hold_pairs)
            view_names = gm.views_for_config("quad_event_object_context")
            train_base = gm.build_multiview_pair_feature_matrix(
                base_features, reduced_custom, view_names, train_pairs
            )
            hold_custom = {name: matrix[hold_indices] for name, matrix in reduced_custom.items()}
            hold_base = gm.build_multiview_pair_feature_matrix(
                hold_base_features, hold_custom, view_names, hold_pairs
            )
            train_trace = ttf.build_trace_pair_feature_components(
                trace_features, trace_matrices, train_pairs
            )["residual"]
            hold_trace_features = [trace_features[index] for index in hold_indices]
            hold_trace_matrices = {name: matrix[hold_indices] for name, matrix in trace_matrices.items()}
            hold_trace = ttf.build_trace_pair_feature_components(
                hold_trace_features, hold_trace_matrices, hold_pairs
            )["residual"]
            train_matrix = _config_matrix(
                "trilog_residual", train_base,
                np.zeros((len(train_pairs), 0), dtype=np.float32),
                np.zeros((len(train_pairs), 0), dtype=np.float32),
                train_trace, np.zeros((len(train_pairs), 0), dtype=np.float32),
                np.zeros((len(train_pairs), 0), dtype=np.float32),
            )
            hold_matrix = _config_matrix(
                "trilog_residual", hold_base,
                np.zeros((len(hold_pairs), 0), dtype=np.float32),
                np.zeros((len(hold_pairs), 0), dtype=np.float32),
                hold_trace, np.zeros((len(hold_pairs), 0), dtype=np.float32),
                np.zeros((len(hold_pairs), 0), dtype=np.float32),
            )
            pretrain_args = argparse.Namespace(
                random_state=seed, device=args.device, epochs=args.pretrain_epochs,
                batch_size=args.pretrain_batch_size, lr=args.pretrain_lr,
                weight_decay=args.pretrain_weight_decay, dropout=args.pretrain_dropout,
                early_stop_patience=args.pretrain_patience,
                focal_gamma=args.pretrain_focal_gamma,
            )
            fake_weight = train_weight[fake_mask].copy()
            fake_weight /= max(float(np.mean(fake_weight)), 1e-12)
            pretrain_model = ttm.train_trilog_pair_model(
                train_matrix[fake_mask], train_y[fake_mask], fake_weight,
                int(train_base.shape[1]), pretrain_args,
            )
            rng = np.random.default_rng(seed * 1009 + int(holdout["start"]))
            fake_rows = np.flatnonzero(fake_mask)
            replay_count = min(int(args.replay_pairs), len(fake_rows))
            replay_rows = rng.choice(fake_rows, size=replay_count, replace=False)
            official_rows = np.flatnonzero(official_mask)
            official_pairs = [train_pairs[int(row)] for row in official_rows]

            packages: dict[str, dict] = {"fake_pretrain": pretrain_model}
            if len(official_rows):
                common = dict(
                    official_matrix=train_matrix[official_rows],
                    official_labels=train_y[official_rows],
                    official_weight=train_weight[official_rows],
                )
                packages["official_affine"] = ttm.fit_official_affine_calibration(
                    pretrain_model, args=_adaptation_args(args, seed), **common
                )
                packages["official_last_rank"] = ttm.fine_tune_official_pair_model(
                    pretrain_model, official_pairs=official_pairs,
                    replay_matrix=train_matrix[replay_rows],
                    args=_adaptation_args(
                        args, seed, finetune_scope="last",
                        ranking_weight=args.ranking_weight,
                    ), **common,
                )
                packages["official_head_connect"] = ttm.fine_tune_official_pair_model(
                    pretrain_model, official_pairs=official_pairs,
                    replay_matrix=train_matrix[replay_rows],
                    args=_adaptation_args(
                        args, seed, finetune_scope="head",
                        ranking_weight=args.ranking_weight,
                        connectivity_weight=args.connectivity_weight,
                        transitivity_weight=args.transitivity_weight,
                    ), **common,
                )
            for variant in args.variants:
                if variant not in packages:
                    print(f"[theta-v4] skip {variant}: no official source pairs", flush=True)
                    continue
                flat = ttm.predict_trilog_pair_model(packages[variant], hold_matrix)
                probability = _probability_matrix(hold_pairs, flat, len(hold_indices))
                prob_path = output_dir / "probs" / f"{holdout['name']}_{variant}_seed{seed}.npy"
                prob_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(prob_path, probability)
                diagnostics = _probability_diagnostics(flat, hold_y)
                for clusterer in args.clusterers:
                    if clusterer == "average":
                        clustered = gc.agglomerative_avg(
                            probability, len(set(holdout["labels"]))
                        )
                    else:
                        clustered = gc.quality_selected_clustering(
                            probability, len(set(holdout["labels"]))
                        )
                    pred_path = (
                        output_dir / "preds"
                        / f"{holdout['name']}_{variant}_{clusterer}_seed{seed}.csv"
                    )
                    pred = write_pred(pred_path, holdout["cases"], clustered.labels)
                    ba, tpr, tnr = pairwise_scores(holdout["labels"], pred)
                    row = {
                        "dataset": holdout["name"], "seed": seed, "variant": variant,
                        "clusterer": clusterer,
                        "selected_clusterer": clustered.trajectory[-1].get("candidate", "average")
                        if clustered.trajectory else "average",
                        "official_sources": ";".join(official_sources),
                        "BA": ba, "TPR": tpr, "TNR": tnr,
                        **diagnostics,
                        "cases": len(hold_indices), "k": len(set(holdout["labels"])),
                        "fake_pretrain_pairs": int(np.sum(fake_mask)),
                        "official_finetune_pairs": int(np.sum(official_mask)),
                        "replay_pairs": replay_count,
                        "input_dim": int(train_matrix.shape[1]),
                        "runtime_sec": time.perf_counter() - started,
                        "pred_path": str(pred_path), "prob_path": str(prob_path),
                        "pair_stats": json.dumps(pair_stats, sort_keys=True),
                        "cluster_trajectory": json.dumps(clustered.trajectory, sort_keys=True),
                    }
                    rows.append(row)
                    print(
                        f"[theta-v4] target={holdout['name']} source={official_sources} "
                        f"seed={seed} variant={variant} clusterer={clusterer} "
                        f"BA={ba:.6f} AUC={diagnostics['pair_AUC']:.6f} "
                        f"Brier={diagnostics['pair_Brier']:.6f} "
                        f"pos50={diagnostics['positive_p50']:.3f} "
                        f"neg90={diagnostics['negative_p90']:.3f}",
                        flush=True,
                    )
                hard_rows.extend(_hard_pair_rows(
                    holdout["name"], seed, variant, holdout["cases"],
                    hold_pairs, hold_y, flat,
                ))

    if not rows:
        raise RuntimeError("Theta v4 produced no results")
    write_csv(output_dir / "results.csv", rows, list(rows[0]))
    write_csv(output_dir / "hard_pairs.csv", hard_rows, list(hard_rows[0]))
    summary = _summarize(rows)
    write_csv(output_dir / "summary.csv", summary, list(summary[0]))
    manifest = {key: value for key, value in vars(args).items()}
    manifest["datasets"] = [str(resolve(path)) for path in args.datasets]
    manifest["output_dir"] = str(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print("\n| rank | variant | clusterer | official mean | macro BA | worst BA | pair AUC | Brier |")
    print("|---:|---|---|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(summary, 1):
        print(
            f"| {rank} | {row['variant']} | {row['clusterer']} | "
            f"{row['official_mean_BA']:.4f} | "
            f"{row['mean_BA']:.4f} | {row['worst_BA']:.4f} | "
            f"{row['mean_pair_AUC']:.4f} | {row['mean_pair_Brier']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
