#!/usr/bin/env python3
"""Trace/completion extensions for unified multi-dataset training.

Experimental only. This runner reuses the unified episodic objective while
adding anchor-window trace features and/or cached completion canonical JSON.
The formal prediction entry point is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import completion_case_features as ccf
import official_style_features as osf
import pairwise_llm_features as plf
import trace_anchor as ta
import run_unified_multidataset_experiments as ume
from run_experiments import pairwise_scores
from run_official_full_retrain_experiments import write_csv, write_pred


def _auxiliary_matrix(
    feature_set: str,
    trace_features: Sequence[ta.AnchorTraceFeature] | None,
    completion_features: Sequence[ccf.CompletionCaseFeature] | None,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray | None:
    blocks: list[np.ndarray] = []
    if "trace" in feature_set:
        if trace_features is None:
            raise RuntimeError("trace feature set requested without trace features")
        blocks.append(ta.build_anchor_trace_pair_feature_matrix(trace_features, pairs))
    if "completion" in feature_set:
        if completion_features is None:
            raise RuntimeError("completion feature set requested without completion features")
        blocks.append(ccf.build_completion_pair_feature_matrix(completion_features, pairs))
    return np.hstack(blocks).astype(np.float32, copy=False) if blocks else None


def _predict_probability(
    model_pkg: dict,
    features: list[plf.LLMCaseFeature],
    auxiliary: np.ndarray | None,
    batch_size: int,
) -> np.ndarray:
    import torch

    pairs = [(i, j) for i in range(len(features)) for j in range(i + 1, len(features))]
    matrix = plf.build_rich_pair_feature_matrix(
        features, pairs, feature_mode="llm_dual_struct_det_summary"
    )
    if auxiliary is not None:
        if len(auxiliary) != len(matrix):
            raise RuntimeError(f"auxiliary rows={len(auxiliary)} pair rows={len(matrix)}")
        matrix = np.hstack([matrix, auxiliary]).astype(np.float32, copy=False)
    matrix = model_pkg["scaler"].transform(matrix).astype(np.float32)
    model = model_pkg["model"]
    device = model_pkg["device"]
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            logits = model(torch.from_numpy(matrix[start:start + batch_size]).to(device))
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    flat = np.concatenate(scores).astype(np.float32) if scores else np.zeros(0, dtype=np.float32)
    probability = np.eye(len(features), dtype=np.float32)
    for (i, j), value in zip(pairs, flat):
        probability[i, j] = probability[j, i] = float(value)
    return probability


def run_fold(
    args: argparse.Namespace,
    datasets: list[Path],
    holdout_index: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    holdout = datasets[holdout_index]
    train_datasets = [dataset for idx, dataset in enumerate(datasets) if idx != holdout_index]
    ordered = train_datasets + [holdout]
    slices = ume.build_slices(ordered)
    train_stop = slices[-1].start

    llm_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=True,
    )
    all_features, _ = plf.build_llm_case_features_for_inputs(
        [dataset / "input.csv" for dataset in ordered],
        svd_dim=args.svd_dim,
        llm_args=llm_args,
    )
    train_features = all_features[:train_stop]
    holdout_features = all_features[train_stop:]
    llm_reducer = plf.fit_llm_reducer(
        train_features, args.llm_reduce_dim, random_state=seed
    )
    summary_reducer = plf.fit_llm_summary_reducer(
        train_features, args.llm_reduce_dim, random_state=seed
    )
    plf.apply_llm_reducer(holdout_features, llm_reducer, args.llm_reduce_dim)
    plf.apply_llm_summary_reducer(
        holdout_features, summary_reducer, args.llm_reduce_dim
    )

    pair_data = ume.build_pair_data(train_features, slices[:-1], args, seed)
    base_train = plf.build_rich_pair_feature_matrix(
        train_features,
        pair_data.pairs,
        feature_mode="llm_dual_struct_det_summary",
    )
    holdout_pairs = [
        (i, j) for i in range(len(holdout_features)) for j in range(i + 1, len(holdout_features))
    ]

    need_trace = any("trace" in feature_set for feature_set in args.feature_sets)
    need_completion = any("completion" in feature_set for feature_set in args.feature_sets)
    trace_all = None
    completion_all = None
    if need_trace:
        trace_all, trace_debug = ta.build_anchor_trace_case_features(
            [dataset / "input.csv" for dataset in ordered],
            window_size=args.trace_window_size,
        )
        write_csv(
            args.output_dir / "debug" / f"trace_{holdout.name}_seed{seed}.csv",
            trace_debug,
            list(trace_debug[0]) if trace_debug else ["case_id"],
        )
    if need_completion:
        completion_all, completion_debug = ccf.build_completion_case_features(
            [dataset / "input.csv" for dataset in ordered],
            cache_dir=args.completion_cache_dir,
            strict=args.strict_completion,
        )
        write_csv(
            args.output_dir / "debug" / f"completion_{holdout.name}_seed{seed}.csv",
            completion_debug,
            list(completion_debug[0]) if completion_debug else ["case_id"],
        )

    gold = slices[-1].labels
    k = len(set(gold))
    rows: list[dict] = []
    stats: list[dict] = []
    for feature_set in args.feature_sets:
        train_aux = _auxiliary_matrix(
            feature_set,
            trace_all[:train_stop] if trace_all is not None else None,
            completion_all[:train_stop] if completion_all is not None else None,
            pair_data.pairs,
        )
        holdout_aux = _auxiliary_matrix(
            feature_set,
            trace_all[train_stop:] if trace_all is not None else None,
            completion_all[train_stop:] if completion_all is not None else None,
            holdout_pairs,
        )
        train_aux_for_fit = train_aux
        if feature_set == "trace_dropout" and train_aux is not None:
            train_aux_for_fit = train_aux.copy()
            rng = np.random.default_rng(seed * 1009 + holdout_index * 97 + 31)
            drop_mask = rng.random(len(train_aux_for_fit)) < args.trace_feature_dropout
            train_aux_for_fit[drop_mask] = 0.0
        matrix = (
            np.hstack([base_train, train_aux_for_fit]).astype(np.float32, copy=False)
            if train_aux_for_fit is not None else base_train
        )
        print(
            f"[fold] holdout={holdout.name} seed={seed} feature_set={feature_set} "
            f"input_dim={matrix.shape[1]} pairs={len(matrix)} "
            f"triplets={len(pair_data.triplets)} triangles={len(pair_data.triangles)}",
            flush=True,
        )
        for config in args.configs:
            started = time.perf_counter()
            model_pkg = ume.train_unified_model(matrix, pair_data, args, config, seed)
            raw_probability = _predict_probability(
                model_pkg, holdout_features, holdout_aux, args.predict_batch_size
            )
            if feature_set == "trace_dropout" and holdout_aux is not None:
                missing_trace_probability = _predict_probability(
                    model_pkg,
                    holdout_features,
                    np.zeros_like(holdout_aux),
                    args.predict_batch_size,
                )
                raw_probability = (
                    args.trace_view_alpha * raw_probability
                    + (1.0 - args.trace_view_alpha) * missing_trace_probability
                ).astype(np.float32)
                np.fill_diagonal(raw_probability, 1.0)
            model_dir = args.output_dir / "models" / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"holdout_{holdout.name}_{feature_set}_{config}.pt"
            preproc_path = model_path.with_suffix(".preproc.pkl")
            import torch

            torch.save({
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model_pkg["model"].state_dict().items()
                },
                "input_dim": model_pkg["input_dim"],
                "width": model_pkg["width"],
                "representation_dim": model_pkg["representation_dim"],
                "dropout": model_pkg["dropout"],
                "config": config,
                "feature_set": feature_set,
            }, model_path)
            with preproc_path.open("wb") as file:
                pickle.dump({
                    "scaler": model_pkg["scaler"],
                    "llm_reducer": llm_reducer,
                    "llm_summary_reducer": summary_reducer,
                    "llm_reduce_dim": args.llm_reduce_dim,
                    "trace_window_size": args.trace_window_size,
                    "trace_feature_dropout": args.trace_feature_dropout,
                    "trace_view_alpha": args.trace_view_alpha,
                }, file)

            for gamma in args.graph_gammas:
                probability = ume.graph_refine_probability(raw_probability, gamma)
                labels = plf.cluster_from_probability(probability, k)
                method = f"{feature_set}_{config}_graph{gamma:g}"
                pred_path = (
                    args.output_dir / "preds"
                    / f"{holdout.name}_{method}_seed{seed}.csv"
                )
                prob_path = (
                    args.output_dir / "probs"
                    / f"{holdout.name}_{method}_seed{seed}.npy"
                )
                pred = write_pred(pred_path, slices[-1].cases, labels)
                prob_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(prob_path, probability)
                ba, tpr, tnr = pairwise_scores(gold, pred)
                rows.append({
                    "seed": seed,
                    "holdout_dataset": holdout.name,
                    "train_datasets": "+".join(dataset.name for dataset in train_datasets),
                    "feature_set": feature_set,
                    "config": config,
                    "graph_gamma": gamma,
                    "BA": ba,
                    "TPR": tpr,
                    "TNR": tnr,
                    "k": k,
                    "cases": len(gold),
                    "num_pred_clusters": len(set(pred)),
                    "input_dim": matrix.shape[1],
                    "best_val_loss": model_pkg["best_val_loss"],
                    "runtime_sec": time.perf_counter() - started,
                    "model_path": str(model_path),
                    "pred_path": str(pred_path),
                    "prob_path": str(prob_path),
                })
                print(
                    f"[eval] holdout={holdout.name} seed={seed} "
                    f"feature_set={feature_set} config={config} graph={gamma:g} "
                    f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                    flush=True,
                )
            for stat in pair_data.stats:
                stats.append({
                    "seed": seed,
                    "holdout_dataset": holdout.name,
                    "feature_set": feature_set,
                    "config": config,
                    **stat,
                })
    return rows, stats


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["feature_set"]), str(row["config"]), float(row["graph_gamma"]))
        ].append(row)
    output: list[dict] = []
    for (feature_set, config, gamma), values in groups.items():
        dataset_means = []
        for dataset in sorted({str(row["holdout_dataset"]) for row in values}):
            dataset_means.append(float(np.mean([
                float(row["BA"]) for row in values
                if row["holdout_dataset"] == dataset
            ])))
        output.append({
            "feature_set": feature_set,
            "config": config,
            "graph_gamma": gamma,
            "mean_BA": float(np.mean(dataset_means)),
            "min_dataset_BA": float(np.min(dataset_means)),
            "std_dataset_BA": float(np.std(dataset_means)),
            "robust_score": float(np.mean(dataset_means) - 0.5 * np.std(dataset_means)),
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])),
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])),
            "datasets": len(dataset_means),
            "runs": len(values),
        })
    return sorted(
        output,
        key=lambda row: (
            float(row["robust_score"]),
            float(row["min_dataset_BA"]),
            float(row["mean_BA"]),
        ),
        reverse=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified trace/completion experiments")
    p.add_argument("--datasets", nargs="+", type=Path, default=ume.DEFAULT_DATASETS)
    p.add_argument("--holdouts", nargs="+", default=["all"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--feature-sets",
        nargs="+",
        choices=("base", "trace", "trace_dropout", "completion", "trace_completion"),
        default=("base", "trace"),
    )
    p.add_argument(
        "--configs",
        nargs="+",
        choices=("balanced", "rank", "rank_trans", "rank_trans_domain"),
        default=("rank_trans_domain",),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-reduce-dim", type=int, default=64)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument(
        "--completion-cache-dir",
        type=Path,
        default=Path("/tmp/regr_fail_completion_cache"),
    )
    p.add_argument("--strict-completion", action="store_true")
    p.add_argument("--trace-window-size", type=int, default=64)
    p.add_argument("--trace-feature-dropout", type=float, default=0.50)
    p.add_argument("--trace-view-alpha", type=float, default=0.50)
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.6)
    p.add_argument("--hard-positive-ratio", type=float, default=0.25)
    p.add_argument("--max-pairs-per-dataset", type=int, default=30000)
    p.add_argument("--max-aux-per-dataset", type=int, default=10000)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--representation-dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--aux-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--ranking-weight", type=float, default=0.25)
    p.add_argument("--ranking-margin", type=float, default=0.5)
    p.add_argument("--transitivity-weight", type=float, default=0.10)
    p.add_argument("--domain-weight", type=float, default=0.05)
    p.add_argument("--domain-grl-scale", type=float, default=0.20)
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--graph-gammas", nargs="+", type=float, default=[0.0, 0.10])
    p.add_argument("--predict-batch-size", type=int, default=100000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [ume.resolve(path) for path in args.datasets]
    if args.holdouts == ["all"]:
        holdout_indices = list(range(len(datasets)))
    else:
        by_name = {dataset.name: idx for idx, dataset in enumerate(datasets)}
        holdout_indices = [by_name[name] for name in args.holdouts]
    rows: list[dict] = []
    stats: list[dict] = []
    for seed in args.seeds:
        for holdout_index in holdout_indices:
            fold_rows, fold_stats = run_fold(args, datasets, holdout_index, seed)
            rows.extend(fold_rows)
            stats.extend(fold_stats)
            write_csv(args.output_dir / "results.partial.csv", rows, list(rows[0]))
    fields = [
        "seed", "holdout_dataset", "train_datasets", "feature_set", "config",
        "graph_gamma", "BA", "TPR", "TNR", "k", "cases",
        "num_pred_clusters", "input_dim", "best_val_loss", "runtime_sec",
        "model_path", "pred_path", "prob_path",
    ]
    write_csv(args.output_dir / "results.csv", rows, fields)
    summary = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summary, list(summary[0]))
    if stats:
        write_csv(args.output_dir / "pair_stats.csv", stats, list(stats[0]))
    completion_config = ccf.load_completion_config()
    manifest = {
        "datasets": [str(dataset) for dataset in datasets],
        "feature_sets": list(args.feature_sets),
        "configs": list(args.configs),
        "seeds": args.seeds,
        "trace_window_size": args.trace_window_size,
        "completion_available": completion_config is not None,
        "completion_model": completion_config["model"] if completion_config else "",
        "formal_predictor_modified": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\n| rank | features | config | graph | mean BA | min BA | robust | TPR | TNR |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(summary[:20], 1):
        print(
            f"| {rank} | {row['feature_set']} | {row['config']} | "
            f"{row['graph_gamma']:.2f} | {row['mean_BA']:.4f} | "
            f"{row['min_dataset_BA']:.4f} | {row['robust_score']:.4f} | "
            f"{row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
