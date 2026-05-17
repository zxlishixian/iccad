#!/usr/bin/env python3
"""Anchor-window trace structural feature experiments.

Experimental only. Training reads half-split gold labels; official prediction is
unchanged and trace remains disabled by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import trace_anchor as ta
import trace_features as tf
from run_experiments import pairwise_scores, read_gold
from run_half_split_experiments import opposite_part, part_for_bit, stratified_half_split
from run_input_signal_experiments import (
    ENSEMBLE_WEIGHTS,
    build_val_parts,
    find_ensemble_models,
    print_wide,
    summarize,
)
from train_pairwise_llm import sample_pairs

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    PROJECT_ROOT / "fake_dataset/first_batch_dataset",
    PROJECT_ROOT / "fake_dataset/stage2_dataset_working",
    PROJECT_ROOT / "fake_dataset/stage3_dataset_32bugs_640cases",
]


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _temperature(prob: np.ndarray, temp: float) -> np.ndarray:
    if abs(float(temp) - 1.0) < 1e-12:
        return prob.astype(np.float32, copy=False)
    clipped = np.clip(prob.astype(np.float64), 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped)) / float(temp)
    out = 1.0 / (1.0 + np.exp(-logits))
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def _find_rich_model(model_root: Path, model_tag: str, seed: int, combo: int) -> Path:
    path = model_root / f"model_seed{seed}_combo{combo:03b}_{model_tag}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing rich model: {path}")
    return path


def _split_parts(datasets: Sequence[Path], seed: int, combo: int, split_root: Path) -> tuple[list[Path], list[Path], list[dict]]:
    train_inputs: list[Path] = []
    train_golds: list[Path] = []
    val_parts: list[dict] = []
    for idx, ds in enumerate(datasets):
        splits = stratified_half_split(ds, seed, split_root)
        train_part = part_for_bit((combo >> idx) & 1)
        val_part = opposite_part(train_part)
        train_inputs.append(splits[train_part]["input"])
        train_golds.append(splits[train_part]["gold"])
        val_info = dict(splits[val_part])
        val_info["dataset"] = ds.name
        val_parts.append(val_info)
    return train_inputs, train_golds, val_parts


def _labels_for_inputs(train_inputs: Sequence[Path], train_golds: Sequence[Path], n_features: int) -> list[str]:
    labels: list[str] = []
    offset = 0
    for inp, gold in zip(train_inputs, train_golds):
        gold_labels = read_gold(gold)
        offset += len(gold_labels)
        if offset > n_features:
            raise RuntimeError(f"feature/label mismatch in {inp}")
        labels.extend(gold_labels)
    if len(labels) != n_features:
        raise RuntimeError(f"feature/label mismatch: {len(labels)} vs {n_features}")
    return labels


def _tail_features_for_inputs(input_csvs: Sequence[Path], tail_lines: int) -> list[tf.TraceCaseFeature]:
    out: list[tf.TraceCaseFeature] = []
    for input_csv in input_csvs:
        out.extend(tf.build_trace_case_features(input_csv, tail_lines=tail_lines))
    return out


def _trace_matrix(mode: str, trace_features: Sequence, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    if mode == "tail":
        return tf.build_trace_pair_feature_matrix(trace_features, pairs)
    if mode == "anchor":
        return ta.build_anchor_trace_pair_feature_matrix(trace_features, pairs)
    raise ValueError(f"unknown trace feature mode: {mode}")


def _build_aug_matrix(
    rich_features: list[plf.LLMCaseFeature],
    trace_features: Sequence,
    pairs: Sequence[tuple[int, int]],
    feature_mode: str,
    trace_mode: str,
) -> np.ndarray:
    rich = plf.build_rich_pair_feature_matrix(rich_features, list(pairs), feature_mode=feature_mode)
    trace = _trace_matrix(trace_mode, trace_features, pairs)
    return np.hstack([rich, trace]).astype(np.float32, copy=False)


def _predict_aug_probability_matrix(
    model_pkg: dict,
    rich_features: list[plf.LLMCaseFeature],
    trace_features: Sequence,
    trace_mode: str,
    batch_size: int,
) -> np.ndarray:
    plf.prepare_features_for_model(model_pkg, rich_features)
    n = len(rich_features)
    probs = np.eye(n, dtype=np.float32)
    pairs: list[tuple[int, int]] = []
    feature_mode = str(model_pkg.get("feature_mode", "llm_dual_struct_det_summary"))

    def flush() -> None:
        if not pairs:
            return
        X = _build_aug_matrix(rich_features, trace_features, pairs, feature_mode, trace_mode)
        scaler = model_pkg.get("scaler")
        if scaler is not None:
            X = scaler.transform(X)
        model = model_pkg["model"]
        model_type = str(model_pkg.get("model_type", ""))
        if model_type == "mlp":
            import torch

            device = model_pkg.get("device", "cpu")
            model.to(device)
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(X.astype(np.float32)).to(device)).detach().cpu()
                batch_probs = torch.sigmoid(logits).numpy().astype(np.float32)
        elif hasattr(model, "predict_proba"):
            batch_probs = model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            batch_probs = np.clip(model.predict(X).astype(np.float32), 1e-6, 1.0 - 1e-6)
        for (i, j), prob in zip(pairs, batch_probs):
            probs[i, j] = float(prob)
            probs[j, i] = float(prob)
        pairs.clear()

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            if len(pairs) >= batch_size:
                flush()
    flush()
    return probs


def _ensemble_probability(
    part: dict,
    ensemble_pkgs: list[dict],
    llm_cache_dir: Path,
    svd_dim: int,
    batch_size: int,
) -> np.ndarray:
    ensemble_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=llm_cache_dir,
        svd_dim=svd_dim,
    )
    features, _ = plf.build_llm_case_features(part["input"], svd_dim=svd_dim, llm_args=ensemble_args)
    return plf.predict_probability_matrix_ensemble(
        ensemble_pkgs,
        list(ENSEMBLE_WEIGHTS),
        features,
        ensemble_mode="prob_average",
        batch_size=batch_size,
    )


def _evaluate_probability(
    prob: np.ndarray,
    part: dict,
    run_name: str,
    method: str,
    seed: int,
    combo: int,
    runtime: float,
    located_by_counts: Counter | None = None,
    fallback_rate: float = 0.0,
) -> dict:
    labels = plf.cluster_from_probability(prob.astype(np.float32), part["k"])
    pred = [f"bucket_{label:03d}" for label in labels]
    ba, tpr, tnr = pairwise_scores(read_gold(part["gold"]), pred)
    return {
        "run_name": run_name,
        "method": method,
        "feature_mode": "llm_dual_struct_det_summary",
        "llm_reduce_dim": 64,
        "positive_sampling": "det_low",
        "negative_sampling": "det_high",
        "blend_alpha": "0.88",
        "seed": seed,
        "combo": f"{combo:03b}",
        "dataset": part["dataset"],
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "num_cases": part["num_cases"],
        "k": part["k"],
        "num_pred_clusters": len(set(labels)),
        "runtime_sec": runtime,
        "located_by_counts": json.dumps(dict(located_by_counts or {}), sort_keys=True),
        "fallback_rate": fallback_rate,
    }


def _evaluate_current_best(args: argparse.Namespace, datasets: Sequence[Path], seed: int, combo: int) -> list[dict]:
    rich_model = plf.load_model_pkg(_find_rich_model(args.model_root, args.model_tag, seed, combo))
    rich_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=True,
    )
    ensemble_pkgs = [plf.load_model_pkg(path) for path in find_ensemble_models(args.ensemble_model_dir, seed, combo)]
    rows: list[dict] = []
    for part in build_val_parts(datasets, seed, combo, args.split_root / f"seed_{seed}"):
        t0 = time.perf_counter()
        rich_features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=rich_args)
        p_rich = plf.predict_probability_matrix_sklearn(rich_model, rich_features, batch_size=args.predict_batch_size)
        p_ens = _ensemble_probability(part, ensemble_pkgs, args.llm_cache_dir, args.svd_dim, args.predict_batch_size)
        prob = args.alpha * _temperature(p_rich, args.rich_temp) + (1.0 - args.alpha) * _temperature(p_ens, args.ensemble_temp)
        rows.append(_evaluate_probability(prob, part, "no_trace_current_best", "no_trace_current_best", seed, combo, time.perf_counter() - t0))
        print(f"[eval] seed={seed} dataset={part['dataset']} no_trace BA={rows[-1]['BA']:.6f}", file=sys.stderr)
    return rows


def _train_trace_augmented(
    args: argparse.Namespace,
    datasets: Sequence[Path],
    seed: int,
    combo: int,
    trace_mode: str,
    window_size: int,
) -> tuple[dict, list[dict]]:
    train_inputs, train_golds, _val_parts = _split_parts(datasets, seed, combo, args.output_dir / "splits" / f"seed_{seed}")
    llm_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=True,
    )
    rich_features, _ = plf.build_llm_case_features_for_inputs(train_inputs, svd_dim=args.svd_dim, llm_args=llm_args)
    labels = _labels_for_inputs(train_inputs, train_golds, len(rich_features))
    llm_reducer = plf.fit_llm_reducer(rich_features, args.llm_reduce_dim, random_state=seed)
    llm_summary_reducer = plf.fit_llm_summary_reducer(rich_features, args.llm_reduce_dim, random_state=seed)
    if trace_mode == "tail":
        trace_feats = _tail_features_for_inputs(train_inputs, tail_lines=window_size)
        debug_rows: list[dict] = []
    else:
        trace_feats, debug_rows = ta.build_anchor_trace_case_features(train_inputs, window_size=window_size)
    if len(trace_feats) != len(rich_features):
        raise RuntimeError(f"trace/rich feature mismatch: {len(trace_feats)} vs {len(rich_features)}")
    pairs, y, pair_stats = sample_pairs(
        rich_features,
        labels,
        negative_ratio=args.negative_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_positive_ratio=args.hard_positive_ratio,
        max_train_pairs=args.max_train_pairs,
        random_state=seed,
    )
    X = _build_aug_matrix(rich_features, trace_feats, pairs, args.feature_mode, trace_mode)
    print(
        f"[train] seed={seed} mode={trace_mode} window={window_size} dim={X.shape[1]} "
        f"pairs={len(pairs)} pos={pair_stats['positive_pairs']} neg={pair_stats['negative_pairs']}",
        file=sys.stderr,
    )
    model_pkg = plf.train_mlp_model(
        X,
        y,
        input_dim=X.shape[1],
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        random_state=seed,
        mlp_arch="residual",
        loss="focal",
        early_stop_patience=args.early_stop_patience,
        layernorm=True,
        batchnorm=False,
    )
    model_pkg.update({
        "feature_mode": args.feature_mode,
        "llm_reduce_dim": args.llm_reduce_dim,
        "llm_reducer": llm_reducer,
        "llm_summary_reducer": llm_summary_reducer,
        "svd_dim": args.svd_dim,
        "trace_feature_mode": trace_mode,
        "trace_window_size": window_size,
        "pair_stats": pair_stats,
    })
    return model_pkg, debug_rows


def _evaluate_trace_augmented(
    args: argparse.Namespace,
    datasets: Sequence[Path],
    seed: int,
    combo: int,
    trace_mode: str,
    window_size: int,
    model_pkg: dict,
) -> tuple[list[dict], list[dict]]:
    _, _, val_parts = _split_parts(datasets, seed, combo, args.output_dir / "splits" / f"seed_{seed}")
    llm_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=True,
    )
    ensemble_pkgs = [plf.load_model_pkg(path) for path in find_ensemble_models(args.ensemble_model_dir, seed, combo)]
    rows: list[dict] = []
    debug_all: list[dict] = []
    for part in val_parts:
        t0 = time.perf_counter()
        rich_features, _ = plf.build_llm_case_features(part["input"], svd_dim=args.svd_dim, llm_args=llm_args)
        if trace_mode == "tail":
            trace_feats = tf.build_trace_case_features(part["input"], tail_lines=window_size)
            debug_rows: list[dict] = []
            located_counts = Counter(f.file_status for f in trace_feats)
            fallback_rate = 0.0
        else:
            trace_feats, debug_rows = ta.build_anchor_trace_case_features([part["input"]], window_size=window_size)
            located_counts = Counter(row["located_by"] for row in debug_rows)
            fallback_rate = float(located_counts.get("tail", 0)) / max(1, len(debug_rows))
            debug_all.extend(dict(row, seed=seed, dataset=part["dataset"], window_size=window_size) for row in debug_rows)
        p_rich = _predict_aug_probability_matrix(model_pkg, rich_features, trace_feats, trace_mode, args.predict_batch_size)
        p_ens = _ensemble_probability(part, ensemble_pkgs, args.llm_cache_dir, args.svd_dim, args.predict_batch_size)
        prob = args.alpha * _temperature(p_rich, args.rich_temp) + (1.0 - args.alpha) * _temperature(p_ens, args.ensemble_temp)
        run_name = f"{trace_mode}_trace_struct_w{window_size}"
        rows.append(_evaluate_probability(prob, part, run_name, "anchor_trace_augmented" if trace_mode == "anchor" else "tail_trace_augmented", seed, combo, time.perf_counter() - t0, located_counts, fallback_rate))
        print(f"[eval] seed={seed} dataset={part['dataset']} {run_name} BA={rows[-1]['BA']:.6f}", file=sys.stderr)
    return rows, debug_all


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    datasets = [(d if d.is_absolute() else (PROJECT_ROOT / d).resolve()) for d in args.datasets]
    rows: list[dict] = []
    debug_rows: list[dict] = []
    for seed in args.seeds:
        rows.extend(_evaluate_current_best(args, datasets, seed, args.combo))
        if args.include_tail:
            model_pkg, _ = _train_trace_augmented(args, datasets, seed, args.combo, "tail", args.tail_lines)
            trace_rows, trace_debug = _evaluate_trace_augmented(args, datasets, seed, args.combo, "tail", args.tail_lines, model_pkg)
            rows.extend(trace_rows)
            debug_rows.extend(trace_debug)
        for window_size in args.window_sizes:
            model_pkg, train_debug = _train_trace_augmented(args, datasets, seed, args.combo, "anchor", window_size)
            debug_rows.extend(dict(row, seed=seed, dataset="train", window_size=window_size) for row in train_debug)
            trace_rows, trace_debug = _evaluate_trace_augmented(args, datasets, seed, args.combo, "anchor", window_size, model_pkg)
            rows.extend(trace_rows)
            debug_rows.extend(trace_debug)
    return rows, debug_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run anchor-guided trace window structural experiments.")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/anchor_trace_exp_seed0"))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--window-sizes", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--include-tail", action="store_true", default=False)
    p.add_argument("--tail-lines", type=int, default=500)
    p.add_argument("--model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--split-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64/splits"))
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--feature-mode", default="llm_dual_struct_det_summary")
    p.add_argument("--llm-reduce-dim", type=int, default=64)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temp", type=float, default=1.15)
    p.add_argument("--ensemble-temp", type=float, default=1.00)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--max-train-pairs", type=int, default=300000)
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.5)
    p.add_argument("--hard-positive-ratio", type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    p.add_argument("--predict-batch-size", type=int, default=100000)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, debug_rows = run(args)
    result_header = [
        "run_name", "method", "seed", "combo", "dataset", "BA", "TPR", "TNR",
        "num_cases", "k", "num_pred_clusters", "runtime_sec", "located_by_counts", "fallback_rate",
    ]
    _write_csv(args.output_dir / "results.csv", rows, result_header)
    summary_rows = summarize(rows)
    summary_header = [
        "run_name", "method", "feature_mode", "reduce_dim", "positive_sampling",
        "negative_sampling", "blend_alpha", "dataset", "mean_BA", "std_BA",
        "mean_TPR", "mean_TNR", "num_runs",
    ]
    _write_csv(args.output_dir / "summary.csv", summary_rows, summary_header)
    if debug_rows:
        debug_fields = sorted({key for row in debug_rows for key in row})
        _write_csv(args.output_dir / "anchor_debug.csv", debug_rows, debug_fields)
    print_wide(summary_rows)
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")
    if debug_rows:
        print(f"Anchor debug: {args.output_dir / 'anchor_debug.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
