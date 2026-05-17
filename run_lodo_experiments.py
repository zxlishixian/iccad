#!/usr/bin/env python3
"""Leave-one-dataset-out validation for experimental pairwise methods.

This is a research utility. It trains on gold.csv from the training datasets and
validates on a held-out dataset. The official regr_fail_bucketing.py path is not
changed and remains gold/meta/trace-free by default.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold, read_pred
from run_half_split_experiments import DEFAULT_DATASETS
from run_input_signal_calibration import _temperature
from run_input_signal_experiments import ENSEMBLE_WEIGHTS
from train_pairwise_llm import sample_pairs

PROJECT_ROOT = Path(__file__).resolve().parent
ENSEMBLE_TYPES = ("logistic", "gbdt", "mlp")


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_labels_for_inputs(gold_paths: Sequence[Path]) -> list[str]:
    labels: list[str] = []
    for gold in gold_paths:
        labels.extend(read_gold(gold))
    return labels


def _dataset_k(dataset: Path) -> int:
    return max(1, len(set(read_gold(dataset / "gold.csv"))))


def _run_predictor(python: str, input_csv: Path, output_csv: Path, k: int, method: str, llm_cache_dir: Path) -> tuple[float, list[str]]:
    cmd = [
        python,
        str(PROJECT_ROOT / "regr_fail_bucketing.py"),
        "--input", str(input_csv),
        "--output", str(output_csv),
        "--k", str(k),
        "--parser", "drain",
        "--cluster", "agglomerative",
        "--cluster-factor", "0.875",
        "--svd-dim", "64",
        "--feature-level", "baseline",
        "--normalizer", "v1",
        "--line-mode", "default",
        "--template-weighting", "quality",
        "--token-weight-mode", "none",
    ]
    if method == "llm_concat_features":
        cmd.extend([
            "--llm-mode", "embedding",
            "--llm-fusion", "concat",
            "--llm-doc-style", "features",
            "--llm-weight", "4.0",
            "--llm-cache-dir", str(llm_cache_dir),
        ])
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=PROJECT_ROOT, check=False)
    runtime = time.perf_counter() - t0
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{method} predictor failed: {proc.stdout}\n{proc.stderr}")
    return runtime, read_pred(output_csv)


def _train_model_pkg(
    features: list[plf.LLMCaseFeature],
    labels: Sequence[str],
    feature_mode: str,
    model_type: str,
    random_state: int,
    args: argparse.Namespace,
) -> dict:
    pairs, y, stats = sample_pairs(
        features,
        labels,
        negative_ratio=args.negative_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        hard_positive_ratio=args.hard_positive_ratio,
        max_train_pairs=args.max_train_pairs if model_type == "mlp_rich" else args.max_ensemble_pairs,
        random_state=random_state,
        positive_sampling=args.positive_sampling,
        negative_sampling=args.negative_sampling,
    )
    X = plf.build_rich_pair_feature_matrix(features, pairs, feature_mode=feature_mode)
    if model_type == "logistic":
        pkg = plf.train_logistic_model(X, y, random_state=random_state)
    elif model_type == "gbdt":
        pkg = plf.train_gbdt_model(X, y, random_state=random_state)
    else:
        pkg = plf.train_mlp_model(
            X,
            y,
            input_dim=X.shape[1],
            mlp_arch=args.mlp_arch if model_type == "mlp_rich" else "shallow",
            loss=args.loss if model_type == "mlp_rich" else "bce",
            dropout=args.dropout if model_type == "mlp_rich" else args.ensemble_dropout,
            batch_size=args.batch_size,
            epochs=args.epochs if model_type == "mlp_rich" else args.ensemble_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            random_state=random_state,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            early_stop_patience=args.early_stop_patience,
            layernorm=True,
            batchnorm=False,
        )
    pkg.update({
        "feature_mode": feature_mode,
        "svd_dim": args.svd_dim,
        "llm_reduce_dim": args.llm_reduce_dim if feature_mode in ({"rich", "rich_no_det"} | plf.DUAL_FEATURE_MODES) else 0,
        "pair_stats": stats,
    })
    return pkg


def _train_lodo_models(train_datasets: Sequence[Path], output_dir: Path, tag: str, args: argparse.Namespace) -> tuple[dict, list[dict], list[Path]]:
    model_dir = output_dir / "models" / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    rich_path = model_dir / "rich_current_best.pt"
    ensemble_paths = [model_dir / f"ensemble_{name}.{'pt' if name == 'mlp' else 'pkl'}" for name in ENSEMBLE_TYPES]
    if args.reuse_models and rich_path.exists() and all(p.exists() for p in ensemble_paths):
        rich = plf.load_model_pkg(rich_path)
        ensemble = [plf.load_model_pkg(p) for p in ensemble_paths]
        return rich, ensemble, [rich_path, *ensemble_paths]

    train_inputs = [ds / "input.csv" for ds in train_datasets]
    train_golds = [ds / "gold.csv" for ds in train_datasets]
    llm_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=True,
    )
    features, _ = plf.build_llm_case_features_for_inputs(train_inputs, svd_dim=args.svd_dim, llm_args=llm_args)
    labels = _read_labels_for_inputs(train_golds)
    if len(features) != len(labels):
        raise RuntimeError(f"feature/label mismatch for {tag}: {len(features)} vs {len(labels)}")

    llm_reducer = plf.fit_llm_reducer(features, args.llm_reduce_dim, random_state=args.random_state)
    summary_reducer = plf.fit_llm_summary_reducer(features, args.llm_reduce_dim, random_state=args.random_state)
    rich = _train_model_pkg(features, labels, "llm_dual_struct_det_summary", "mlp_rich", args.random_state, args)
    rich.update({
        "llm_reducer": llm_reducer,
        "llm_summary_reducer": summary_reducer,
        "llm_reduce_dim": args.llm_reduce_dim,
        "mlp_arch": args.mlp_arch,
        "loss": args.loss,
    })
    plf.save_model_pkg(rich, rich_path)

    # The soft-voting ensemble intentionally stays on the compact summary21 pair features.
    ensemble: list[dict] = []
    for model_name, path in zip(ENSEMBLE_TYPES, ensemble_paths):
        pkg = _train_model_pkg(features, labels, "summary21", model_name, args.random_state, args)
        plf.save_model_pkg(pkg, path)
        ensemble.append(pkg)
    return rich, ensemble, [rich_path, *ensemble_paths]


def _predict_pairwise(prob: np.ndarray, k: int, pred_path: Path) -> list[str]:
    labels = plf.cluster_from_probability(prob.astype(np.float32), k)
    pred = [f"bucket_{label:03d}" for label in labels]
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket"])
        for item in pred:
            writer.writerow([item])
    return pred


def _eval_row(train_datasets: Sequence[Path], test_dataset: Path, method: str, pred: Sequence[str], runtime: float, model_path: str, pred_path: Path) -> dict:
    gold = read_gold(test_dataset / "gold.csv")
    ba, tpr, tnr = pairwise_scores(gold, pred)
    return {
        "train_datasets": "+".join(ds.name for ds in train_datasets),
        "test_dataset": test_dataset.name,
        "method": method,
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "num_cases": len(gold),
        "k": len(set(gold)),
        "num_pred_clusters": len(set(pred)),
        "runtime_sec": runtime,
        "model_path": model_path,
        "pred_path": str(pred_path),
    }


def run_lodo(args: argparse.Namespace) -> list[dict]:
    datasets = [(d if d.is_absolute() else (PROJECT_ROOT / d).resolve()) for d in args.datasets]
    rows: list[dict] = []
    for test_dataset in datasets:
        train_datasets = [ds for ds in datasets if ds != test_dataset]
        tag = "train_" + "_".join(ds.name for ds in train_datasets) + "__test_" + test_dataset.name
        print(f"\n[LODO] train={'+'.join(ds.name for ds in train_datasets)} test={test_dataset.name}", file=sys.stderr)
        k = _dataset_k(test_dataset)

        for method in ("deterministic", "llm_concat_features"):
            pred_path = args.output_dir / "preds" / tag / f"{method}.csv"
            runtime, pred = _run_predictor(args.python, test_dataset / "input.csv", pred_path, k, method, args.llm_cache_dir)
            rows.append(_eval_row(train_datasets, test_dataset, method, pred, runtime, "", pred_path))
            print(f"  [{method}] BA={rows[-1]['BA']:.6f} TPR={rows[-1]['TPR']:.6f} TNR={rows[-1]['TNR']:.6f}", file=sys.stderr)

        t0 = time.perf_counter()
        rich_model, ensemble_models, model_paths = _train_lodo_models(train_datasets, args.output_dir, tag, args)
        train_runtime = time.perf_counter() - t0
        llm_args_dual = plf._make_llm_args("embedding", llm_doc_style="features", llm_cache_dir=args.llm_cache_dir, svd_dim=args.svd_dim, llm_dual=True)
        val_features, _ = plf.build_llm_case_features(test_dataset / "input.csv", svd_dim=args.svd_dim, llm_args=llm_args_dual)

        t0 = time.perf_counter()
        p_ens = plf.predict_probability_matrix_ensemble(ensemble_models, list(ENSEMBLE_WEIGHTS), val_features, ensemble_mode="prob_average", batch_size=args.predict_batch_size)
        pred_path = args.output_dir / "preds" / tag / "pairwise_soft_voting_ensemble.csv"
        pred = _predict_pairwise(p_ens, k, pred_path)
        rows.append(_eval_row(train_datasets, test_dataset, "pairwise_soft_voting_ensemble", pred, train_runtime + time.perf_counter() - t0, ",".join(str(p) for p in model_paths[1:]), pred_path))
        print(f"  [ensemble] BA={rows[-1]['BA']:.6f} TPR={rows[-1]['TPR']:.6f} TNR={rows[-1]['TNR']:.6f}", file=sys.stderr)

        t0 = time.perf_counter()
        p_rich = _temperature(plf.predict_probability_matrix_sklearn(rich_model, val_features, batch_size=args.predict_batch_size), args.rich_temperature)
        p_ens_t = _temperature(p_ens, args.ensemble_temperature)
        prob = float(args.alpha) * p_rich + (1.0 - float(args.alpha)) * p_ens_t
        pred_path = args.output_dir / "preds" / tag / "current_no_trace_calibrated_blend.csv"
        pred = _predict_pairwise(prob, k, pred_path)
        rows.append(_eval_row(train_datasets, test_dataset, "current_no_trace_calibrated_blend", pred, train_runtime + time.perf_counter() - t0, ",".join(str(p) for p in model_paths), pred_path))
        print(f"  [current_best] BA={rows[-1]['BA']:.6f} TPR={rows[-1]['TPR']:.6f} TNR={rows[-1]['TNR']:.6f}", file=sys.stderr)
    return rows


def _print_table(rows: Sequence[dict]) -> None:
    methods = []
    datasets = []
    by = {}
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
        if row["test_dataset"] not in datasets:
            datasets.append(row["test_dataset"])
        by[(row["method"], row["test_dataset"])] = row
    print("\n| method | " + " | ".join(datasets) + " | mean_BA |")
    print("|---" + "|---:" * (len(datasets) + 1) + "|")
    for method in methods:
        vals = []
        cells = []
        for ds in datasets:
            r = by.get((method, ds))
            if r:
                vals.append(float(r["BA"]))
                cells.append(f"{float(r['BA']):.4f} ({float(r['TPR']):.3f}/{float(r['TNR']):.3f})")
            else:
                cells.append("")
        mean = float(np.mean(vals)) if vals else 0.0
        print(f"| {method} | " + " | ".join(cells) + f" | {mean:.4f} |")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run leave-one-dataset-out validation for current experimental best.")
    p.add_argument("--python", default="/home/lishixian/miniforge3/envs/collab-overcooked/bin/python")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/lodo_exp"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-reduce-dim", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temperature", type=float, default=1.15)
    p.add_argument("--ensemble-temperature", type=float, default=1.00)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--ensemble-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--max-train-pairs", type=int, default=300000)
    p.add_argument("--max-ensemble-pairs", type=int, default=200000)
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.5)
    p.add_argument("--hard-positive-ratio", type=float, default=0.5)
    p.add_argument("--positive-sampling", choices=("det_low", "diverse"), default="det_low")
    p.add_argument("--negative-sampling", choices=("det_high", "confusable"), default="det_high")
    p.add_argument("--mlp-arch", choices=("shallow", "deep", "residual"), default="residual")
    p.add_argument("--loss", choices=("bce", "focal"), default="focal")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--ensemble-dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", default="auto")
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--reuse-models", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_lodo(args)
    fields = ["train_datasets", "test_dataset", "method", "BA", "TPR", "TNR", "num_cases", "k", "num_pred_clusters", "runtime_sec", "model_path", "pred_path"]
    _write_csv(args.output_dir / "results.csv", rows, fields)
    _write_csv(args.output_dir / "summary.csv", rows, fields)
    _print_table(rows)
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    print(f"Summary: {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
