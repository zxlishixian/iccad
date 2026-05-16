#!/usr/bin/env python3
"""Run pairwise LLM learning experiments with half-split cross-scale validation.

Compares:
  1. logistic  – sklearn LogisticRegression
  2. gbdt      – sklearn HistGradientBoostingClassifier
  3. mlp       – small PyTorch MLP

against deterministic and LLM concat baselines.

Output: /tmp/pairwise_llm_exp/results.csv + summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import regr_fail_bucketing as rfb
from run_experiments import pairwise_scores, read_gold, read_pred
from run_half_split_experiments import DEFAULT_DATASETS, opposite_part, part_for_bit, stratified_half_split

PROJECT_ROOT = Path(__file__).resolve().parent


def run_baseline_predict(
    python: str,
    input_csv: Path,
    output_csv: Path,
    k: int,
    method: str,
    llm_args: dict | None = None,
) -> float:
    """Run regr_fail_bucketing.py for baseline methods."""
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
            "--llm-weight", "4.0",
            "--llm-doc-style", "features",
            "--llm-cache-dir", str(llm_args.get("llm_cache_dir", "/tmp/regr_fail_llm_cache")),
        ])
    elif method == "llm_concat_summary":
        cmd.extend([
            "--llm-mode", "embedding",
            "--llm-fusion", "concat",
            "--llm-weight", "4.0",
            "--llm-doc-style", "summary",
            "--llm-cache-dir", str(llm_args.get("llm_cache_dir", "/tmp/regr_fail_llm_cache")),
        ])

    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, cwd=PROJECT_ROOT)
    runtime = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"baseline {method} failed: {proc.stderr or proc.stdout}")
    return runtime


def run_pairwise_predict(
    python: str,
    input_csv: Path,
    output_csv: Path,
    k: int,
    model_path: Path,
    model_type: str,
    svd_dim: int,
    use_llm: bool,
    llm_doc_style: str,
    llm_cache_dir: Path,
    predict_batch_size: int = 100000,
) -> float:
    """Run pairwise model inference using pairwise_llm_features."""
    import numpy as np
    import pairwise_llm_features as plf

    start = time.perf_counter()

    if use_llm:
        llm_args = plf._make_llm_args(
            llm_mode="embedding",
            llm_doc_style=llm_doc_style,
            llm_cache_dir=llm_cache_dir,
            svd_dim=svd_dim,
        )
    else:
        llm_args = None

    features, _bundle = plf.build_llm_case_features(input_csv, svd_dim=svd_dim, llm_args=llm_args)
    model_pkg = plf.load_model_pkg(model_path)
    prob = plf.predict_probability_matrix_sklearn(model_pkg, features, batch_size=predict_batch_size)
    labels = plf.cluster_from_probability(prob, k)

    # Write output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket"])
        for label in labels:
            writer.writerow([f"bucket_{label:03d}"])

    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pairwise LLM learning half-split experiments.")
    parser.add_argument("--python", default="/home/lishixian/miniforge3/envs/collab-overcooked/bin/python")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pairwise_llm_exp"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--combo", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=("logistic", "gbdt", "mlp"), default=["logistic", "gbdt", "mlp"])
    parser.add_argument(
        "--feature-mode",
        choices=("summary21", "rich", "rich_no_llm", "rich_no_det", "llm_dual", "llm_dual_struct", "llm_dual_struct_det_summary"),
        default="summary21",
    )
    parser.add_argument("--llm-reduce-dim", type=int, default=128)
    parser.add_argument("--mlp-arch", choices=("shallow", "deep", "residual"), default="shallow")
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", default="auto")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--layernorm", action="store_true", default=True)
    parser.add_argument("--no-layernorm", action="store_true", default=False)
    parser.add_argument("--batchnorm", action="store_true", default=False)
    parser.add_argument("--baselines", nargs="+", choices=("deterministic", "llm_concat_features", "llm_concat_summary"),
                        default=["deterministic", "llm_concat_features"])
    parser.add_argument("--use-llm", action="store_true", default=True)
    parser.add_argument("--no-llm", action="store_true", default=False)
    parser.add_argument("--llm-doc-style", choices=("features", "summary"), default="features")
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-train-pairs", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--hard-positive-ratio", type=float, default=0.5)
    parser.add_argument("--predict-batch-size", type=int, default=100000)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if args.no_llm:
        args.use_llm = False
    if args.no_layernorm:
        args.layernorm = False

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        (Path(d) if Path(d).is_absolute() else (PROJECT_ROOT / d).resolve())
        for d in DEFAULT_DATASETS
    ]

    llm_kwargs = {"llm_cache_dir": str(args.llm_cache_dir)}

    results: list[dict] = []

    for seed in args.seeds:
        combo = args.combo

        # Build splits once per seed
        split_root = args.output_dir / "splits" / f"seed_{seed}"
        all_val_parts: list[dict] = []
        train_inputs: list[Path] = []
        for idx, ds in enumerate(datasets):
            splits = stratified_half_split(ds, seed, split_root)
            train_part = part_for_bit((combo >> idx) & 1)
            val_part = opposite_part(train_part)
            train_inputs.append(splits[train_part]["input"])
            val_info = dict(splits[val_part])
            val_info["dataset"] = ds.name
            all_val_parts.append(val_info)

        # --- Train pairwise models ---
        for model_type in args.methods:
            print(f"\n=== Training {model_type} seed={seed} combo={combo:03b} ===", file=sys.stderr)
            model_tag = model_type
            method_label = f"pairwise_{model_type}"
            if model_type == "mlp" or args.feature_mode != "summary21":
                model_tag = f"{model_type}_{args.feature_mode}_{args.mlp_arch}_{args.loss}"
                method_label = f"pairwise_{model_tag}"
            train_cmd = [
                args.python,
                str(PROJECT_ROOT / "train_pairwise_llm.py"),
                "--datasets", *[str(d) for d in datasets],
                "--output-dir", str(args.output_dir / "models"),
                "--model-type", model_type,
                "--model-tag", model_tag,
                "--feature-mode", args.feature_mode,
                "--llm-reduce-dim", str(args.llm_reduce_dim),
                "--mlp-arch", args.mlp_arch,
                "--loss", args.loss,
                "--focal-gamma", str(args.focal_gamma),
                "--focal-alpha", str(args.focal_alpha),
                "--early-stop-patience", str(args.early_stop_patience),
                "--svd-dim", str(args.svd_dim),
                "--llm-doc-style", args.llm_doc_style,
                "--llm-cache-dir", str(args.llm_cache_dir),
                "--seed", str(seed),
                "--combo", str(combo),
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--max-train-pairs", str(args.max_train_pairs),
                "--batch-size", str(args.batch_size),
                "--dropout", str(args.dropout),
                "--negative-ratio", str(args.negative_ratio),
                "--hard-negative-ratio", str(args.hard_negative_ratio),
                "--hard-positive-ratio", str(args.hard_positive_ratio),
                "--predict-batch-size", str(args.predict_batch_size),
            ]
            if args.hidden_dims:
                train_cmd.extend(["--hidden-dims", *[str(d) for d in args.hidden_dims]])
            if args.no_layernorm:
                train_cmd.append("--no-layernorm")
            if args.batchnorm:
                train_cmd.append("--batchnorm")
            if not args.use_llm:
                train_cmd.append("--no-llm")

            t0 = time.perf_counter()
            proc = subprocess.run(train_cmd, text=True, capture_output=True, check=False, cwd=PROJECT_ROOT)
            train_runtime = time.perf_counter() - t0
            if proc.returncode != 0:
                print(f"ERROR training {model_type}: {proc.stderr or proc.stdout}", file=sys.stderr)
                continue

            # Find model path
            ext = "pt" if model_type == "mlp" else "pkl"
            model_path = args.output_dir / "models" / f"model_seed{seed}_combo{combo:03b}_{model_tag}.{ext}"

            # Evaluate on each val part
            for part in all_val_parts:
                pred_path = (
                    args.output_dir / "preds" /
                    f"seed{seed}_combo{combo:03b}_{part['dataset']}_{model_type}.csv"
                )
                try:
                    runtime = run_pairwise_predict(
                        args.python,
                        part["input"],
                        pred_path,
                        part["k"],
                        model_path,
                        model_type,
                        args.svd_dim,
                        args.use_llm,
                        args.llm_doc_style,
                        args.llm_cache_dir,
                        args.predict_batch_size,
                    )
                except Exception as exc:
                    print(f"ERROR predicting {model_type} {part['dataset']}: {exc}", file=sys.stderr)
                    continue

                gold = read_gold(part["gold"])
                pred = read_pred(pred_path)
                ba, tpr, tnr = pairwise_scores(gold, pred)
                results.append({
                    "seed": seed,
                    "combo": f"{combo:03b}",
                    "dataset": part["dataset"],
                    "method": method_label,
                    "doc_style": args.llm_doc_style if args.use_llm else "none",
                    "model_type": model_type,
                    "BA": ba,
                    "TPR": tpr,
                    "TNR": tnr,
                    "num_cases": part["num_cases"],
                    "k": part["k"],
                    "num_pred_clusters": len(set(pred)),
                    "runtime_sec": runtime,
                    "model_path": str(model_path),
                    "pred_path": str(pred_path),
                })
                print(
                    f"done seed={seed} combo={combo:03b} dataset={part['dataset']} "
                    f"method=pairwise_{model_type} BA={ba:.6f}",
                    file=sys.stderr,
                )

        # --- Baselines ---
        for bl_method in args.baselines:
            # Baselines are the same for each seed since k and combos don't change the prediction
            # But we compute gold scores per split part
            for part in all_val_parts:
                pred_path = (
                    args.output_dir / "preds" /
                    f"seed{seed}_combo{combo:03b}_{part['dataset']}_{bl_method}.csv"
                )
                try:
                    runtime = run_baseline_predict(
                        args.python, part["input"], pred_path, part["k"], bl_method, llm_kwargs
                    )
                except Exception as exc:
                    print(f"ERROR baseline {bl_method}: {exc}", file=sys.stderr)
                    continue

                gold = read_gold(part["gold"])
                pred = read_pred(pred_path)
                ba, tpr, tnr = pairwise_scores(gold, pred)
                results.append({
                    "seed": seed,
                    "combo": f"{combo:03b}",
                    "dataset": part["dataset"],
                    "method": bl_method,
                    "doc_style": "features" if "features" in bl_method else ("summary" if "summary" in bl_method else "none"),
                    "model_type": "none",
                    "BA": ba,
                    "TPR": tpr,
                    "TNR": tnr,
                    "num_cases": part["num_cases"],
                    "k": part["k"],
                    "num_pred_clusters": len(set(pred)) if pred else 0,
                    "runtime_sec": runtime,
                    "model_path": "",
                    "pred_path": str(pred_path),
                })
                print(
                    f"done seed={seed} combo={combo:03b} dataset={part['dataset']} "
                    f"method={bl_method} BA={ba:.6f}",
                    file=sys.stderr,
                )

    # --- Write results ---
    result_header = [
        "seed", "combo", "dataset", "method", "doc_style", "model_type",
        "BA", "TPR", "TNR", "num_cases", "k", "num_pred_clusters",
        "runtime_sec", "model_path", "pred_path",
    ]
    results_path = args.output_dir / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_header)
        writer.writeheader()
        writer.writerows(results)

    # --- Summarize ---
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in results:
        groups[(row["method"], row["dataset"])].append(row)

    summary_rows: list[dict] = []
    for (method, dataset), items in sorted(groups.items()):
        summary_rows.append({
            "method": method,
            "dataset": dataset,
            "mean_BA": statistics.mean(float(r["BA"]) for r in items),
            "std_BA": statistics.stdev(float(r["BA"]) for r in items) if len(items) > 1 else 0.0,
            "mean_TPR": statistics.mean(float(r["TPR"]) for r in items),
            "mean_TNR": statistics.mean(float(r["TNR"]) for r in items),
            "num_runs": len(items),
        })

    summary_path = args.output_dir / "summary.csv"
    summary_header = ["method", "dataset", "mean_BA", "std_BA", "mean_TPR", "mean_TNR", "num_runs"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_header)
        writer.writeheader()
        writer.writerows(summary_rows)

    # --- Print markdown table ---
    print("\n| method | dataset | mean_BA | mean_TPR | mean_TNR | runs |")
    print("|---|---|---:|---:|---:|---:|")
    for row in summary_rows:
        print(
            f"| {row['method']} | {row['dataset']} | {float(row['mean_BA']):.6f} "
            f"| {float(row['mean_TPR']):.6f} | {float(row['mean_TNR']):.6f} | {row['num_runs']} |"
        )

    # Print comparison vs LLM concat features baseline
    print("\n--- Delta vs llm_concat_features ---")
    concat_means: dict[str, float] = {}
    for row in summary_rows:
        if row["method"] == "llm_concat_features":
            concat_means[row["dataset"]] = float(row["mean_BA"])

    if concat_means:
        print("| method | dataset | mean_BA | vs_concat_features |")
        print("|---|---|---:|---:|")
        for row in summary_rows:
            ds = row["dataset"]
            ba = float(row["mean_BA"])
            delta = ba - concat_means.get(ds, 0.0)
            delta_str = f"{delta:+.6f}" if delta else "+0.000000"
            print(f"| {row['method']} | {ds} | {ba:.6f} | {delta_str} |")

    print(f"\nResults: {results_path}")
    print(f"Summary:  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
