#!/usr/bin/env python3
"""Experimental leave-one-dataset-out supervised token-weight experiments.

These scripts are research utilities for supervised token weighting experiments.
Current validation showed that learned token weights did not consistently
outperform the no-weight baseline, so they are not enabled by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from run_experiments import DATASETS, pairwise_scores, read_gold, read_pred


def run_split(
    python: str,
    output_dir: Path,
    train_datasets: list[tuple[str, Path, Path, int]],
    test_dataset: tuple[str, Path, Path, int],
    cluster_factor: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_names = [item[0] for item in train_datasets]
    test_name, input_csv, gold_csv, k = test_dataset
    split_name = f"train_{'_'.join(train_names)}__test_{test_name}_cf{str(cluster_factor).replace('.', 'p')}"
    weights_path = output_dir / f"{split_name}_token_weights.json"
    pred_path = output_dir / f"{split_name}_pred.csv"

    train_cmd = [
        python,
        "train_token_weights.py",
        "--datasets",
        *[str(item[1].parent) for item in train_datasets],
        "--output",
        str(weights_path),
    ]
    train_proc = subprocess.run(train_cmd, text=True, capture_output=True, check=False)
    if train_proc.returncode != 0:
        raise RuntimeError(train_proc.stderr or train_proc.stdout)

    pred_cmd = [
        python,
        "regr_fail_bucketing.py",
        "--input",
        str(input_csv),
        "--output",
        str(pred_path),
        "--k",
        str(k),
        "--parser",
        "drain",
        "--cluster",
        "agglomerative",
        "--cluster-factor",
        str(cluster_factor),
        "--token-weights",
        str(weights_path),
        "--token-weight-mode",
        "repeat",
    ]
    start = time.perf_counter()
    pred_proc = subprocess.run(pred_cmd, text=True, capture_output=True, check=False)
    runtime = time.perf_counter() - start
    if pred_proc.returncode != 0:
        raise RuntimeError(pred_proc.stderr or pred_proc.stdout)

    gold = read_gold(gold_csv)
    pred = read_pred(pred_path)
    ba, tpr, tnr = pairwise_scores(gold, pred)
    return {
        "train_datasets": "+".join(train_names),
        "test_dataset": test_name,
        "cluster_factor": cluster_factor,
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "num_pred_clusters": len(set(pred)),
        "runtime_sec": runtime,
        "token_weights_path": str(weights_path),
    }


def print_csv(rows: Sequence[dict]) -> None:
    header = [
        "train_datasets",
        "test_dataset",
        "cluster_factor",
        "BA",
        "TPR",
        "TNR",
        "num_pred_clusters",
        "runtime_sec",
        "token_weights_path",
    ]
    print(",".join(header))
    for row in rows:
        values = []
        for key in header:
            value = row[key]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        print(",".join(values))


def print_markdown(rows: Sequence[dict]) -> None:
    print("\n| train_datasets | test_dataset | cluster_factor | BA | TPR | TNR | pred_clusters | runtime_sec |", file=sys.stderr)
    print("|---|---|---:|---:|---:|---:|---:|---:|", file=sys.stderr)
    for row in rows:
        print(
            f"| {row['train_datasets']} | {row['test_dataset']} | {row['cluster_factor']:.2f} "
            f"| {row['BA']:.6f} | {row['TPR']:.6f} | {row['TNR']:.6f} "
            f"| {row['num_pred_clusters']} | {row['runtime_sec']:.3f} |",
            file=sys.stderr,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leave-one-dataset-out supervised experiments.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/supervised_exp"))
    parser.add_argument("--cluster-factors", nargs="+", type=float, default=[1.0, 1.25, 1.5])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = []
    for test_idx, test_dataset in enumerate(DATASETS):
        train_datasets = [dataset for idx, dataset in enumerate(DATASETS) if idx != test_idx]
        for cluster_factor in args.cluster_factors:
            row = run_split(args.python, args.output_dir, train_datasets, test_dataset, cluster_factor)
            rows.append(row)
            print(
                f"done train={row['train_datasets']} test={row['test_dataset']} "
                f"cf={cluster_factor} BA={row['BA']:.6f}",
                file=sys.stderr,
            )
    print_csv(rows)
    print_markdown(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
