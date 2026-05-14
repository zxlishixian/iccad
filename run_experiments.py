#!/usr/bin/env python3
"""Run local parser/backend ablations on the three bundled datasets."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections import Counter
from math import comb
from pathlib import Path
from typing import Sequence


DATASETS = [
    ("first_batch_dataset", Path("dataset/first_batch_dataset/input.csv"), Path("dataset/first_batch_dataset/gold.csv"), 8),
    ("stage2_dataset_working", Path("dataset/stage2_dataset_working/input.csv"), Path("dataset/stage2_dataset_working/gold.csv"), 16),
    (
        "stage3_dataset_32bugs_640cases",
        Path("dataset/stage3_dataset_32bugs_640cases/input.csv"),
        Path("dataset/stage3_dataset_32bugs_640cases/gold.csv"),
        32,
    ),
]
COMBOS = [
    ("simple", "kmeans"),
    ("simple", "agglomerative"),
    ("drain", "kmeans"),
    ("drain", "agglomerative"),
]


def pick_col(fieldnames: Sequence[str], names: Sequence[str], fallback: int = 0) -> str:
    normalized = {"".join(ch for ch in name.lower() if ch.isalnum()): name for name in fieldnames}
    for name in names:
        key = "".join(ch for ch in name.lower() if ch.isalnum())
        if key in normalized:
            return normalized[key]
    return fieldnames[min(fallback, len(fieldnames) - 1)]


def read_gold(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    bug_col = pick_col(fields, ("bug_id", "bug", "gold", "label"), 1 if len(fields) > 1 else 0)
    return [row[bug_col] for row in rows]


def read_pred(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    bucket_col = pick_col(fields, ("bucket", "Bucket", "cluster", "label"), 0)
    return [row[bucket_col] for row in rows]


def pairwise_scores(gold: Sequence[str], pred: Sequence[str]) -> tuple[float, float, float]:
    bug_counts = Counter(gold)
    bucket_counts = Counter(pred)
    joint_counts = Counter(zip(gold, pred))
    positives = sum(comb(n, 2) for n in bug_counts.values() if n >= 2)
    pred_positives = sum(comb(n, 2) for n in bucket_counts.values() if n >= 2)
    tp = sum(comb(n, 2) for n in joint_counts.values() if n >= 2)
    fn = positives - tp
    fp = pred_positives - tp
    total = comb(len(gold), 2)
    negatives = total - positives
    tn = negatives - fp
    tpr = tp / positives if positives else 0.0
    tnr = tn / negatives if negatives else 0.0
    return (tpr + tnr) / 2.0, tpr, tnr


def run_one(
    python: str,
    dataset_name: str,
    input_csv: Path,
    gold_csv: Path,
    k: int,
    parser: str,
    cluster: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{dataset_name}_{parser}_{cluster}.csv"
    cmd = [
        python,
        "regr_fail_bucketing.py",
        "--input",
        str(input_csv),
        "--output",
        str(pred_path),
        "--k",
        str(k),
        "--parser",
        parser,
        "--cluster",
        cluster,
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    runtime = time.perf_counter() - start
    if proc.returncode != 0:
        return {
            "dataset": dataset_name,
            "parser": parser,
            "cluster": cluster,
            "cases": "",
            "k": k,
            "num_pred_clusters": "",
            "BA": 0.0,
            "TPR": 0.0,
            "TNR": 0.0,
            "runtime_sec": runtime,
            "status": f"FAILED: {proc.stderr.strip() or proc.stdout.strip()}",
        }
    gold = read_gold(gold_csv)
    pred = read_pred(pred_path)
    ba, tpr, tnr = pairwise_scores(gold, pred)
    return {
        "dataset": dataset_name,
        "parser": parser,
        "cluster": cluster,
        "cases": len(gold),
        "k": k,
        "num_pred_clusters": len(set(pred)),
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "runtime_sec": runtime,
        "status": "ok",
    }


def print_table(rows: Sequence[dict]) -> None:
    header = ["dataset", "parser", "cluster", "cases", "k", "num_pred_clusters", "BA", "TPR", "TNR", "runtime_sec"]
    print(",".join(header))
    for row in rows:
        values = []
        for key in header:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        print(",".join(values))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local bucketing ablations.")
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/regr_fail_bucketing_experiments"))
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = []
    for dataset_name, input_csv, gold_csv, k in DATASETS:
        for parser_name, cluster_name in COMBOS:
            row = run_one(args.python, dataset_name, input_csv, gold_csv, k, parser_name, cluster_name, args.output_dir)
            rows.append(row)
            print(
                f"done dataset={dataset_name} parser={parser_name} cluster={cluster_name} "
                f"BA={row['BA']:.6f} runtime={row['runtime_sec']:.3f}s",
                file=sys.stderr,
            )
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
