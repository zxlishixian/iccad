#!/usr/bin/env python3
"""Family-grouped leave-one-family-out validation for experimental Theta."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import evaluation_leakage_guard as elg
import official_style_features as osf
from run_experiments import pairwise_scores, read_gold
from train_theta import infer_family


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("official_format_fake_dataset/directed_cross_v4"),
    Path("official_format_fake_dataset/benchmark5_final"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--embedding-mode", choices=["embedding", "none"], default="embedding")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument(
        "--model-type", choices=["gbdt", "logistic", "gated_mlp"], default="gbdt"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--clusterers", nargs="+", choices=["theta_graph", "average"], default=["theta_graph", "average"])
    parser.add_argument("--max-pairs-per-family", type=int, default=24000)
    parser.add_argument("--top-l", type=int, default=48)
    parser.add_argument("--full-pair-limit", type=int, default=300)
    parser.add_argument(
        "--candidate-mode",
        choices=["concat", "multiview_anchor"],
        default="concat",
    )
    parser.add_argument("--anchors-per-cluster", type=int, default=2)
    parser.add_argument("--anchor-cluster-count", type=int, default=8)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/theta_llm_cache"))
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def read_prediction(path: Path, expected_cases: Sequence[str]) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected_cases):
        raise RuntimeError(f"prediction row mismatch for {path}: {len(rows)} vs {len(expected_cases)}")
    by_case = {str(row["Case"]): str(row["bucket"]) for row in rows}
    return [by_case[str(case)] for case in expected_cases]


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> list[dict]:
    methods: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        methods[str(row["clusterer"])].append(row)
    output: list[dict] = []
    for clusterer, values in methods.items():
        dataset_means: dict[str, float] = {}
        family_values: dict[str, list[float]] = defaultdict(list)
        for dataset in sorted({str(row["dataset"]) for row in values}):
            selected = [float(row["BA"]) for row in values if row["dataset"] == dataset]
            dataset_means[dataset] = statistics.mean(selected)
            family = next(str(row["holdout_family"]) for row in values if row["dataset"] == dataset)
            family_values[family].append(dataset_means[dataset])
        family_means = {family: statistics.mean(scores) for family, scores in family_values.items()}
        output.append({
            "clusterer": clusterer,
            "dataset_macro_BA": statistics.mean(dataset_means.values()),
            "family_macro_BA": statistics.mean(family_means.values()),
            "worst_dataset_BA": min(dataset_means.values()),
            "mean_TPR": statistics.mean(float(row["TPR"]) for row in values),
            "mean_TNR": statistics.mean(float(row["TNR"]) for row in values),
            "datasets": len(dataset_means),
            "families": len(family_means),
            "runs": len(values),
            "dataset_BA": json.dumps(dataset_means, sort_keys=True),
            "family_BA": json.dumps(family_means, sort_keys=True),
        })
    return sorted(output, key=lambda row: (row["family_macro_BA"], row["worst_dataset_BA"]), reverse=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = [resolve(path) for path in args.datasets]
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    families = {dataset: infer_family(dataset) for dataset in datasets}
    results: list[dict] = []

    for holdout_family in sorted(set(families.values())):
        train_datasets = [dataset for dataset in datasets if families[dataset] != holdout_family]
        held_datasets = [dataset for dataset in datasets if families[dataset] == holdout_family]
        for seed in args.seeds:
            fold_dir = output_dir / "folds" / holdout_family / f"seed_{seed}"
            model_dir = fold_dir / "model"
            manifest_path = model_dir / "manifest.json"
            if not (args.resume and manifest_path.exists()):
                command = [
                    args.python, str(ROOT / "train_theta.py"),
                    "--train-datasets", *(str(path) for path in train_datasets),
                    "--output-dir", str(model_dir),
                    "--embedding-mode", args.embedding_mode,
                    "--embedding-dim", str(args.embedding_dim),
                    "--model-type", args.model_type,
                    "--max-pairs-per-family", str(args.max_pairs_per_family),
                    "--random-state", str(seed),
                    "--llm-cache-dir", str(args.llm_cache_dir),
                    "--device", args.device,
                    "--epochs", str(args.epochs),
                ]
                print("[theta-lofo] " + " ".join(command), flush=True)
                subprocess.run(command, cwd=ROOT, check=True, timeout=args.timeout_sec)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            elg.assert_held_out(manifest, held_datasets)

            for dataset in held_datasets:
                cases = osf.read_cases(dataset / "input.csv")
                gold = read_gold(osf.gold_path(dataset))
                k = len(set(gold))
                for clusterer in args.clusterers:
                    pred_path = fold_dir / "preds" / f"{dataset.name}_{clusterer}.csv"
                    diag_path = fold_dir / "diagnostics" / f"{dataset.name}_{clusterer}.json"
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    diag_path.parent.mkdir(parents=True, exist_ok=True)
                    started = time.perf_counter()
                    if not (args.resume and pred_path.exists() and diag_path.exists()):
                        command = [
                            args.python, str(ROOT / "theta_inference.py"),
                            "--input", str(dataset / "input.csv"),
                            "--output", str(pred_path),
                            "--k", str(k),
                            "--model-dir", str(model_dir),
                            "--clusterer", clusterer,
                            "--top-l", str(args.top_l),
                            "--full-pair-limit", str(args.full_pair_limit),
                            "--candidate-mode", args.candidate_mode,
                            "--anchors-per-cluster", str(args.anchors_per_cluster),
                            "--anchor-cluster-count", str(args.anchor_cluster_count),
                            "--llm-cache-dir", str(args.llm_cache_dir),
                            "--diagnostics", str(diag_path),
                        ]
                        subprocess.run(command, cwd=ROOT, check=True, timeout=args.timeout_sec)
                    runtime = time.perf_counter() - started
                    pred = read_prediction(pred_path, cases)
                    ba, tpr, tnr = pairwise_scores(gold, pred)
                    diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
                    results.append({
                        "seed": seed,
                        "holdout_family": holdout_family,
                        "dataset": dataset.name,
                        "clusterer": clusterer,
                        "cases": len(cases),
                        "k": k,
                        "num_pred_clusters": len(set(pred)),
                        "BA": ba,
                        "TPR": tpr,
                        "TNR": tnr,
                        "runtime_sec": runtime,
                        "candidate_pairs": diagnostics["candidate_graph"]["pairs"],
                        "candidate_density": diagnostics["candidate_graph"].get("density", 1.0),
                        "candidate_mode": diagnostics["candidate_graph"].get("mode", "full"),
                        "pred_path": str(pred_path),
                        "model_dir": str(model_dir),
                    })
                    print(
                        f"[theta-lofo] family={holdout_family} seed={seed} "
                        f"dataset={dataset.name} clusterer={clusterer} "
                        f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                        flush=True,
                    )
    write_rows(output_dir / "results.csv", results)
    summary = summarize(results)
    write_rows(output_dir / "summary.csv", summary)
    print("\n| clusterer | family macro BA | dataset macro BA | worst BA | TPR | TNR |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in summary:
        print(
            f"| {row['clusterer']} | {row['family_macro_BA']:.4f} | "
            f"{row['dataset_macro_BA']:.4f} | {row['worst_dataset_BA']:.4f} | "
            f"{row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
