#!/usr/bin/env python3
"""Screen smaller seed ensembles against the frozen five-seed teacher.

This is an experimental, cache-only compression pass.  It never reads logs or
changes the submission path: each candidate averages a subset of strict-LODO
probability matrices, then evaluates the same fixed-k average-linkage
clustering protocol used by the full five-seed model.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred
from run_selective_multiview_experiments import load_seed_stack, resolve, _symmetrize


DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]
OFFICIAL_DATASETS = {"benchmark_set_1", "benchmark_set_2"}


def candidate_subsets(seeds: Sequence[int], max_subset_size: int) -> list[tuple[int, ...]]:
    unique = tuple(dict.fromkeys(seeds))
    subsets: list[tuple[int, ...]] = []
    for size in range(1, min(max_subset_size, len(unique)) + 1):
        subsets.extend(itertools.combinations(unique, size))
    if unique not in subsets:
        subsets.append(unique)
    return subsets


def blend_seed_stack(dual: np.ndarray, five: np.ndarray, beta: float) -> np.ndarray:
    if dual.shape != five.shape:
        raise ValueError(f"dual/five shape mismatch: {dual.shape} != {five.shape}")
    return np.clip((1.0 - beta) * dual + beta * five, 0.0, 1.0).astype(np.float32)


def probability_error(candidate: np.ndarray, teacher: np.ndarray) -> tuple[float, float]:
    tri = np.triu_indices_from(candidate, k=1)
    left = np.asarray(candidate[tri], dtype=np.float64)
    right = np.asarray(teacher[tri], dtype=np.float64)
    if not len(left):
        return 0.0, 1.0
    mae = float(np.mean(np.abs(left - right)))
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        correlation = 1.0 if np.allclose(left, right) else 0.0
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])
    return mae, correlation


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["seed_subset"])].append(row)
    output: list[dict] = []
    for subset, values in groups.items():
        dataset_scores = {str(row["dataset"]): float(row["BA"]) for row in values}
        official = [float(row["BA"]) for row in values if row["dataset"] in OFFICIAL_DATASETS]
        fake = [float(row["BA"]) for row in values if row["dataset"] not in OFFICIAL_DATASETS]
        output.append({
            "seed_subset": subset,
            "num_seeds": int(values[0]["num_seeds"]),
            "mean_BA": float(np.mean(list(dataset_scores.values()))),
            "worst_BA": float(np.min(list(dataset_scores.values()))),
            "official_mean_BA": float(np.mean(official)) if official else float("nan"),
            "fake_mean_BA": float(np.mean(fake)) if fake else float("nan"),
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])),
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])),
            "mean_teacher_MAE": float(np.mean([float(row["teacher_MAE"]) for row in values])),
            "mean_teacher_corr": float(np.mean([float(row["teacher_corr"]) for row in values])),
            "estimated_model_fraction": float(values[0]["num_seeds"]) / 5.0,
            "dataset_BA": json.dumps(dataset_scores, sort_keys=True),
        })
    return sorted(
        output,
        key=lambda row: (row["mean_BA"], row["worst_BA"], -row["num_seeds"]),
        reverse=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict-LODO seed ensemble compression screen")
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.50)
    parser.add_argument("--model-type", default="gbdt")
    parser.add_argument("--graph-tag", default="agglomerative_complete")
    parser.add_argument("--clusterer", default="agglomerative_avg")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prob_dir = resolve(args.prob_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preds").mkdir(exist_ok=True)
    subsets = candidate_subsets(args.seeds, args.max_subset_size)
    full_subset = tuple(dict.fromkeys(args.seeds))
    rows: list[dict] = []

    for dataset_arg in args.datasets:
        dataset = resolve(dataset_arg)
        name = dataset.name
        cases = osf.read_cases(dataset / "input.csv")
        gold = read_gold(osf.gold_path(dataset))
        k = len(set(gold))
        dual = load_seed_stack(
            prob_dir, name, "dual", args.seeds, args.model_type, args.graph_tag
        )
        five = load_seed_stack(
            prob_dir, name, "quad_event_object_context", args.seeds,
            args.model_type, args.graph_tag,
        )
        blended = blend_seed_stack(dual, five, args.beta)
        seed_to_index = {seed: index for index, seed in enumerate(args.seeds)}
        teacher = _symmetrize(np.mean(blended, axis=0))

        for subset in subsets:
            indices = [seed_to_index[seed] for seed in subset]
            probability = _symmetrize(np.mean(blended[indices], axis=0))
            result = gc.cluster_probability_graph(probability, k, args.clusterer)
            subset_tag = "-".join(map(str, subset))
            pred_path = output_dir / "preds" / f"{name}_seeds{subset_tag}.csv"
            pred = write_pred(pred_path, cases, result.labels)
            ba, tpr, tnr = pairwise_scores(gold, pred)
            mae, correlation = probability_error(probability, teacher)
            rows.append({
                "dataset": name,
                "seed_subset": subset_tag,
                "num_seeds": len(subset),
                "is_teacher": subset == full_subset,
                "BA": ba,
                "TPR": tpr,
                "TNR": tnr,
                "teacher_MAE": mae,
                "teacher_corr": correlation,
                "cases": len(cases),
                "k": k,
                "num_pred_clusters": len(set(pred)),
                "clusterer": args.clusterer,
                "pred_path": str(pred_path),
            })
        print(f"[seed-compress] dataset={name} candidates={len(subsets)}", flush=True)

    result_fields = sorted({key for row in rows for key in row})
    write_csv(output_dir / "results.csv", rows, result_fields)
    summary = summarize(rows)
    summary_fields = sorted({key for row in summary for key in row})
    write_csv(output_dir / "summary.csv", summary, summary_fields)

    print("\n| rank | seeds | n | mean BA | worst BA | official | fake | teacher MAE | corr | model fraction |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(summary[:20], 1):
        print(
            f"| {rank} | {row['seed_subset']} | {row['num_seeds']} | "
            f"{row['mean_BA']:.4f} | {row['worst_BA']:.4f} | "
            f"{row['official_mean_BA']:.4f} | {row['fake_mean_BA']:.4f} | "
            f"{row['mean_teacher_MAE']:.4f} | {row['mean_teacher_corr']:.4f} | "
            f"{row['estimated_model_fraction']:.2f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
