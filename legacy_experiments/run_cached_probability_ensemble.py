#!/usr/bin/env python3
"""Average cached per-seed probability matrices and evaluate clustering."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred
from run_selective_multiview_experiments import resolve, _symmetrize


DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cached probability ensemble")
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--view", required=True)
    parser.add_argument("--model-type", default="gbdt")
    parser.add_argument("--graph-tag", default="agglomerative_avg")
    parser.add_argument("--clusterer", default="agglomerative_avg")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prob_dir = resolve(args.prob_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preds").mkdir(exist_ok=True)
    rows = []
    for dataset_arg in args.datasets:
        dataset = resolve(dataset_arg)
        name = dataset.name
        cases = osf.read_cases(dataset / "input.csv")
        gold = read_gold(osf.gold_path(dataset))
        matrices = []
        for seed in args.seeds:
            path = prob_dir / (
                f"{name}_{args.view}_{args.model_type}_{args.graph_tag}_seed{seed}.npy"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            matrices.append(_symmetrize(np.load(path)))
        probability = _symmetrize(np.mean(np.stack(matrices), axis=0))
        k = len(set(gold))
        result = gc.cluster_probability_graph(probability, k, args.clusterer)
        pred_path = output_dir / "preds" / f"{name}_ensemble.csv"
        pred = write_pred(pred_path, cases, result.labels)
        ba, tpr, tnr = pairwise_scores(gold, pred)
        rows.append({
            "dataset": name,
            "seeds": "-".join(map(str, args.seeds)),
            "BA": ba,
            "TPR": tpr,
            "TNR": tnr,
            "cases": len(cases),
            "k": k,
            "num_pred_clusters": len(set(pred)),
            "pred_path": str(pred_path),
        })
        print(f"[ensemble] dataset={name} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}")
    write_csv(output_dir / "results.csv", rows, list(rows[0]))
    summary = [{
        "mean_BA": float(np.mean([row["BA"] for row in rows])),
        "worst_BA": float(np.min([row["BA"] for row in rows])),
        "mean_TPR": float(np.mean([row["TPR"] for row in rows])),
        "mean_TNR": float(np.mean([row["TNR"] for row in rows])),
        "datasets": len(rows),
    }]
    write_csv(output_dir / "summary.csv", summary, list(summary[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
