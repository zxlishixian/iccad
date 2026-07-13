#!/usr/bin/env python3
"""Strict-LODO simulation of deterministic-first sparse expert refinement.

The selector and anchors use only sim/regr deterministic features. Cached LODO
five-view probabilities are revealed only for selected-to-anchor pairs. Gold is
loaded after prediction for evaluation and never influences selection.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold
from run_selective_multiview_experiments import (
    case_difficulty,
    deterministic_proxy_stack,
    load_seed_stack,
    select_expert_cases,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def choose_cluster_anchors(
    labels: np.ndarray,
    similarity: np.ndarray,
    selected: np.ndarray,
    anchors_per_cluster: int,
) -> dict[int, list[int]]:
    anchors: dict[int, list[int]] = {}
    for label in sorted(set(int(value) for value in labels)):
        members = np.flatnonzero(labels == label)
        candidates = members[~selected[members]]
        if len(candidates) < anchors_per_cluster:
            candidates = members
        centrality = np.asarray([
            float(np.mean(similarity[idx, members[members != idx]]))
            if len(members) > 1 else 1.0
            for idx in candidates
        ])
        order = np.argsort(-centrality, kind="mergesort")
        anchors[label] = [int(candidates[idx]) for idx in order[:anchors_per_cluster]]
    return anchors


def sparse_refine_labels(
    base_labels: np.ndarray,
    deterministic_similarity: np.ndarray,
    expert_probability: np.ndarray,
    selected: np.ndarray,
    anchors_per_cluster: int,
    expert_weight: float,
    min_probability: float,
    margin: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    labels = np.asarray(base_labels, dtype=np.int32).copy()
    anchors = choose_cluster_anchors(
        labels, deterministic_similarity, selected, anchors_per_cluster
    )
    counts = {label: int(np.sum(labels == label)) for label in anchors}
    moved = 0
    evaluated_edges: set[tuple[int, int]] = set()
    for idx in np.flatnonzero(selected):
        own_label = int(labels[idx])
        cluster_scores: dict[int, float] = {}
        for label, cluster_anchors in anchors.items():
            values = []
            for anchor in cluster_anchors:
                if anchor == idx:
                    continue
                edge = (min(int(idx), anchor), max(int(idx), anchor))
                evaluated_edges.add(edge)
                score = (
                    expert_weight * float(expert_probability[idx, anchor])
                    + (1.0 - expert_weight) * float(deterministic_similarity[idx, anchor])
                )
                values.append(score)
            if values:
                cluster_scores[label] = float(np.mean(values))
        if own_label not in cluster_scores or not cluster_scores:
            continue
        best_label, best_score = max(cluster_scores.items(), key=lambda item: item[1])
        own_score = cluster_scores[own_label]
        if (
            best_label != own_label
            and counts[own_label] > 1
            and best_score >= min_probability
            and best_score - own_score >= margin
        ):
            labels[idx] = best_label
            counts[own_label] -= 1
            counts[best_label] += 1
            moved += 1
    return labels, {
        "moved_cases": moved,
        "expert_edges": len(evaluated_edges),
        "anchor_cases": len({idx for values in anchors.values() for idx in values}),
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["combo"])].append(row)
    output = []
    for combo, values in groups.items():
        bas = [float(row["BA"]) for row in values]
        output.append({
            "combo": combo,
            "mean_BA": float(np.mean(bas)),
            "worst_BA": float(np.min(bas)),
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])),
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])),
            "mean_selected_cases": float(np.mean([float(row["selected_cases"]) for row in values])),
            "mean_anchor_cases": float(np.mean([float(row["anchor_cases"]) for row in values])),
            "mean_expert_edges": float(np.mean([float(row["expert_edges"]) for row in values])),
            "mean_pair_coverage": float(np.mean([float(row["pair_coverage"]) for row in values])),
            "dataset_BA": json.dumps({str(row["dataset"]): float(row["BA"]) for row in values}, sort_keys=True),
        })
    return sorted(output, key=lambda row: (row["mean_BA"], row["worst_BA"]), reverse=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse anchor-refinement LODO simulation")
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.15, 0.25])
    parser.add_argument("--max-selected", nargs="+", type=int, default=[40, 80])
    parser.add_argument("--anchors-per-cluster", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--expert-weights", nargs="+", type=float, default=[0.50, 0.75])
    parser.add_argument("--min-probabilities", nargs="+", type=float, default=[0.55, 0.65])
    parser.add_argument("--margins", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--model-type", default="gbdt")
    parser.add_argument("--graph-tag", default="agglomerative_complete")
    parser.add_argument("--source-clusterer", default="agglomerative_complete")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prob_dir = resolve(args.prob_dir)
    output_dir = resolve(args.output_dir)
    rows: list[dict] = []
    for dataset_arg in args.datasets:
        dataset = resolve(dataset_arg)
        name = dataset.name
        cases = osf.read_cases(dataset / "input.csv")
        gold = read_gold(osf.gold_path(dataset))
        k = len(set(gold))
        det_features, _ = plf.build_llm_case_features_for_inputs(
            [dataset / "input.csv"], parser="drain", svd_dim=64,
            llm_args=None, log_llm_disabled=False,
        )
        det_stack = deterministic_proxy_stack(det_features)
        difficulty, _components, det_similarity = case_difficulty(
            det_stack, k, args.source_clusterer
        )
        base_labels = np.asarray(
            gc.cluster_probability_graph(det_similarity, k, args.source_clusterer).labels,
            dtype=np.int32,
        )
        five_stack = load_seed_stack(
            prob_dir, name, "quad_event_object_context", args.seeds,
            args.model_type, args.graph_tag,
        )
        five_probability = np.mean(five_stack, axis=0).astype(np.float32)
        total_pairs = max(1, len(cases) * (len(cases) - 1) // 2)

        base_ba, base_tpr, base_tnr = pairwise_scores(gold, [str(v) for v in base_labels])
        rows.append({
            "dataset": name, "combo": "deterministic_proxy_base",
            "BA": base_ba, "TPR": base_tpr, "TNR": base_tnr,
            "cases": len(cases), "k": k, "selected_cases": 0,
            "anchor_cases": 0, "moved_cases": 0, "expert_edges": 0,
            "pair_coverage": 0.0,
        })

        for fraction, max_selected, anchors_per_cluster, expert_weight, min_prob, margin in itertools.product(
            args.fractions, args.max_selected, args.anchors_per_cluster,
            args.expert_weights, args.min_probabilities, args.margins,
        ):
            selected = select_expert_cases(difficulty, fraction, 2, max_selected)
            labels, stats = sparse_refine_labels(
                base_labels, det_similarity, five_probability, selected,
                anchors_per_cluster, expert_weight, min_prob, margin,
            )
            ba, tpr, tnr = pairwise_scores(gold, [str(v) for v in labels])
            combo = (
                f"sparse_f{fraction:.2f}_cap{max_selected}_a{anchors_per_cluster}_"
                f"w{expert_weight:.2f}_p{min_prob:.2f}_m{margin:.2f}"
            )
            rows.append({
                "dataset": name, "combo": combo,
                "BA": ba, "TPR": tpr, "TNR": tnr,
                "cases": len(cases), "k": k,
                "selected_cases": int(np.sum(selected)),
                **stats,
                "pair_coverage": float(stats["expert_edges"]) / total_pairs,
            })
        print(f"[sparse-anchor] dataset={name} cases={len(cases)} configs={len(rows)}", flush=True)

    summary = summarize(rows)
    write_csv(output_dir / "results.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    print("| rank | combo | mean BA | worst BA | TPR | TNR | selected | anchors | edges | coverage |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(summary[:15], 1):
        print(
            f"| {rank} | {row['combo']} | {row['mean_BA']:.4f} | {row['worst_BA']:.4f} | "
            f"{row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} | "
            f"{row['mean_selected_cases']:.1f} | {row['mean_anchor_cases']:.1f} | "
            f"{row['mean_expert_edges']:.1f} | {row['mean_pair_coverage']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
