#!/usr/bin/env python3
"""Evaluate dual-first selective five-view inference from strict-LODO caches.

Difficulty and expert selection use only dual-model probabilities. Gold labels
are loaded after clustering for evaluation. The deployable policy applies the
five-view branch only when both pair endpoints received extra embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _symmetrize(prob: np.ndarray) -> np.ndarray:
    out = np.asarray(prob, dtype=np.float32)
    out = np.clip((out + out.T) * 0.5, 0.0, 1.0)
    np.fill_diagonal(out, 1.0)
    return out


def load_seed_stack(
    prob_dir: Path,
    dataset: str,
    view: str,
    seeds: Sequence[int],
    model_type: str,
    graph_tag: str,
) -> np.ndarray:
    matrices = []
    for seed in seeds:
        filename = f"{dataset}_{view}_{model_type}_{graph_tag}_seed{seed}.npy"
        path = prob_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        matrices.append(_symmetrize(np.load(path)))
    return np.stack(matrices, axis=0)


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks.astype(np.float32)


def case_difficulty(
    dual_stack: np.ndarray,
    k: int,
    source_clusterer: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    seeds, n, _ = dual_stack.shape
    mean_prob = _symmetrize(np.mean(dual_stack, axis=0))
    clipped = np.clip(mean_prob, 1e-6, 1.0 - 1e-6)
    pair_entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    np.fill_diagonal(pair_entropy, 0.0)
    entropy = np.sum(pair_entropy, axis=1) / max(1, n - 1)

    disagreement_matrix = np.std(dual_stack, axis=0)
    np.fill_diagonal(disagreement_matrix, 0.0)
    disagreement = np.sum(disagreement_matrix, axis=1) / max(1, n - 1)

    seed_labels = np.stack([
        np.asarray(gc.cluster_probability_graph(prob, k, source_clusterer).labels, dtype=np.int32)
        for prob in dual_stack
    ])
    coassoc = np.mean(
        seed_labels[:, :, None] == seed_labels[:, None, :], axis=0, dtype=np.float32
    )
    instability_matrix = 4.0 * coassoc * (1.0 - coassoc)
    np.fill_diagonal(instability_matrix, 0.0)
    instability = np.sum(instability_matrix, axis=1) / max(1, n - 1)

    base_labels = np.asarray(
        gc.cluster_probability_graph(mean_prob, k, source_clusterer).labels, dtype=np.int32
    )
    margin_difficulty = np.zeros(n, dtype=np.float32)
    unique_labels = np.unique(base_labels)
    for idx in range(n):
        own = np.flatnonzero(base_labels == base_labels[idx])
        own = own[own != idx]
        own_score = float(np.mean(mean_prob[idx, own])) if len(own) else 0.0
        other_scores = [
            float(np.mean(mean_prob[idx, base_labels == label]))
            for label in unique_labels
            if label != base_labels[idx]
        ]
        nearest_other = max(other_scores, default=0.0)
        margin_difficulty[idx] = nearest_other - own_score

    components = {
        "entropy": entropy.astype(np.float32),
        "disagreement": disagreement.astype(np.float32),
        "instability": instability.astype(np.float32),
        "margin_difficulty": margin_difficulty.astype(np.float32),
    }
    score = (
        0.35 * _rank01(components["entropy"])
        + 0.25 * _rank01(components["disagreement"])
        + 0.25 * _rank01(components["instability"])
        + 0.15 * _rank01(components["margin_difficulty"])
    ).astype(np.float32)
    return score, components, mean_prob


def deterministic_proxy_stack(features: Sequence[plf.LLMCaseFeature]) -> np.ndarray:
    """Build three cheap P_same-like views without embedding or labels."""
    n = len(features)
    matrices = [np.eye(n, dtype=np.float32) for _ in range(3)]
    for i in range(n):
        for j in range(i + 1, n):
            structured = plf.build_structured_pair_feature_vector(features[i], features[j])
            det = plf.build_det_scalar_summary_vector(features[i], features[j])
            det_cos = float(np.clip((det[0] + 1.0) * 0.5, 0.0, 1.0))
            token = float(np.mean(det[2:6]))
            signature = float(max(structured[11], structured[12]))
            mismatch_op = float(max(structured[5], structured[6]))
            source = float(max(structured[0], structured[1], structured[2]))
            conflict = float(max(structured[16], structured[17]))
            values = (
                0.45 * det_cos + 0.30 * token + 0.15 * signature + 0.10 * mismatch_op - 0.15 * conflict,
                0.30 * det_cos + 0.25 * token + 0.30 * signature + 0.15 * source - 0.20 * conflict,
                0.55 * det_cos + 0.20 * token + 0.15 * mismatch_op + 0.10 * source - 0.10 * conflict,
            )
            for matrix, value in zip(matrices, values):
                matrix[i, j] = matrix[j, i] = float(np.clip(value, 0.0, 1.0))
    return np.stack(matrices)


def select_expert_cases(
    difficulty: np.ndarray,
    fraction: float,
    min_selected: int,
    max_selected: int,
) -> np.ndarray:
    n = len(difficulty)
    if fraction <= 0.0 or n == 0:
        return np.zeros(n, dtype=bool)
    count = max(min_selected, int(math.ceil(fraction * n)))
    count = min(n, max_selected if max_selected > 0 else n, count)
    order = np.argsort(-difficulty, kind="mergesort")
    selected = np.zeros(n, dtype=bool)
    selected[order[:count]] = True
    return selected


def expand_with_neighbors(
    selected: np.ndarray,
    mean_dual: np.ndarray,
    neighbors_per_case: int,
    hard_max_selected: int,
) -> np.ndarray:
    if neighbors_per_case <= 0 or not np.any(selected):
        return selected.copy()
    expanded = selected.copy()
    anchors = np.flatnonzero(selected)
    for idx in anchors:
        order = np.argsort(-mean_dual[idx], kind="mergesort")
        added = 0
        for neighbor in order:
            if neighbor == idx:
                continue
            if expanded[neighbor]:
                continue
            expanded[neighbor] = True
            added += 1
            if np.sum(expanded) >= hard_max_selected:
                return expanded
            if added >= neighbors_per_case:
                break
    return expanded


def selective_probability(
    dual: np.ndarray,
    five: np.ndarray,
    selected: np.ndarray,
    beta: float,
    endpoint_policy: str,
) -> tuple[np.ndarray, int]:
    expert = _symmetrize((1.0 - beta) * dual + beta * five)
    if endpoint_policy == "both":
        mask = selected[:, None] & selected[None, :]
    elif endpoint_policy == "either":
        mask = selected[:, None] | selected[None, :]
    else:
        raise ValueError(f"unknown endpoint policy: {endpoint_policy}")
    np.fill_diagonal(mask, False)
    out = dual.copy()
    out[mask] = expert[mask]
    out = _symmetrize(out)
    return out, int(np.sum(np.triu(mask, 1)))


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["policy"], row["selector"], row["fraction"], row["neighbors_per_case"],
            row["endpoint_policy"], row["clusterer"],
        )
        groups[key].append(row)
    output = []
    for key, values in groups.items():
        dataset_scores = {str(row["dataset"]): float(row["BA"]) for row in values}
        bas = list(dataset_scores.values())
        output.append({
            "policy": key[0],
            "selector": key[1],
            "fraction": key[2],
            "neighbors_per_case": key[3],
            "endpoint_policy": key[4],
            "clusterer": key[5],
            "mean_BA": float(np.mean(bas)),
            "worst_BA": float(np.min(bas)),
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])),
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])),
            "mean_selected_fraction": float(np.mean([float(row["selected_fraction"]) for row in values])),
            "mean_pair_coverage": float(np.mean([float(row["expert_pair_fraction"]) for row in values])),
            "datasets": len(values),
            "dataset_BA": json.dumps(dataset_scores, sort_keys=True),
        })
    return sorted(output, key=lambda row: (row["mean_BA"], row["worst_BA"]), reverse=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selective five-view strict-LODO simulation")
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.15, 0.30, 0.50])
    parser.add_argument("--neighbors-per-case", nargs="+", type=int, default=[0])
    parser.add_argument("--endpoint-policies", nargs="+", choices=["both", "either"], default=["both"])
    parser.add_argument("--selectors", nargs="+", choices=["dual", "deterministic"], default=["dual"])
    parser.add_argument("--min-selected", type=int, default=2)
    parser.add_argument("--max-selected", type=int, default=20)
    parser.add_argument("--max-expanded-selected", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.50)
    parser.add_argument("--model-type", default="gbdt")
    parser.add_argument("--graph-tag", default="agglomerative_complete")
    parser.add_argument("--source-clusterer", default="agglomerative_complete")
    parser.add_argument("--clusterers", nargs="+", default=["agglomerative_avg"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prob_dir = resolve(args.prob_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preds").mkdir(exist_ok=True)
    rows: list[dict] = []
    debug_rows: list[dict] = []

    for dataset_arg in args.datasets:
        dataset = resolve(dataset_arg)
        name = dataset.name
        cases = osf.read_cases(dataset / "input.csv")
        gold = read_gold(osf.gold_path(dataset))
        k = len(set(gold))
        dual_stack = load_seed_stack(
            prob_dir, name, "dual", args.seeds, args.model_type, args.graph_tag
        )
        five_stack = load_seed_stack(
            prob_dir, name, "quad_event_object_context", args.seeds,
            args.model_type, args.graph_tag,
        )
        difficulty, components, mean_dual = case_difficulty(
            dual_stack, k, args.source_clusterer
        )
        mean_five = _symmetrize(np.mean(five_stack, axis=0))

        policies = [("dual", "none", 0.0, 0, "both", np.zeros(len(cases), dtype=bool), mean_dual)]
        full = _symmetrize((1.0 - args.beta) * mean_dual + args.beta * mean_five)
        policies.append(("full_five", "none", 1.0, 0, "both", np.ones(len(cases), dtype=bool), full))
        selector_data = {"dual": (difficulty, components, mean_dual)}
        if "deterministic" in args.selectors:
            det_features, _ = plf.build_llm_case_features_for_inputs(
                [dataset / "input.csv"], parser="drain", svd_dim=64,
                llm_args=None, log_llm_disabled=False,
            )
            det_stack = deterministic_proxy_stack(det_features)
            selector_data["deterministic"] = case_difficulty(
                det_stack, k, args.source_clusterer
            )
        for selector in args.selectors:
            selector_difficulty, selector_components, selector_similarity = selector_data[selector]
            for fraction in args.fractions:
                base_selected = select_expert_cases(
                    selector_difficulty, fraction, args.min_selected, args.max_selected
                )
                for neighbors in args.neighbors_per_case:
                    selected = expand_with_neighbors(
                        base_selected, selector_similarity, neighbors, args.max_expanded_selected
                    )
                    for endpoint_policy in args.endpoint_policies:
                        prob, _ = selective_probability(
                            mean_dual, mean_five, selected, args.beta, endpoint_policy
                        )
                        policies.append((
                            "selective", selector, fraction, neighbors,
                            endpoint_policy, selected, prob,
                        ))
            for idx, case in enumerate(cases):
                debug_rows.append({
                    "dataset": name,
                    "case": case,
                    "selector": selector,
                    "difficulty": float(selector_difficulty[idx]),
                    **{key: float(value[idx]) for key, value in selector_components.items()},
                })

        total_pairs = max(1, len(cases) * (len(cases) - 1) // 2)
        for policy, selector, fraction, neighbors, endpoint_policy, selected, prob in policies:
            if policy == "dual":
                expert_pairs = 0
            elif policy == "full_five":
                expert_pairs = total_pairs
            else:
                _, expert_pairs = selective_probability(
                    mean_dual, mean_five, selected, args.beta, endpoint_policy
                )
            for clusterer in args.clusterers:
                result = gc.cluster_probability_graph(prob, k, clusterer)
                pred_path = output_dir / "preds" / (
                    f"{name}_{policy}_f{fraction:.2f}_n{neighbors}_{endpoint_policy}_{clusterer}.csv"
                )
                pred = write_pred(pred_path, cases, result.labels)
                ba, tpr, tnr = pairwise_scores(gold, pred)
                rows.append({
                    "dataset": name,
                    "policy": policy,
                    "selector": selector,
                    "fraction": fraction,
                    "neighbors_per_case": neighbors,
                    "endpoint_policy": endpoint_policy,
                    "clusterer": clusterer,
                    "BA": ba,
                    "TPR": tpr,
                    "TNR": tnr,
                    "cases": len(cases),
                    "k": k,
                    "selected_cases": int(np.sum(selected)),
                    "selected_fraction": float(np.mean(selected)),
                    "extra_embedding_docs": int(np.sum(selected) * 3),
                    "expert_pairs": expert_pairs,
                    "expert_pair_fraction": expert_pairs / total_pairs,
                    "pred_path": str(pred_path),
                })
                print(
                    f"[selective] dataset={name} policy={policy} fraction={fraction:.2f} "
                    f"selector={selector} selected={int(np.sum(selected))}/{len(cases)} "
                    f"pair_coverage={expert_pairs/total_pairs:.3f} "
                    f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                    flush=True,
                )

    write_csv(output_dir / "results.csv", rows, sorted({key for row in rows for key in row}))
    summary = summarize(rows)
    write_csv(output_dir / "summary.csv", summary, sorted({key for row in summary for key in row}))
    write_csv(
        output_dir / "difficulty_debug.csv", debug_rows,
        ["dataset", "case", "selector", "difficulty", "entropy", "disagreement", "instability", "margin_difficulty"],
    )
    print("\n| rank | policy | selector | fraction | neighbors | endpoints | mean BA | worst BA | selected | pair coverage |")
    print("|---:|---|---|---:|---:|---|---:|---:|---:|---:|")
    for rank, row in enumerate(summary[:20], 1):
        print(
            f"| {rank} | {row['policy']} | {row['selector']} | {row['fraction']:.2f} | {row['neighbors_per_case']} | "
            f"{row['endpoint_policy']} | {row['mean_BA']:.4f} | {row['worst_BA']:.4f} | "
            f"{row['mean_selected_fraction']:.3f} | {row['mean_pair_coverage']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
