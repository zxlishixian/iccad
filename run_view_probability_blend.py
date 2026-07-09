#!/usr/bin/env python3
"""Blend saved dual/multi-view probability matrices.

Experimental only. This script reuses probability matrices produced by
``run_graph_multiview_experiments.py`` and does not modify the formal
``regr_fail_bucketing.py`` predictor.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("official_format_fake_dataset/directed_cross_v2"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["expert_view"]), float(row["beta"]), str(row["graph_method"]))].append(row)
    out: list[dict] = []
    for (expert, beta, graph_method), values in groups.items():
        dataset_means = {}
        for dataset in sorted({str(v["dataset"]) for v in values}):
            subset = [v for v in values if v["dataset"] == dataset]
            dataset_means[dataset] = float(np.mean([float(v["BA"]) for v in subset]))
        bas = list(dataset_means.values())
        out.append({
            "expert_view": expert,
            "beta": beta,
            "graph_method": graph_method,
            "mean_BA": float(np.mean(bas)) if bas else 0.0,
            "std_BA": float(np.std(bas)) if bas else 0.0,
            "worst_BA": float(np.min(bas)) if bas else 0.0,
            "mean_TPR": float(np.mean([float(v["TPR"]) for v in values])) if values else 0.0,
            "mean_TNR": float(np.mean([float(v["TNR"]) for v in values])) if values else 0.0,
            "datasets": len(dataset_means),
            "runs": len(values),
            "dataset_means": json.dumps(dataset_means, sort_keys=True),
        })
    return sorted(out, key=lambda r: (r["mean_BA"], r["worst_BA"]), reverse=True)


def _dataset_means(rows: Sequence[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        subset = [row for row in rows if str(row["dataset"]) == dataset]
        out[dataset] = float(np.mean([float(row["BA"]) for row in subset]))
    return out


def _candidate_score(rows: Sequence[dict], target_dataset: str, policy: str) -> tuple[float, float, float]:
    train_rows = [row for row in rows if str(row["dataset"]) != target_dataset]
    means = list(_dataset_means(train_rows).values())
    if not means:
        return (float("-inf"), float("-inf"), float("-inf"))
    mean_ba = float(np.mean(means))
    worst_ba = float(np.min(means))
    std_ba = float(np.std(means))
    if policy == "lodo_worst":
        return (worst_ba, mean_ba, -std_ba)
    if policy == "lodo_guarded":
        return (mean_ba - 0.5 * std_ba, worst_ba, mean_ba)
    return (mean_ba, worst_ba, -std_ba)


def select_lodo_rows(rows: Sequence[dict], policy: str, joint_expert: bool) -> tuple[list[dict], list[dict]]:
    selected: list[dict] = []
    debug: list[dict] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    experts = sorted({str(row["expert_view"]) for row in rows})
    for target in datasets:
        expert_groups = [experts] if joint_expert else [[expert] for expert in experts]
        for expert_group in expert_groups:
            candidates: list[tuple[tuple[float, float, float], str, float, list[dict]]] = []
            for expert in expert_group:
                betas = sorted({float(row["beta"]) for row in rows if str(row["expert_view"]) == expert})
                for beta in betas:
                    subset = [
                        row for row in rows
                        if str(row["expert_view"]) == expert and abs(float(row["beta"]) - beta) < 1e-9
                    ]
                    candidates.append((_candidate_score(subset, target, policy), expert, beta, subset))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            score, expert, beta, subset = candidates[0]
            target_rows = [row for row in subset if str(row["dataset"]) == target]
            label = "joint" if joint_expert else expert
            for row in target_rows:
                item = dict(row)
                item["selection_policy"] = policy
                item["selection_mode"] = "joint_expert_beta" if joint_expert else "per_expert_beta"
                item["selected_expert"] = expert
                item["selected_beta"] = beta
                selected.append(item)
            debug.append({
                "target_dataset": target,
                "selection_policy": policy,
                "selection_mode": "joint_expert_beta" if joint_expert else "per_expert_beta",
                "group": label,
                "selected_expert": expert,
                "selected_beta": beta,
                "score_0": score[0],
                "score_1": score[1],
                "score_2": score[2],
                "train_dataset_means": json.dumps(_dataset_means([
                    row for row in subset if str(row["dataset"]) != target
                ]), sort_keys=True),
            })
    return selected, debug


def summarize_selected(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(
            str(row.get("selection_policy", "")),
            str(row.get("selection_mode", "")),
            str(row.get("selected_expert", "")),
        )].append(row)
    out: list[dict] = []
    for (policy, mode, expert), values in groups.items():
        dataset_means = _dataset_means(values)
        bas = list(dataset_means.values())
        betas = sorted({float(row["selected_beta"]) for row in values})
        out.append({
            "selection_policy": policy,
            "selection_mode": mode,
            "selected_expert": expert,
            "selected_betas": json.dumps(betas),
            "mean_BA": float(np.mean(bas)) if bas else 0.0,
            "std_BA": float(np.std(bas)) if bas else 0.0,
            "worst_BA": float(np.min(bas)) if bas else 0.0,
            "mean_TPR": float(np.mean([float(row["TPR"]) for row in values])) if values else 0.0,
            "mean_TNR": float(np.mean([float(row["TNR"]) for row in values])) if values else 0.0,
            "datasets": len(dataset_means),
            "runs": len(values),
            "dataset_means": json.dumps(dataset_means, sort_keys=True),
        })
    return sorted(out, key=lambda row: (row["mean_BA"], row["worst_BA"]), reverse=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend cached dual/multi-view probability matrices")
    parser.add_argument("--prob-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--base-view", default="dual")
    parser.add_argument("--expert-views", nargs="+", default=["quad_event_object_context"])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--model-type", default="gbdt")
    parser.add_argument("--source-graph-method", default="agglomerative_complete")
    parser.add_argument("--graph-method", default="agglomerative_complete")
    parser.add_argument("--selection-policies", nargs="+", default=["lodo_mean", "lodo_guarded", "lodo_worst"])
    parser.add_argument("--selection-modes", nargs="+", default=["per_expert", "joint"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prob_dir = resolve(args.prob_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preds").mkdir(exist_ok=True)
    (args.output_dir / "probs").mkdir(exist_ok=True)

    rows: list[dict] = []
    for dataset_arg in args.datasets:
        dataset = resolve(dataset_arg)
        name = dataset.name
        cases = osf.read_cases(dataset / "input.csv")
        gold = read_gold(osf.gold_path(dataset))
        k = len(set(gold))
        for seed in args.seeds:
            base_path = prob_dir / f"{name}_{args.base_view}_{args.model_type}_{args.source_graph_method}_seed{seed}.npy"
            if not base_path.exists():
                print(f"[blend] missing base prob: {base_path}", flush=True)
                continue
            p_base = np.load(base_path).astype(np.float32)
            for expert in args.expert_views:
                expert_path = prob_dir / f"{name}_{expert}_{args.model_type}_{args.source_graph_method}_seed{seed}.npy"
                if not expert_path.exists():
                    print(f"[blend] missing expert prob: {expert_path}", flush=True)
                    continue
                p_expert = np.load(expert_path).astype(np.float32)
                for beta in args.betas:
                    prob = ((1.0 - beta) * p_base + beta * p_expert).astype(np.float32)
                    prob = (prob + prob.T) * 0.5
                    np.fill_diagonal(prob, 1.0)
                    result = gc.cluster_probability_graph(prob, k, args.graph_method)
                    pred_path = (
                        args.output_dir / "preds" /
                        f"{name}_blend_{expert}_b{beta:.2f}_{args.graph_method}_seed{seed}.csv"
                    )
                    pred = write_pred(pred_path, cases, result.labels)
                    prob_path = (
                        args.output_dir / "probs" /
                        f"{name}_blend_{expert}_b{beta:.2f}_{args.graph_method}_seed{seed}.npy"
                    )
                    np.save(prob_path, prob)
                    ba, tpr, tnr = pairwise_scores(gold, pred)
                    rows.append({
                        "dataset": name,
                        "seed": seed,
                        "base_view": args.base_view,
                        "expert_view": expert,
                        "beta": beta,
                        "graph_method": args.graph_method,
                        "BA": ba,
                        "TPR": tpr,
                        "TNR": tnr,
                        "k": k,
                        "cases": len(gold),
                        "num_clusters": len(set(pred)),
                        "pred_path": str(pred_path),
                        "prob_path": str(prob_path),
                    })
                    print(
                        f"[blend] dataset={name} seed={seed} expert={expert} beta={beta:.2f} "
                        f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                        flush=True,
                    )

    result_fields = [
        "dataset", "seed", "base_view", "expert_view", "beta", "graph_method",
        "BA", "TPR", "TNR", "k", "cases", "num_clusters", "pred_path", "prob_path",
    ]
    write_csv(args.output_dir / "results.csv", rows, result_fields)
    summary = summarize(rows)
    summary_fields = [
        "expert_view", "beta", "graph_method", "mean_BA", "std_BA",
        "worst_BA", "mean_TPR", "mean_TNR", "datasets", "runs", "dataset_means",
    ]
    write_csv(args.output_dir / "summary.csv", summary, summary_fields)
    selected_rows: list[dict] = []
    selected_debug: list[dict] = []
    for policy in args.selection_policies:
        for mode in args.selection_modes:
            rows_sel, debug_sel = select_lodo_rows(rows, policy, joint_expert=(mode == "joint"))
            selected_rows.extend(rows_sel)
            selected_debug.extend(debug_sel)
    if selected_rows:
        selected_fields = result_fields + ["selection_policy", "selection_mode", "selected_expert", "selected_beta"]
        write_csv(args.output_dir / "selected_results.csv", selected_rows, selected_fields)
        selected_summary = summarize_selected(selected_rows)
        selected_summary_fields = [
            "selection_policy", "selection_mode", "selected_expert", "selected_betas",
            "mean_BA", "std_BA", "worst_BA", "mean_TPR", "mean_TNR",
            "datasets", "runs", "dataset_means",
        ]
        write_csv(args.output_dir / "selected_summary.csv", selected_summary, selected_summary_fields)
        if selected_debug:
            write_csv(args.output_dir / "selection_debug.csv", selected_debug, sorted({k for row in selected_debug for k in row}))

    if summary:
        print("\n| rank | expert | beta | mean BA | worst BA | TPR | TNR |")
        print("|---:|---|---:|---:|---:|---:|---:|")
        for rank, row in enumerate(summary[:20], 1):
            print(
                f"| {rank} | {row['expert_view']} | {row['beta']:.2f} | "
                f"{row['mean_BA']:.4f} | {row['worst_BA']:.4f} | "
                f"{row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
