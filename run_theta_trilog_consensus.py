#!/usr/bin/env python3
"""Label-safe episode-level consensus evaluation for Theta v3 TriLog.

This experimental runner never changes the formal predictor.  Per-episode
pair models must already have been trained in strict LODO mode.  The runner
turns their fixed-k partitions into a seed co-association matrix and blends
that graph with an existing no-trace probability matrix.  For nested LODO
selection, a target episode's gold is excluded while choosing the blend and
clusterer; its gold is used only for the final reported score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

import graph_clustering as gc
import official_style_features as osf
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("official_format_fake_dataset/directed_cross_v4"),
    Path("official_format_fake_dataset/benchmark5_final"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]
OFFICIAL_NAMES = {"benchmark_set_1", "benchmark_set_2"}


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _find_probability(roots: Sequence[Path], filename: str) -> Path | None:
    for root in roots:
        for candidate in (root / filename, root / "probs" / filename):
            if candidate.exists():
                return candidate
    return None


def _trace_probability_name(dataset: str, config: str, seed: int) -> str:
    return f"{dataset}_{config}_gbdt_agglomerative_avg_seed{seed}.npy"


def _coassociation(probabilities: Sequence[np.ndarray], k: int) -> np.ndarray:
    if not probabilities:
        raise ValueError("at least one TriLog probability matrix is required")
    labels = [
        np.asarray(gc.agglomerative_avg(probability, k).labels, dtype=np.int32)
        for probability in probabilities
    ]
    output = np.mean(
        [label[:, None] == label[None, :] for label in labels], axis=0
    ).astype(np.float32)
    np.fill_diagonal(output, 1.0)
    return output


def _soften_coassociation(matrix: np.ndarray, epsilon: float) -> np.ndarray:
    epsilon = float(np.clip(epsilon, 0.0, 0.49))
    output = epsilon + (1.0 - 2.0 * epsilon) * np.asarray(matrix, dtype=np.float32)
    np.fill_diagonal(output, 1.0)
    return output.astype(np.float32, copy=False)


def _selected_method(result: gc.GraphClusterResult) -> str:
    for row in reversed(result.trajectory):
        if row.get("action") == "quality_selected":
            return str(row.get("candidate", ""))
    return result.method


def _candidate_key(row: dict) -> tuple[float, str]:
    return float(row["blend_weight"]), str(row["clusterer"])


def _choose_nested_candidate(
    candidates: Sequence[dict],
    target: str,
    official_source_weight: float,
) -> tuple[tuple[float, str], float]:
    grouped: dict[tuple[float, str], list[tuple[float, float]]] = {}
    for row in candidates:
        dataset = str(row["dataset"])
        if dataset == target:
            continue
        weight = float(official_source_weight) if dataset in OFFICIAL_NAMES else 1.0
        grouped.setdefault(_candidate_key(row), []).append((float(row["BA"]), weight))
    if not grouped:
        raise ValueError(f"no source episodes available to select candidate for {target}")
    scored: list[tuple[float, float, int, tuple[float, str]]] = []
    for key, values in grouped.items():
        numerator = sum(value * weight for value, weight in values)
        denominator = sum(weight for _, weight in values)
        score = numerator / max(denominator, 1e-12)
        # Conservative tie-break: less trace influence, then stable CLI order.
        scored.append((score, -key[0], -len(key[1]), key))
    best = max(scored, key=lambda item: item[:3])
    return best[3], float(best[0])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Theta TriLog seed-consensus evaluation")
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--base-prob-dir", type=Path, required=True)
    parser.add_argument("--trace-output-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-config", default="trilog_residual")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--blend-weights", nargs="+", type=float, default=[0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2])
    parser.add_argument("--clusterers", nargs="+", default=["agglomerative_avg", "quality_selected"])
    parser.add_argument("--coassoc-epsilon", type=float, default=0.1)
    parser.add_argument("--official-source-weight", type=float, default=4.0)
    parser.add_argument("--allow-partial-seeds", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = [resolve(path) for path in args.datasets]
    base_root = resolve(args.base_prob_dir)
    trace_roots = [resolve(path) for path in args.trace_output_dirs]
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict] = []
    episode_debug: list[dict] = []
    episode_payload: dict[str, dict] = {}
    for dataset in datasets:
        name = dataset.name
        labels = read_gold(osf.gold_path(dataset))
        cases = osf.read_cases(dataset / "input.csv")
        k = len(set(labels))
        base_path = _find_probability([base_root], f"{name}_seed_mean.npy")
        if base_path is None:
            raise FileNotFoundError(f"no base probability for {name} under {base_root}")
        base_probability = np.asarray(np.load(base_path), dtype=np.float32)
        trace_probabilities: list[np.ndarray] = []
        trace_paths: list[str] = []
        missing_seeds: list[int] = []
        for seed in args.seeds:
            path = _find_probability(trace_roots, _trace_probability_name(name, args.trace_config, seed))
            if path is None:
                missing_seeds.append(seed)
                continue
            probability = np.asarray(np.load(path), dtype=np.float32)
            if probability.shape != base_probability.shape:
                raise ValueError(
                    f"probability shape mismatch for {name} seed {seed}: "
                    f"{probability.shape} != {base_probability.shape}"
                )
            trace_probabilities.append(probability)
            trace_paths.append(str(path))
        if missing_seeds and not args.allow_partial_seeds:
            raise FileNotFoundError(f"{name} is missing TriLog seeds {missing_seeds}")
        if not trace_probabilities:
            raise FileNotFoundError(f"{name} has no usable TriLog probabilities")
        consensus = _soften_coassociation(
            _coassociation(trace_probabilities, k), args.coassoc_epsilon
        )
        episode_payload[name] = {
            "dataset": dataset,
            "labels": labels,
            "cases": cases,
            "k": k,
            "base": base_probability,
            "consensus": consensus,
        }
        episode_debug.append({
            "dataset": name,
            "cases": len(labels),
            "k": k,
            "base_prob_path": str(base_path),
            "trace_seeds_requested": ";".join(map(str, args.seeds)),
            "trace_seeds_loaded": len(trace_probabilities),
            "missing_seeds": ";".join(map(str, missing_seeds)),
            "trace_prob_paths": json.dumps(trace_paths),
            "coassoc_mean": float(np.mean(consensus)),
            "coassoc_std": float(np.std(consensus)),
        })
        for blend_weight in args.blend_weights:
            probability = (
                (1.0 - float(blend_weight)) * base_probability
                + float(blend_weight) * consensus
            ).astype(np.float32)
            np.fill_diagonal(probability, 1.0)
            for clusterer in args.clusterers:
                result = gc.cluster_probability_graph(probability, k, clusterer)
                ba, tpr, tnr = pairwise_scores(labels, result.labels)
                candidates.append({
                    "dataset": name,
                    "blend_weight": float(blend_weight),
                    "clusterer": clusterer,
                    "selected_clusterer": _selected_method(result),
                    "BA": ba,
                    "TPR": tpr,
                    "TNR": tnr,
                    "num_clusters": len(set(result.labels)),
                    "trace_seeds": len(trace_probabilities),
                })

    selected_rows: list[dict] = []
    for name, payload in episode_payload.items():
        key, source_score = _choose_nested_candidate(
            candidates, name, args.official_source_weight
        )
        blend_weight, clusterer = key
        probability = (
            (1.0 - blend_weight) * payload["base"]
            + blend_weight * payload["consensus"]
        ).astype(np.float32)
        np.fill_diagonal(probability, 1.0)
        result = gc.cluster_probability_graph(probability, payload["k"], clusterer)
        pred_path = output_dir / "preds" / f"{name}_nested_consensus.csv"
        pred = write_pred(pred_path, payload["cases"], result.labels)
        prob_path = output_dir / "probs" / f"{name}_nested_consensus.npy"
        prob_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prob_path, probability)
        ba, tpr, tnr = pairwise_scores(payload["labels"], pred)
        selected_rows.append({
            "dataset": name,
            "selection_protocol": "nested_episode_lodo",
            "blend_weight": blend_weight,
            "clusterer": clusterer,
            "selected_clusterer": _selected_method(result),
            "source_selection_BA": source_score,
            "BA": ba,
            "TPR": tpr,
            "TNR": tnr,
            "cases": len(payload["labels"]),
            "k": payload["k"],
            "num_clusters": len(set(pred)),
            "pred_path": str(pred_path),
            "prob_path": str(prob_path),
        })

    write_csv(output_dir / "candidates.csv", candidates, list(candidates[0]))
    write_csv(output_dir / "results.csv", selected_rows, list(selected_rows[0]))
    write_csv(output_dir / "episode_debug.csv", episode_debug, list(episode_debug[0]))
    macro = float(np.mean([float(row["BA"]) for row in selected_rows]))
    official = float(np.mean([
        float(row["BA"]) for row in selected_rows if row["dataset"] in OFFICIAL_NAMES
    ]))
    summary = [{
        "protocol": "nested_episode_lodo",
        "macro_BA": macro,
        "official_mean_BA": official,
        "worst_BA": float(np.min([float(row["BA"]) for row in selected_rows])),
        "datasets": len(selected_rows),
    }]
    write_csv(output_dir / "summary.csv", summary, list(summary[0]))
    print("\n| dataset | beta | clusterer | selected | BA | TPR | TNR |")
    print("|---|---:|---|---|---:|---:|---:|")
    for row in selected_rows:
        print(
            f"| {row['dataset']} | {row['blend_weight']:.3f} | {row['clusterer']} | "
            f"{row['selected_clusterer']} | {row['BA']:.4f} | {row['TPR']:.4f} | {row['TNR']:.4f} |"
        )
    print(f"macro_BA={macro:.6f} official_mean_BA={official:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
