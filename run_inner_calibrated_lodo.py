#!/usr/bin/env python3
"""Cross-domain inner calibration for strict LODO pair probabilities.

For each target dataset and seed, parameters are selected only from OOF
predictions on the other datasets. Target gold is used solely for final scoring.
This is experimental and does not modify the formal predictor.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_rows(paths: Sequence[Path]) -> list[dict]:
    rows: list[dict] = []
    for directory in paths:
        path = directory / "results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def transform_probability(gated: np.ndarray, residual: np.ndarray, beta: float, temperature: float, bias: float) -> np.ndarray:
    mixed = float(beta) * gated.astype(np.float64) + (1.0 - float(beta)) * residual.astype(np.float64)
    clipped = np.clip(mixed, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped)) / max(float(temperature), 1e-6) + float(bias)
    output = 1.0 / (1.0 + np.exp(-logits))
    output = (output + output.T) * 0.5
    np.fill_diagonal(output, 1.0)
    return output.astype(np.float32)


def effective_k(k: int, factor: float, n: int) -> int:
    return max(1, min(n, int(round(float(k) * float(factor)))))


def score_probability(prob: np.ndarray, gold: list[str], factor: float) -> tuple[float, float, float, list[int]]:
    labels = plf.cluster_from_probability(prob, effective_k(len(set(gold)), factor, len(gold)))
    pred = [f"bucket_{value:03d}" for value in labels]
    ba, tpr, tnr = pairwise_scores(gold, pred)
    return ba, tpr, tnr, labels


def cache_key(seed: int, dataset: str, beta: float, temperature: float, bias: float, factor: float) -> tuple:
    return (seed, dataset, beta, temperature, bias, factor)


def choose_params(
    seed: int,
    target: str,
    datasets: Sequence[str],
    probabilities: dict[tuple[int, str, str], np.ndarray],
    golds: dict[str, list[str]],
    beta_grid: Sequence[float],
    temperature_grid: Sequence[float],
    bias_grid: Sequence[float],
    factor_grid: Sequence[float],
    score_cache: dict[tuple, tuple[float, float, float]],
) -> tuple[dict, list[dict]]:
    inner = [name for name in datasets if name != target]
    candidates: list[tuple] = []
    detail_rows: list[dict] = []
    for beta in beta_grid:
        for temperature in temperature_grid:
            for bias in bias_grid:
                for factor in factor_grid:
                    scores = []
                    tprs = []
                    tnrs = []
                    for dataset in inner:
                        key = cache_key(seed, dataset, beta, temperature, bias, factor)
                        if key not in score_cache:
                            prob = transform_probability(
                                probabilities[(seed, dataset, "gated_mlp")],
                                probabilities[(seed, dataset, "res_mlp")],
                                beta, temperature, bias,
                            )
                            ba, tpr, tnr, _ = score_probability(prob, golds[dataset], factor)
                            score_cache[key] = (ba, tpr, tnr)
                        ba, tpr, tnr = score_cache[key]
                        scores.append(ba); tprs.append(tpr); tnrs.append(tnr)
                    mean_ba = float(np.mean(scores))
                    min_ba = float(np.min(scores))
                    balance = -abs(float(np.mean(tprs)) - float(np.mean(tnrs)))
                    # Prefer less correction when validation scores tie.
                    simplicity = -(
                        abs(beta - 0.75) + abs(temperature - 1.0)
                        + abs(bias) + abs(factor - 1.0)
                    )
                    candidates.append((mean_ba, min_ba, balance, simplicity, beta, temperature, bias, factor))
    best = max(candidates)
    selected = {
        "inner_mean_BA": best[0], "inner_min_BA": best[1],
        "beta": best[4], "temperature": best[5], "bias": best[6],
        "cluster_factor": best[7],
    }
    for dataset in inner:
        ba, tpr, tnr = score_cache[cache_key(
            seed, dataset, selected["beta"], selected["temperature"],
            selected["bias"], selected["cluster_factor"]
        )]
        detail_rows.append({
            "seed": seed, "target_dataset": target, "inner_dataset": dataset,
            **selected, "BA": ba, "TPR": tpr, "TNR": tnr,
        })
    return selected, detail_rows


def error_delta(gold: Sequence[str], base_labels: Sequence[int], candidate_labels: Sequence[int]) -> dict[str, int]:
    values = {"fixed_FP": 0, "new_FP": 0, "fixed_FN": 0, "new_FN": 0}
    for i in range(len(gold)):
        for j in range(i + 1, len(gold)):
            truth = gold[i] == gold[j]
            before = base_labels[i] == base_labels[j]
            after = candidate_labels[i] == candidate_labels[j]
            if before == after:
                continue
            if truth and not before and after: values["fixed_FN"] += 1
            elif truth and before and not after: values["new_FN"] += 1
            elif not truth and before and not after: values["fixed_FP"] += 1
            elif not truth and not before and after: values["new_FP"] += 1
    return values


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows: groups[str(row["dataset"])].append(row)
    output = []
    for dataset, values in sorted(groups.items()):
        bas = np.asarray([float(x["BA"]) for x in values])
        output.append({
            "dataset": dataset, "seeds": len(values),
            "mean_BA": float(np.mean(bas)), "std_BA": float(np.std(bas, ddof=1)) if len(bas) > 1 else 0.0,
            "min_BA": float(np.min(bas)), "max_BA": float(np.max(bas)),
            "mean_TPR": float(np.mean([float(x["TPR"]) for x in values])),
            "mean_TNR": float(np.mean([float(x["TNR"]) for x in values])),
            "mean_inner_BA": float(np.mean([float(x["inner_mean_BA"]) for x in values])),
        })
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inner-calibrate strict LODO gated/residual probabilities")
    p.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--betas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    p.add_argument("--biases", nargs="+", type=float, default=[-0.25, 0.0, 0.25])
    p.add_argument("--cluster-factors", nargs="+", type=float, default=[0.8, 0.9, 1.0, 1.1, 1.2])
    return p.parse_args()


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [resolve(path) for path in args.datasets]
    dataset_by_name = {path.name: path for path in datasets}
    names = list(dataset_by_name)
    raw_rows = read_rows(args.result_dirs)
    probabilities: dict[tuple[int, str, str], np.ndarray] = {}
    row_lookup: dict[tuple[int, str, str], dict] = {}
    for row in raw_rows:
        seed = int(row["seed"]); name = str(row["holdout_dataset"]); arch = str(row.get("model_arch", ""))
        if seed not in args.seeds or name not in dataset_by_name or arch not in {"res_mlp", "gated_mlp"}:
            continue
        probabilities[(seed, name, arch)] = np.load(row["prob_path"])
        row_lookup[(seed, name, arch)] = row
    missing = [(seed, name, arch) for seed in args.seeds for name in names for arch in ("res_mlp", "gated_mlp") if (seed, name, arch) not in probabilities]
    if missing:
        raise RuntimeError(f"missing probability matrices ({len(missing)}): {missing[:8]}")
    golds = {name: read_gold(osf.gold_path(path)) for name, path in dataset_by_name.items()}
    cases = {name: osf.read_cases(path / "input.csv") for name, path in dataset_by_name.items()}
    score_cache: dict[tuple, tuple[float, float, float]] = {}
    results = []; choices = []; inner_rows = []; deltas = []
    for seed in args.seeds:
        for target in names:
            selected, detail = choose_params(
                seed, target, names, probabilities, golds,
                args.betas, args.temperatures, args.biases, args.cluster_factors, score_cache,
            )
            inner_rows.extend(detail)
            final_prob = transform_probability(
                probabilities[(seed, target, "gated_mlp")],
                probabilities[(seed, target, "res_mlp")],
                selected["beta"], selected["temperature"], selected["bias"],
            )
            ba, tpr, tnr, labels = score_probability(final_prob, golds[target], selected["cluster_factor"])
            base_prob = probabilities[(seed, target, "gated_mlp")]
            _, _, _, base_labels = score_probability(base_prob, golds[target], 1.0)
            pred_path = args.output_dir / "preds" / f"{target}_seed{seed}.csv"
            prob_path = args.output_dir / "probs" / f"{target}_seed{seed}.npy"
            write_pred(pred_path, cases[target], labels); prob_path.parent.mkdir(parents=True, exist_ok=True); np.save(prob_path, final_prob)
            row = {
                "seed": seed, "dataset": target, **selected,
                "BA": ba, "TPR": tpr, "TNR": tnr,
                "reference_k": len(set(golds[target])),
                "selected_k": effective_k(len(set(golds[target])), selected["cluster_factor"], len(golds[target])),
                "pred_path": str(pred_path), "prob_path": str(prob_path),
            }
            results.append(row); choices.append({"seed": seed, "dataset": target, **selected})
            deltas.append({"seed": seed, "dataset": target, **error_delta(golds[target], base_labels, labels)})
            print(f"[calibrated] seed={seed} dataset={target} BA={ba:.6f} beta={selected['beta']:.2f} temp={selected['temperature']:.2f} bias={selected['bias']:+.2f} factor={selected['cluster_factor']:.2f}", flush=True)
    write_csv(args.output_dir / "results.csv", results, list(results[0]))
    summary = summarize(results); write_csv(args.output_dir / "summary.csv", summary, list(summary[0]))
    write_csv(args.output_dir / "calibration_choices.csv", choices, list(choices[0]))
    write_csv(args.output_dir / "inner_validation.csv", inner_rows, list(inner_rows[0]))
    write_csv(args.output_dir / "error_deltas.csv", deltas, list(deltas[0]))
    manifest = {"result_dirs": [str(x) for x in args.result_dirs], "seeds": args.seeds, "betas": args.betas, "temperatures": args.temperatures, "biases": args.biases, "cluster_factors": args.cluster_factors, "target_gold_used_for_selection": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("| dataset | BA | std | min | TPR | TNR |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in summary: print(f"| {row['dataset']} | {row['mean_BA']:.4f} | {row['std_BA']:.4f} | {row['min_BA']:.4f} | {row['mean_TPR']:.4f} | {row['mean_TNR']:.4f} |")
    print(f"macro mean BA={np.mean([float(x['mean_BA']) for x in summary]):.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
