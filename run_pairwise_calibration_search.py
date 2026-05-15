#!/usr/bin/env python3
"""Search probability calibration settings for an existing pairwise_mlp model.

This script is experimental and reads gold.csv only for validation. It does not
change the official predictor defaults.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pairwise_features as pf
from run_experiments import pairwise_scores, read_gold
from run_half_split_experiments import DEFAULT_DATASETS, opposite_part, part_for_bit, stratified_half_split
from train_pairwise_mlp import cluster_from_probability


def load_model(model_path: Path, device: str):
    import torch

    device = pf.resolve_torch_device(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = pf.build_pairwise_mlp_model(
        int(checkpoint["input_dim"]),
        hidden_dims=checkpoint.get("hidden_dims", (256, 128)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint, device


def validation_parts(args: argparse.Namespace) -> list[dict]:
    parts = []
    split_root = args.output_dir / "splits" / f"seed_{args.seed}"
    for idx, dataset in enumerate(args.datasets):
        dataset = Path(dataset)
        splits = stratified_half_split(dataset, args.seed, split_root)
        train_part = part_for_bit((args.combo >> idx) & 1)
        val_part = opposite_part(train_part)
        info = dict(splits[val_part])
        info["dataset"] = dataset.name
        info["val_part"] = val_part
        parts.append(info)
    return parts


def evaluate_calibration(raw_prob: np.ndarray, features: list[pf.CaseFeature], gold: Sequence[str], k: int, params: dict) -> dict:
    prob = pf.calibrate_probability_matrix(
        raw_prob,
        features,
        primary_floor=params["primary_floor"],
        op_pair_floor=params["op_pair_floor"],
        mismatch_floor=params["mismatch_floor"],
        conflict_penalty=params["conflict_penalty"],
        cosine_gate=params["mismatch_cosine_gate"],
    )
    labels = cluster_from_probability(prob, k)
    pred = [f"bucket_{label:03d}" for label in labels]
    ba, tpr, tnr = pairwise_scores(gold, pred)
    return {"BA": ba, "TPR": tpr, "TNR": tnr, "num_pred_clusters": len(set(pred))}


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    key_fields = ("primary_floor", "op_pair_floor", "mismatch_floor", "conflict_penalty", "mismatch_cosine_gate")
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)
    summary = []
    for key, items in groups.items():
        bas = [float(row["BA"]) for row in items]
        tprs = [float(row["TPR"]) for row in items]
        tnrs = [float(row["TNR"]) for row in items]
        entry = {field: value for field, value in zip(key_fields, key)}
        entry.update(
            {
                "mean_BA": statistics.mean(bas),
                "mean_TPR": statistics.mean(tprs),
                "mean_TNR": statistics.mean(tnrs),
                "min_BA": min(bas),
                "num_parts": len(items),
            }
        )
        summary.append(entry)
    summary.sort(key=lambda row: (row["mean_BA"], row["min_BA"], row["mean_TPR"]), reverse=True)
    return summary


def print_top(summary: Sequence[dict], limit: int) -> None:
    print("| rank | primary | op_pair | mismatch | conflict | cosine_gate | mean_BA | mean_TPR | mean_TNR | min_BA |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(summary[:limit], start=1):
        print(
            f"| {idx} | {row['primary_floor']:.2f} | {row['op_pair_floor']:.2f} | "
            f"{row['mismatch_floor']:.2f} | {row['conflict_penalty']:.2f} | "
            f"{row['mismatch_cosine_gate']:.2f} | {row['mean_BA']:.6f} | "
            f"{row['mean_TPR']:.6f} | {row['mean_TNR']:.6f} | {row['min_BA']:.6f} |"
        )


def parse_float_list(values: Sequence[str]) -> list[float]:
    return [float(value) for value in values]


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    model, checkpoint, device = load_model(args.model, args.device)
    svd_dim = int(checkpoint.get("svd_dim", args.svd_dim))
    parts = validation_parts(args)
    cached = []
    for part in parts:
        features, _ = pf.build_case_features(part["input"], parser="drain", svd_dim=svd_dim)
        raw_prob = pf.predict_probability_matrix(
            model,
            features,
            device=device,
            batch_size=args.predict_batch_size,
            prob_bias=args.prob_bias,
            prob_temperature=args.prob_temperature,
        )
        cached.append((part, features, raw_prob, read_gold(part["gold"])))

    rows = []
    grid = itertools.product(
        args.primary_floors,
        args.op_pair_floors,
        args.mismatch_floors,
        args.conflict_penalties,
        args.mismatch_cosine_gates,
    )
    for primary_floor, op_pair_floor, mismatch_floor, conflict_penalty, mismatch_cosine_gate in grid:
        params = {
            "primary_floor": primary_floor,
            "op_pair_floor": op_pair_floor,
            "mismatch_floor": mismatch_floor,
            "conflict_penalty": conflict_penalty,
            "mismatch_cosine_gate": mismatch_cosine_gate,
        }
        for part, features, raw_prob, gold in cached:
            metrics = evaluate_calibration(raw_prob, features, gold, part["k"], params)
            rows.append(
                {
                    "dataset": part["dataset"],
                    "val_part": part["val_part"],
                    "k": part["k"],
                    "num_cases": part["num_cases"],
                    **params,
                    **metrics,
                }
            )
    summary = summarize(rows)
    write_csv(args.output_dir / "calibration_results.csv", rows)
    write_csv(args.output_dir / "calibration_summary.csv", summary)
    (args.output_dir / "best_calibration.json").write_text(json.dumps(summary[0], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search pairwise_mlp calibration parameters.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--combo", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pairwise_calibration_search"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--svd-dim", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=100000)
    parser.add_argument("--prob-bias", type=float, default=0.0)
    parser.add_argument("--prob-temperature", type=float, default=1.0)
    parser.add_argument("--primary-floors", nargs="+", type=float, default=[0.0, 0.45, 0.55, 0.65, 0.70])
    parser.add_argument("--op-pair-floors", nargs="+", type=float, default=[0.0, 0.35, 0.45, 0.55, 0.65])
    parser.add_argument("--mismatch-floors", nargs="+", type=float, default=[0.0, 0.25, 0.35, 0.45, 0.55])
    parser.add_argument("--conflict-penalties", nargs="+", type=float, default=[0.0, 0.03, 0.05, 0.08])
    parser.add_argument("--mismatch-cosine-gates", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, summary = run(args)
    print_top(summary, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
