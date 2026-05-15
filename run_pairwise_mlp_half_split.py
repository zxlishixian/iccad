#!/usr/bin/env python3
"""Half-split baseline vs experimental pairwise_mlp comparison."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from run_experiments import pairwise_scores, read_gold, read_pred
from run_half_split_experiments import DEFAULT_DATASETS, opposite_part, part_for_bit, stratified_half_split


PROJECT_ROOT = Path(__file__).resolve().parent


def dataset_name(path: Path) -> str:
    return path.name


def run_cmd(cmd: list[str]) -> float:
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, cwd=PROJECT_ROOT)
    runtime = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return runtime


def run_predict(
    python: str,
    input_csv: Path,
    output_csv: Path,
    k: int,
    method: str,
    args: argparse.Namespace,
    model_path: Path | None = None,
    config_path: Path | None = None,
) -> float:
    cmd = [
        python,
        "regr_fail_bucketing.py",
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
        "--k",
        str(k),
    ]
    if method == "pairwise_mlp":
        calibration = load_calibration(args.calibration_json)
        cmd.extend(
            [
                "--cluster",
                "pairwise_mlp",
                "--pairwise-model",
                str(model_path),
                "--pairwise-device",
                "cpu",
                "--pairwise-primary-floor",
                str(calibration.get("primary_floor", args.pairwise_primary_floor)),
                "--pairwise-op-pair-floor",
                str(calibration.get("op_pair_floor", args.pairwise_op_pair_floor)),
                "--pairwise-mismatch-floor",
                str(calibration.get("mismatch_floor", args.pairwise_mismatch_floor)),
                "--pairwise-conflict-penalty",
                str(calibration.get("conflict_penalty", args.pairwise_conflict_penalty)),
                "--pairwise-mismatch-cosine-gate",
                str(calibration.get("mismatch_cosine_gate", args.pairwise_mismatch_cosine_gate)),
            ]
        )
        if config_path:
            cmd.extend(["--pairwise-config", str(config_path)])
    return run_cmd(cmd)


def load_calibration(path: Path | None) -> dict:
    if not path:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    calibration = {
        "primary_floor": data.get("primary_floor", data.get("pairwise_primary_floor")),
        "op_pair_floor": data.get("op_pair_floor", data.get("pairwise_op_pair_floor")),
        "mismatch_floor": data.get("mismatch_floor", data.get("pairwise_mismatch_floor")),
        "conflict_penalty": data.get("conflict_penalty", data.get("pairwise_conflict_penalty")),
        "mismatch_cosine_gate": data.get("mismatch_cosine_gate", data.get("pairwise_mismatch_cosine_gate")),
    }
    return {key: value for key, value in calibration.items() if value is not None}


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    header = [
        "seed",
        "combo",
        "dataset",
        "method",
        "BA",
        "TPR",
        "TNR",
        "num_cases",
        "k",
        "runtime_sec",
        "model_path",
        "pred_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["dataset"])].append(row)
    out = []
    for (method, dataset), items in sorted(groups.items()):
        out.append(
            {
                "method": method,
                "dataset": dataset,
                "mean_BA": statistics.mean(float(row["BA"]) for row in items),
                "mean_TPR": statistics.mean(float(row["TPR"]) for row in items),
                "mean_TNR": statistics.mean(float(row["TNR"]) for row in items),
                "num_runs": len(items),
            }
        )
    return out


def write_summary(path: Path, rows: Sequence[dict]) -> None:
    header = ["method", "dataset", "mean_BA", "mean_TPR", "mean_TNR", "num_runs"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[dict]) -> None:
    print("| method | dataset | mean_BA | mean_TPR | mean_TNR | runs |")
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['method']} | {row['dataset']} | {float(row['mean_BA']):.6f} "
            f"| {float(row['mean_TPR']):.6f} | {float(row['mean_TNR']):.6f} | {row['num_runs']} |"
        )


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    results = []
    datasets = [Path(path).resolve() for path in args.datasets]
    for seed in args.seeds:
        combo = args.combo
        model_path = args.output_dir / "models" / f"seed_{seed}_combo_{combo:03b}_pairwise_mlp.pt"
        config_path = args.output_dir / "models" / f"seed_{seed}_combo_{combo:03b}_pairwise_config.json"
        train_cmd = [
            args.python,
            "train_pairwise_mlp.py",
            "--datasets",
            *[str(path) for path in datasets],
            "--output",
            str(model_path),
            "--config-output",
            str(config_path),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--max-train-pairs",
            str(args.max_train_pairs),
            "--batch-size",
            str(args.batch_size),
            "--architecture",
            args.architecture,
            "--hidden-dims",
            *[str(dim) for dim in args.hidden_dims],
            "--dropout",
            str(args.dropout),
            "--negative-ratio",
            str(args.negative_ratio),
            "--hard-negative-ratio",
            str(args.hard_negative_ratio),
            "--hard-positive-ratio",
            str(args.hard_positive_ratio),
            "--pos-weight-scale",
            str(args.pos_weight_scale),
            "--pairwise-primary-floor",
            str(args.pairwise_primary_floor),
            "--pairwise-op-pair-floor",
            str(args.pairwise_op_pair_floor),
            "--pairwise-mismatch-floor",
            str(args.pairwise_mismatch_floor),
            "--pairwise-conflict-penalty",
            str(args.pairwise_conflict_penalty),
            "--pairwise-mismatch-cosine-gate",
            str(args.pairwise_mismatch_cosine_gate),
            "--seeds",
            str(seed),
            "--combo",
            str(combo),
        ]
        run_cmd(train_cmd)

        splits_by_dataset = {}
        for dataset in datasets:
            splits_by_dataset[dataset.name] = stratified_half_split(
                dataset,
                seed,
                args.output_dir / "splits" / f"seed_{seed}",
            )

        for idx, dataset in enumerate(datasets):
            train_part = part_for_bit((combo >> idx) & 1)
            val_part = opposite_part(train_part)
            val_info = splits_by_dataset[dataset.name][val_part]
            for method in ("baseline", "pairwise_mlp"):
                pred_path = args.output_dir / "preds" / f"seed_{seed}_combo_{combo:03b}_{dataset.name}_{method}.csv"
                runtime = run_predict(
                    args.python,
                    val_info["input"],
                    pred_path,
                    val_info["k"],
                    method,
                    args,
                    model_path=model_path,
                    config_path=config_path,
                )
                gold = read_gold(val_info["gold"])
                pred = read_pred(pred_path)
                ba, tpr, tnr = pairwise_scores(gold, pred)
                row = {
                    "seed": seed,
                    "combo": f"{combo:03b}",
                    "dataset": dataset.name,
                    "method": method,
                    "BA": ba,
                    "TPR": tpr,
                    "TNR": tnr,
                    "num_cases": val_info["num_cases"],
                    "k": val_info["k"],
                    "runtime_sec": runtime,
                    "model_path": str(model_path) if method == "pairwise_mlp" else "",
                    "pred_path": str(pred_path),
                }
                results.append(row)
                print(
                    f"done seed={seed} combo={combo:03b} dataset={dataset.name} "
                    f"method={method} BA={ba:.6f}",
                    file=sys.stderr,
                )
    summary = summarize(results)
    write_csv(args.output_dir / "results.csv", results)
    write_summary(args.output_dir / "summary.csv", summary)
    return results, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise MLP half-split comparison.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--combo", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pairwise_mlp_exp"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-train-pairs", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--architecture", choices=("plain", "layernorm", "residual"), default="residual")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 512, 256, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--hard-positive-ratio", type=float, default=0.5)
    parser.add_argument("--pos-weight-scale", type=float, default=1.2)
    parser.add_argument("--pairwise-primary-floor", type=float, default=0.70)
    parser.add_argument("--pairwise-op-pair-floor", type=float, default=0.65)
    parser.add_argument("--pairwise-mismatch-floor", type=float, default=0.55)
    parser.add_argument("--pairwise-conflict-penalty", type=float, default=0.05)
    parser.add_argument("--pairwise-mismatch-cosine-gate", type=float, default=0.20)
    parser.add_argument("--calibration-json", type=Path, help="best_calibration.json from run_pairwise_calibration_search.py")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _, summary = run(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
