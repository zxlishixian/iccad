#!/usr/bin/env python3
"""Strict dataset-level LODO training/evaluation for multi-view artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from evaluation_leakage_guard import assert_held_out, dataset_identity


ROOT = Path(__file__).resolve().parent
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
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def fold_training_datasets(datasets: Sequence[Path], holdout: Path) -> list[Path]:
    return [dataset for dataset in datasets if dataset.resolve() != holdout.resolve()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--view-configs", nargs="+", default=["dual", "quad_event_object_context"])
    parser.add_argument("--view-dim", type=int, default=64)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/multiview_lodo_cache"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def run_checked(command: Sequence[str]) -> None:
    print("[strict-lodo] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = [resolve(path) for path in args.datasets]
    if len({dataset.name for dataset in datasets}) != len(datasets):
        raise ValueError("dataset names must be unique for strict LODO")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    for holdout in datasets:
        train_datasets = fold_training_datasets(datasets, holdout)
        fold_dir = output_dir / "folds" / holdout.name
        preflight_manifest = {
            "training_datasets": [
                dataset_identity(dataset, include_case_logs=True)
                for dataset in train_datasets
            ]
        }
        assert_held_out(preflight_manifest, [holdout])
        model_dir = fold_dir / "models"
        eval_dir = fold_dir / "eval"
        manifest_path = model_dir / "manifest.json"
        if not (args.resume and manifest_path.is_file()):
            command = [
                args.python, str(ROOT / "train_multiview_submission.py"),
                "--train-datasets", *(str(path) for path in train_datasets),
                "--output-dir", str(model_dir),
                "--seeds", *(str(seed) for seed in args.seeds),
                "--view-configs", *args.view_configs,
                "--view-dim", str(args.view_dim),
                "--hard-positive-ratio", "1.0",
                "--canonicalize-case-indices",
                "--llm-cache-dir", str(args.llm_cache_dir),
            ]
            run_checked(command)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert_held_out(manifest, [holdout])
        result_path = eval_dir / "results.csv"
        if not (args.resume and result_path.is_file()):
            run_checked([
                args.python, str(ROOT / "run_leakage_safe_multiview_evaluation.py"),
                "--model-dir", str(model_dir),
                "--datasets", str(holdout),
                "--output-dir", str(eval_dir),
            ])
        row = read_rows(result_path)[0]
        row["holdout_dataset"] = holdout.name
        row["train_datasets"] = "|".join(dataset.name for dataset in train_datasets)
        row["num_seeds"] = len(args.seeds)
        all_rows.append(row)

    write_rows(output_dir / "results.csv", all_rows)
    bas = [float(row["BA"]) for row in all_rows]
    tprs = [float(row["TPR"]) for row in all_rows]
    tnrs = [float(row["TNR"]) for row in all_rows]
    official = [
        float(row["BA"]) for row in all_rows
        if row["dataset"] in {"benchmark_set_1", "benchmark_set_2"}
    ]
    summary = [{
        "protocol": f"{len(datasets)}-fold_dataset_lodo",
        "folds": len(datasets),
        "seeds_per_artifact": len(args.seeds),
        "macro_BA": statistics.mean(bas),
        "min_BA": min(bas),
        "max_BA": max(bas),
        "macro_TPR": statistics.mean(tprs),
        "macro_TNR": statistics.mean(tnrs),
        "official_mean_BA": statistics.mean(official),
    }]
    write_rows(output_dir / "summary.csv", summary)
    print(
        f"[strict-lodo] macro_BA={summary[0]['macro_BA']:.6f} "
        f"official_mean_BA={summary[0]['official_mean_BA']:.6f} min_BA={summary[0]['min_BA']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
