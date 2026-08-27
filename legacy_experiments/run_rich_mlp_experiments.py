#!/usr/bin/env python3
"""Run experimental rich pairwise MLP half-split experiments.

This is a research runner. It trains on half-split gold.csv labels and evaluates
on the held-out half for each dataset. Official prediction remains
regr_fail_bucketing.py and does not read gold/meta.
"""

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

from run_half_split_experiments import DEFAULT_DATASETS

PROJECT_ROOT = Path(__file__).resolve().parent

CONFIGS: dict[str, dict[str, str]] = {
    "summary21_shallow_bce": {"feature_mode": "summary21", "mlp_arch": "shallow", "loss": "bce"},
    "summary21_deep_focal": {"feature_mode": "summary21", "mlp_arch": "deep", "loss": "focal"},
    "rich_deep_bce": {"feature_mode": "rich", "mlp_arch": "deep", "loss": "bce"},
    "rich_deep_focal": {"feature_mode": "rich", "mlp_arch": "deep", "loss": "focal"},
    "rich_residual_focal": {"feature_mode": "rich", "mlp_arch": "residual", "loss": "focal"},
    "rich_no_llm_deep_focal": {"feature_mode": "rich_no_llm", "mlp_arch": "deep", "loss": "focal"},
    "rich_no_det_deep_focal": {"feature_mode": "rich_no_det", "mlp_arch": "deep", "loss": "focal"},
    "rich_no_llm_residual_focal": {"feature_mode": "rich_no_llm", "mlp_arch": "residual", "loss": "focal"},
    "rich_no_det_residual_focal": {"feature_mode": "rich_no_det", "mlp_arch": "residual", "loss": "focal"},
    "llm_dual_residual_focal": {"feature_mode": "llm_dual", "mlp_arch": "residual", "loss": "focal"},
    "llm_dual_struct_residual_focal": {"feature_mode": "llm_dual_struct", "mlp_arch": "residual", "loss": "focal"},
    "llm_dual_struct_det_summary_residual_focal": {"feature_mode": "llm_dual_struct_det_summary", "mlp_arch": "residual", "loss": "focal"},
}

DEFAULT_SEARCH = [
    "summary21_shallow_bce",
    "summary21_deep_focal",
    "rich_deep_bce",
    "rich_deep_focal",
    "rich_residual_focal",
]


def _read_config_rows(config_path: Path, run_name: str) -> list[dict]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    rows = []
    for detail in data.get("val_details", []):
        rows.append({
            "run_name": run_name,
            "seed": data.get("seed"),
            "combo": f"{int(data.get('combo', 0)):03b}",
            "method": f"rich_mlp_{run_name}",
            "feature_mode": data.get("feature_mode", ""),
            "arch": data.get("mlp_arch", ""),
            "loss": data.get("loss", ""),
            "llm_reduce_dim": data.get("llm_reduce_dim", ""),
            "dataset": detail.get("dataset"),
            "BA": detail.get("BA"),
            "TPR": detail.get("TPR"),
            "TNR": detail.get("TNR"),
            "num_cases": detail.get("num_cases"),
            "k": detail.get("k"),
            "num_train_pairs": data.get("num_train_pairs"),
            "model_path": data.get("model_path"),
            "config_path": str(config_path),
            "train_time_sec": data.get("train_time_sec"),
            "total_time_sec": data.get("total_time_sec"),
        })
    return rows


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["run_name"]), str(row["dataset"]))].append(row)
    out = []
    for (run_name, dataset), items in sorted(groups.items()):
        bas = [float(r["BA"]) for r in items]
        tprs = [float(r["TPR"]) for r in items]
        tnrs = [float(r["TNR"]) for r in items]
        first = items[0]
        out.append({
            "run_name": run_name,
            "method": first["method"],
            "feature_mode": first["feature_mode"],
            "arch": first["arch"],
            "loss": first["loss"],
            "dataset": dataset,
            "mean_BA": statistics.mean(bas),
            "std_BA": statistics.stdev(bas) if len(bas) > 1 else 0.0,
            "mean_TPR": statistics.mean(tprs),
            "mean_TNR": statistics.mean(tnrs),
            "num_runs": len(items),
        })
    return out


def print_wide_table(summary_rows: Sequence[dict]) -> None:
    by_run: dict[str, dict[str, float]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for row in summary_rows:
        by_run[row["run_name"]][row["dataset"]] = float(row["mean_BA"])
        meta[row["run_name"]] = row
    print("\n| method | feature_mode | arch | loss | first_BA | stage2_BA | stage3_BA | mean_BA |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    for run_name in sorted(by_run):
        scores = by_run[run_name]
        first = scores.get("first_batch_dataset", 0.0)
        stage2 = scores.get("stage2_dataset_working", 0.0)
        stage3 = scores.get("stage3_dataset_32bugs_640cases", 0.0)
        present = [v for v in (first, stage2, stage3) if v]
        mean_ba = statistics.mean(present) if present else 0.0
        m = meta[run_name]
        print(
            f"| rich_mlp_{run_name} | {m['feature_mode']} | {m['arch']} | {m['loss']} "
            f"| {first:.6f} | {stage2:.6f} | {stage3:.6f} | {mean_ba:.6f} |"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rich pairwise MLP experiments.")
    p.add_argument("--python", default="/home/lishixian/miniforge3/envs/collab-overcooked/bin/python")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/rich_pairwise_mlp_exp"))
    p.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=DEFAULT_SEARCH)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--use-llm", action="store_true", default=True)
    p.add_argument("--no-llm", action="store_true", default=False)
    p.add_argument("--llm-doc-style", choices=("features", "summary"), default="features")
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--llm-reduce-dim", type=int, default=128)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--max-train-pairs", type=int, default=300000)
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--hard-negative-ratio", type=float, default=0.5)
    p.add_argument("--hard-positive-ratio", type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", default="auto")
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--no-layernorm", action="store_true", default=False)
    p.add_argument("--batchnorm", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_llm:
        args.use_llm = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [(d if d.is_absolute() else (PROJECT_ROOT / d).resolve()) for d in args.datasets]

    rows: list[dict] = []
    for seed in args.seeds:
        for cfg_name in args.configs:
            cfg = CONFIGS[cfg_name]
            model_dir = args.output_dir / "models" / cfg_name
            model_tag = cfg_name
            cmd = [
                args.python,
                str(PROJECT_ROOT / "train_pairwise_llm.py"),
                "--datasets", *[str(d) for d in datasets],
                "--output-dir", str(model_dir),
                "--model-type", "mlp",
                "--model-tag", model_tag,
                "--feature-mode", cfg["feature_mode"],
                "--mlp-arch", cfg["mlp_arch"],
                "--loss", cfg["loss"],
                "--llm-reduce-dim", str(args.llm_reduce_dim),
                "--svd-dim", str(args.svd_dim),
                "--llm-doc-style", args.llm_doc_style,
                "--llm-cache-dir", str(args.llm_cache_dir),
                "--seed", str(seed),
                "--combo", str(args.combo),
                "--random-state", str(seed),
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--max-train-pairs", str(args.max_train_pairs),
                "--negative-ratio", str(args.negative_ratio),
                "--hard-negative-ratio", str(args.hard_negative_ratio),
                "--hard-positive-ratio", str(args.hard_positive_ratio),
                "--early-stop-patience", str(args.early_stop_patience),
                "--dropout", str(args.dropout),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--focal-gamma", str(args.focal_gamma),
                "--focal-alpha", str(args.focal_alpha),
                "--predict-batch-size", str(args.predict_batch_size),
            ]
            if not args.use_llm:
                cmd.append("--no-llm")
            if args.no_layernorm:
                cmd.append("--no-layernorm")
            if args.batchnorm:
                cmd.append("--batchnorm")

            print(f"\n=== {cfg_name} seed={seed} ===", file=sys.stderr)
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, text=True, capture_output=True, check=False, cwd=PROJECT_ROOT)
            elapsed = time.perf_counter() - t0
            if proc.returncode != 0:
                print(proc.stdout, file=sys.stderr)
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"training failed for {cfg_name} seed={seed}")
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            config_path = model_dir / f"config_seed{seed}_combo{args.combo:03b}_{model_tag}.json"
            cfg_rows = _read_config_rows(config_path, cfg_name)
            for row in cfg_rows:
                row["runner_elapsed_sec"] = elapsed
            rows.extend(cfg_rows)

    result_header = [
        "run_name", "seed", "combo", "method", "feature_mode", "arch", "loss",
        "llm_reduce_dim", "dataset", "BA", "TPR", "TNR", "num_cases", "k",
        "num_train_pairs", "model_path", "config_path", "train_time_sec",
        "total_time_sec", "runner_elapsed_sec",
    ]
    _write_csv(args.output_dir / "results.csv", rows, result_header)
    summary_rows = summarize(rows)
    summary_header = [
        "run_name", "method", "feature_mode", "arch", "loss", "dataset",
        "mean_BA", "std_BA", "mean_TPR", "mean_TNR", "num_runs",
    ]
    _write_csv(args.output_dir / "summary.csv", summary_rows, summary_header)
    print_wide_table(summary_rows)
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    print(f"Summary:  {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
