#!/usr/bin/env python3
"""Run input-signal pairwise MLP experiments.

Experimental only: trains on half-split gold.csv labels and evaluates held-out
halves. It never changes the official regr_fail_bucketing.py default path.
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

import numpy as np

import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold
from run_half_split_experiments import DEFAULT_DATASETS, opposite_part, part_for_bit, stratified_half_split

PROJECT_ROOT = Path(__file__).resolve().parent

CONFIGS: dict[str, dict] = {
    "rich_no_det_residual_focal": {
        "feature_mode": "rich_no_det", "mlp_arch": "residual", "loss": "focal", "llm_reduce_dim": None,
    },
    "llm_dual_residual_focal": {
        "feature_mode": "llm_dual", "mlp_arch": "residual", "loss": "focal", "llm_reduce_dim": None,
    },
    "llm_dual_struct_residual_focal": {
        "feature_mode": "llm_dual_struct", "mlp_arch": "residual", "loss": "focal", "llm_reduce_dim": None,
    },
    "llm_dual_struct_det_summary_residual_focal": {
        "feature_mode": "llm_dual_struct_det_summary", "mlp_arch": "residual", "loss": "focal", "llm_reduce_dim": None,
    },
    "llm_dual_struct_det_summary_dim64": {
        "feature_mode": "llm_dual_struct_det_summary", "mlp_arch": "residual", "loss": "focal", "llm_reduce_dim": 64,
    },
    "llm_dual_struct_det_summary_dim256": {
        "feature_mode": "llm_dual_struct_det_summary", "mlp_arch": "residual", "loss": "focal", "llm_reduce_dim": 256,
    },
}

DEFAULT_CONFIGS = [
    "rich_no_det_residual_focal",
    "llm_dual_residual_focal",
    "llm_dual_struct_residual_focal",
    "llm_dual_struct_det_summary_residual_focal",
    "llm_dual_struct_det_summary_dim64",
    "llm_dual_struct_det_summary_dim256",
]

ENSEMBLE_WEIGHTS = (0.20, 0.40, 0.40)
ENSEMBLE_MODEL_TYPES = ("logistic", "gbdt", "mlp")


def _model_ext(model_type: str) -> str:
    return "pt" if model_type == "mlp" else "pkl"


def _read_config_rows(config_path: Path, run_name: str, blend_alpha: str = "") -> list[dict]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    rows = []
    for detail in data.get("val_details", []):
        rows.append({
            "run_name": run_name,
            "seed": data.get("seed"),
            "combo": f"{int(data.get('combo', 0)):03b}",
            "method": f"input_signal_{run_name}",
            "feature_mode": data.get("feature_mode", ""),
            "arch": data.get("mlp_arch", ""),
            "loss": data.get("loss", ""),
            "llm_reduce_dim": data.get("llm_reduce_dim", ""),
            "blend_alpha": blend_alpha,
            "dataset": detail.get("dataset"),
            "BA": detail.get("BA"),
            "TPR": detail.get("TPR"),
            "TNR": detail.get("TNR"),
            "num_cases": detail.get("num_cases"),
            "k": detail.get("k"),
            "model_path": data.get("model_path"),
            "config_path": str(config_path),
        })
    return rows


def build_val_parts(datasets: Sequence[Path], seed: int, combo: int, split_root: Path) -> list[dict]:
    parts = []
    for idx, ds in enumerate(datasets):
        splits = stratified_half_split(ds, seed, split_root)
        train_part = part_for_bit((combo >> idx) & 1)
        val_part = opposite_part(train_part)
        info = dict(splits[val_part])
        info["dataset"] = ds.name
        parts.append(info)
    return parts


def find_ensemble_models(model_dir: Path, seed: int, combo: int) -> list[Path]:
    paths = []
    for model_type in ENSEMBLE_MODEL_TYPES:
        path = model_dir / f"model_seed{seed}_combo{combo:03b}_{model_type}.{_model_ext(model_type)}"
        if not path.exists():
            raise FileNotFoundError(f"missing ensemble base model: {path}")
        paths.append(path)
    return paths


def evaluate_blend(
    run_name: str,
    model_path: Path,
    output_dir: Path,
    datasets: Sequence[Path],
    seed: int,
    combo: int,
    alpha: float,
    ensemble_model_dir: Path,
    split_root: Path,
    llm_cache_dir: Path,
    svd_dim: int,
    predict_batch_size: int,
) -> list[dict]:
    rich_model = plf.load_model_pkg(model_path)
    feature_mode = str(rich_model.get("feature_mode", ""))
    rich_llm_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=llm_cache_dir,
        svd_dim=svd_dim,
        llm_dual=feature_mode in plf.DUAL_FEATURE_MODES,
    )
    ensemble_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=llm_cache_dir,
        svd_dim=svd_dim,
    )
    ensemble_pkgs = [plf.load_model_pkg(path) for path in find_ensemble_models(ensemble_model_dir, seed, combo)]
    val_parts = build_val_parts(datasets, seed, combo, split_root)
    rows = []
    for part in val_parts:
        t0 = time.perf_counter()
        rich_features, _ = plf.build_llm_case_features(part["input"], svd_dim=svd_dim, llm_args=rich_llm_args)
        p_rich = plf.predict_probability_matrix_sklearn(rich_model, rich_features, batch_size=predict_batch_size)
        ensemble_features, _ = plf.build_llm_case_features(part["input"], svd_dim=svd_dim, llm_args=ensemble_args)
        p_ensemble = plf.predict_probability_matrix_ensemble(
            ensemble_pkgs, list(ENSEMBLE_WEIGHTS), ensemble_features,
            ensemble_mode="prob_average", batch_size=predict_batch_size,
        )
        prob = float(alpha) * p_rich + (1.0 - float(alpha)) * p_ensemble
        labels = plf.cluster_from_probability(prob.astype(np.float32), part["k"])
        gold = read_gold(part["gold"])
        pred = [f"bucket_{label:03d}" for label in labels]
        ba, tpr, tnr = pairwise_scores(gold, pred)
        pred_path = output_dir / "preds" / f"blend_seed{seed}_combo{combo:03b}_{run_name}_a{alpha:.2f}_{part['dataset']}.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["bucket"])
            for item in pred:
                writer.writerow([item])
        rows.append({
            "run_name": f"{run_name}_blend_a{alpha:.2f}",
            "seed": seed,
            "combo": f"{combo:03b}",
            "method": f"input_signal_{run_name}_blend",
            "feature_mode": feature_mode,
            "arch": rich_model.get("mlp_arch", ""),
            "loss": rich_model.get("loss", ""),
            "llm_reduce_dim": rich_model.get("llm_reduce_dim", ""),
            "blend_alpha": f"{alpha:.2f}",
            "dataset": part["dataset"],
            "BA": ba,
            "TPR": tpr,
            "TNR": tnr,
            "num_cases": part["num_cases"],
            "k": part["k"],
            "model_path": str(model_path),
            "config_path": "",
            "runtime_sec": time.perf_counter() - t0,
            "pred_path": str(pred_path),
        })
        print(
            f"[blend] run={run_name} alpha={alpha:.2f} dataset={part['dataset']} "
            f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
            file=sys.stderr,
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["run_name"]), str(row["dataset"]))].append(row)
    out = []
    for (run_name, dataset), items in sorted(groups.items()):
        first = items[0]
        bas = [float(r["BA"]) for r in items]
        tprs = [float(r["TPR"]) for r in items]
        tnrs = [float(r["TNR"]) for r in items]
        out.append({
            "run_name": run_name,
            "method": first["method"],
            "feature_mode": first["feature_mode"],
            "reduce_dim": first.get("llm_reduce_dim", ""),
            "blend_alpha": first.get("blend_alpha", ""),
            "dataset": dataset,
            "mean_BA": statistics.mean(bas),
            "std_BA": statistics.stdev(bas) if len(bas) > 1 else 0.0,
            "mean_TPR": statistics.mean(tprs),
            "mean_TNR": statistics.mean(tnrs),
            "num_runs": len(items),
        })
    return out


def print_wide(summary_rows: Sequence[dict]) -> None:
    by_run: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in summary_rows:
        by_run[row["run_name"]][row["dataset"]] = row
    print("\n| method | feature_mode | reduce_dim | blend_alpha | first_BA | stage2_BA | stage3_BA | mean_BA | first_TPR/TNR | stage2_TPR/TNR | stage3_TPR/TNR |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for run_name in sorted(by_run):
        rows = by_run[run_name]
        fb = rows.get("first_batch_dataset", {})
        s2 = rows.get("stage2_dataset_working", {})
        s3 = rows.get("stage3_dataset_32bugs_640cases", {})
        vals = [float(r.get("mean_BA", 0.0)) for r in (fb, s2, s3) if r]
        mean_ba = statistics.mean(vals) if vals else 0.0
        first = fb or s2 or s3
        def fmt_pair(r: dict) -> str:
            return f"{float(r.get('mean_TPR', 0.0)):.4f}/{float(r.get('mean_TNR', 0.0)):.4f}" if r else ""
        print(
            f"| {run_name} | {first.get('feature_mode', '')} | {first.get('reduce_dim', '')} | {first.get('blend_alpha', '')} "
            f"| {float(fb.get('mean_BA', 0.0)):.6f} | {float(s2.get('mean_BA', 0.0)):.6f} "
            f"| {float(s3.get('mean_BA', 0.0)):.6f} | {mean_ba:.6f} "
            f"| {fmt_pair(fb)} | {fmt_pair(s2)} | {fmt_pair(s3)} |"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pairwise input-signal experiments.")
    p.add_argument("--python", default="/home/lishixian/miniforge3/envs/collab-overcooked/bin/python")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/input_signal_exp"))
    p.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=DEFAULT_CONFIGS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--combo", type=int, default=0)
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
    p.add_argument("--blend-with-ensemble", action="store_true")
    p.add_argument("--blend-alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--blend-configs", nargs="+", default=[])
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--ensemble-split-root", type=Path, default=Path("/tmp/pairwise_llm_exp_full/splits"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [(d if d.is_absolute() else (PROJECT_ROOT / d).resolve()) for d in args.datasets]
    all_rows: list[dict] = []
    trained_model_paths: dict[tuple[int, str], Path] = {}

    for seed in args.seeds:
        for config_name in args.configs:
            cfg = CONFIGS[config_name]
            reduce_dim = int(cfg.get("llm_reduce_dim") or args.llm_reduce_dim)
            model_dir = args.output_dir / "models" / config_name
            cmd = [
                args.python,
                str(PROJECT_ROOT / "train_pairwise_llm.py"),
                "--datasets", *[str(d) for d in datasets],
                "--output-dir", str(model_dir),
                "--model-type", "mlp",
                "--model-tag", config_name,
                "--feature-mode", str(cfg["feature_mode"]),
                "--mlp-arch", str(cfg["mlp_arch"]),
                "--loss", str(cfg["loss"]),
                "--llm-reduce-dim", str(reduce_dim),
                "--svd-dim", str(args.svd_dim),
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
                "--llm-cache-dir", str(args.llm_cache_dir),
            ]
            print(f"\n=== train {config_name} seed={seed} reduce_dim={reduce_dim} ===", file=sys.stderr)
            proc = subprocess.run(cmd, text=True, capture_output=True, check=False, cwd=PROJECT_ROOT)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            if proc.returncode != 0:
                print(proc.stdout, file=sys.stderr)
                raise RuntimeError(f"training failed for {config_name} seed={seed}")
            config_path = model_dir / f"config_seed{seed}_combo{args.combo:03b}_{config_name}.json"
            rows = _read_config_rows(config_path, config_name)
            all_rows.extend(rows)
            trained_model_paths[(seed, config_name)] = Path(rows[0]["model_path"])

    if args.blend_with_ensemble:
        blend_names = args.blend_configs or [name for name in args.configs if name != "rich_no_det_residual_focal"]
        for seed in args.seeds:
            split_root = args.output_dir / "models" / args.configs[0] / "splits" / f"seed_{seed}"
            if not split_root.exists():
                split_root = args.ensemble_split_root / f"seed_{seed}"
            for config_name in blend_names:
                model_path = trained_model_paths.get((seed, config_name))
                if model_path is None:
                    continue
                for alpha in args.blend_alphas:
                    rows = evaluate_blend(
                        config_name,
                        model_path,
                        args.output_dir,
                        datasets,
                        seed,
                        args.combo,
                        alpha,
                        args.ensemble_model_dir,
                        split_root,
                        args.llm_cache_dir,
                        args.svd_dim,
                        args.predict_batch_size,
                    )
                    all_rows.extend(rows)

    result_header = [
        "run_name", "seed", "combo", "method", "feature_mode", "arch", "loss",
        "llm_reduce_dim", "blend_alpha", "dataset", "BA", "TPR", "TNR",
        "num_cases", "k", "model_path", "config_path", "runtime_sec", "pred_path",
    ]
    _write_csv(args.output_dir / "results.csv", all_rows, result_header)
    summary_rows = summarize(all_rows)
    summary_header = [
        "run_name", "method", "feature_mode", "reduce_dim", "blend_alpha", "dataset",
        "mean_BA", "std_BA", "mean_TPR", "mean_TNR", "num_runs",
    ]
    _write_csv(args.output_dir / "summary.csv", summary_rows, summary_header)
    print_wide(summary_rows)
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    print(f"Summary:  {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
