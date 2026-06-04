#!/usr/bin/env python3
"""Evaluate the experimental official-style gated adapter.

This script is experimental. It may read gold/golden CSV files for scoring and
for training the adapter on public official benchmarks. It does not modify the
formal ``regr_fail_bucketing.py`` default path.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf
from run_experiments import pairwise_scores, read_gold
from run_official_style_training_experiments import (
    PROJECT_ROOT,
    build_features_for_art,
    dataset_artifacts,
    prob_from_pair_scores,
    predict_scores,
    score_probability,
    train_model,
    write_csv,
)


def normalized_fields(input_csv: Path) -> set[str]:
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        fields = next(reader, [])
    return {"".join(ch for ch in str(f).lower() if ch.isalnum()) for f in fields}


def is_official_style_input(input_csv: Path) -> bool:
    fields = normalized_fields(input_csv)
    has_named_cols = {"case", "regrlog", "simlog"}.issubset(fields)
    if not has_named_cols:
        return False
    rows, _fields = __import__("regr_fail_bucketing").read_csv_rows(input_csv)
    sample = " ".join(" ".join(str(v) for v in row.values()) for row in rows[:5]).lower()
    return ".gz" in sample or "case_" in sample


def choose_alpha(input_csv: Path, gate: str, fake_alpha: float, official_alpha: float, fixed_alpha: float) -> tuple[float, str]:
    if gate == "fixed":
        return fixed_alpha, f"fixed_alpha={fixed_alpha}"
    official_style = is_official_style_input(input_csv)
    alpha = official_alpha if official_style else fake_alpha
    return alpha, f"auto_gate official_style={official_style}; alpha={alpha}"


def adapter_train_arts(target_art: dict, official_arts: Sequence[dict], mode: str) -> list[dict]:
    if mode == "all_official":
        return list(official_arts)
    if mode == "leave_target_official_out":
        out = [art for art in official_arts if art["name"] != target_art["name"]]
        return out or list(official_arts)
    raise ValueError(f"unknown train_mode: {mode}")


def evaluate_target(
    target_art: dict,
    official_arts: Sequence[dict],
    args: argparse.Namespace,
) -> list[dict]:
    rows: list[dict] = []
    rows.append(score_probability(
        target_art,
        target_art["prob_base"],
        "B0_no_trace_best",
        args.output_dir,
        notes=target_art["base_note"],
        runtime=target_art["base_runtime"],
    ))

    train_arts = adapter_train_arts(target_art, official_arts, args.train_mode)
    X_blocks = [build_features_for_art(art, args.variant, 0) for art in train_arts]
    X_train = np.vstack([x for x, _y in X_blocks])
    y_train = np.concatenate([y for _x, y in X_blocks]).astype(int)
    X_test, _ = build_features_for_art(target_art, args.variant, 0)
    model = train_model(X_train, y_train, args.model_type, args.seed)
    pair_scores = predict_scores(model, X_test)
    p_adapter = prob_from_pair_scores(len(target_art["records"]), target_art["pairs"], pair_scores)
    np.fill_diagonal(p_adapter, 1.0)

    alpha, gate_note = choose_alpha(
        target_art["input_csv"],
        args.gate,
        args.fake_alpha,
        args.official_alpha,
        args.fixed_alpha,
    )
    p_final = float(alpha) * p_adapter + (1.0 - float(alpha)) * target_art["prob_base"]
    np.fill_diagonal(p_final, 1.0)
    method = f"official_gated_adapter_{args.model_type}_{args.gate}"
    row = score_probability(
        target_art,
        p_final,
        method,
        args.output_dir,
        notes=(
            f"train={'+'.join(art['name'] for art in train_arts)}; "
            f"variant={args.variant}; model={args.model_type}; {gate_note}; "
            f"train_pairs={len(y_train)} pos={int(y_train.sum())} neg={int((1-y_train).sum())}"
        ),
        runtime=0.0,
    )
    row.update({"train_dataset": "+".join(art["name"] for art in train_arts)})
    rows.append(row)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate official-style gated adapter.")
    p.add_argument("--official-train-datasets", nargs="+", type=Path, default=[
        Path("test_case/problem/benchmark_set_1"),
        Path("test_case/problem/benchmark_set_2"),
    ])
    p.add_argument("--target-datasets", nargs="+", type=Path, default=[
        Path("fake_dataset/first_batch_dataset"),
        Path("fake_dataset/stage2_dataset_working"),
        Path("fake_dataset/stage3_dataset_32bugs_640cases"),
        Path("test_case/problem/benchmark_set_1"),
        Path("test_case/problem/benchmark_set_2"),
    ])
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/official_gated_adapter_eval"))
    p.add_argument("--reuse-base-probs-dir", type=Path, default=None)
    p.add_argument("--variant", default="tags")
    p.add_argument("--model-type", choices=["logistic", "gbdt"], default="logistic")
    p.add_argument("--gate", choices=["auto", "fixed"], default="auto")
    p.add_argument("--fake-alpha", type=float, default=0.25)
    p.add_argument("--official-alpha", type=float, default=0.50)
    p.add_argument("--fixed-alpha", type=float, default=0.25)
    p.add_argument("--train-mode", choices=["leave_target_official_out", "all_official"], default="leave_target_official_out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window-sizes", nargs="+", type=int, default=[64])
    p.add_argument("--variants", nargs="+", default=["tags"])
    p.add_argument("--rich-model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temp", type=float, default=1.15)
    p.add_argument("--ensemble-temp", type=float, default=1.00)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    return p.parse_args(argv)


def resolve(path: Path) -> Path:
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    official_arts = [dataset_artifacts(resolve(ds), args) for ds in args.official_train_datasets]
    target_paths = [resolve(ds) for ds in args.target_datasets]
    target_by_path = {str(art["dataset"]): art for art in official_arts}
    rows: list[dict] = []
    for path in target_paths:
        target_art = target_by_path.get(str(path)) or dataset_artifacts(path, args)
        rows.extend(evaluate_target(target_art, official_arts, args))

    fields = [
        "train_dataset", "test_dataset", "method", "model_type", "window_size",
        "k", "cases", "num_pred_clusters", "BA", "TPR", "TNR", "runtime_sec",
        "pred_path", "prob_path", "notes",
    ]
    write_csv(args.output_dir / "results.csv", rows, fields)
    write_csv(args.output_dir / "summary.csv", rows, fields)

    print("\n| test | method | BA | TPR | TNR | clusters | notes |")
    print("|---|---|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['test_dataset']} | {row['method']} | {float(row['BA']):.6f} | "
            f"{float(row['TPR']):.6f} | {float(row['TNR']):.6f} | {row['num_pred_clusters']} | "
            f"{str(row.get('notes', ''))[:80]} |"
        )
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
