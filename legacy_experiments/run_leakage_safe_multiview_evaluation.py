#!/usr/bin/env python3
"""Evaluate multi-view artifacts while rejecting train/test overlap by default."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

import official_style_features as osf
from evaluation_leakage_guard import assert_held_out, find_training_overlap
from run_alpha_beta_v2_scored_evaluation import validate_output
from run_experiments import pairwise_scores, read_gold


ROOT = Path(__file__).resolve().parent


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=os.sys.executable)
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    parser.add_argument(
        "--allow-in-sample", action="store_true",
        help="Diagnostic only: score overlapping train data and label it in-sample.",
    )
    return parser.parse_args(argv)


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = resolve(args.model_dir)
    datasets = [resolve(path) for path in args.datasets]
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    overlaps = find_training_overlap(manifest, datasets)
    if not args.allow_in_sample:
        assert_held_out(manifest, datasets)
    overlap_names = {item["evaluation_dataset"] for item in overlaps}

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        cases = [str(case) for case in osf.read_cases(dataset / "input.csv")]
        gold = read_gold(osf.gold_path(dataset))
        k = len(set(gold))
        pred_path = output_dir / "preds" / f"{dataset.name}.csv"
        log_path = output_dir / "logs" / f"{dataset.name}.log"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["BETA_MULTIVIEW_MODEL_DIR"] = str(model_dir)
        env["BETA_LLM_CACHE_DIR"] = str(output_dir / "cache" / dataset.name)
        command = [
            args.python, str(ROOT / "beta_multiview_inference.py"),
            "--input", str(dataset / "input.csv"),
            "--output", str(pred_path), "--k", str(k),
        ]
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=handle,
                stderr=subprocess.STDOUT, timeout=args.timeout_sec, check=False,
            )
        runtime = time.perf_counter() - started
        labels = validate_output(pred_path, cases)
        if completed.returncode != 0 or labels is None:
            raise RuntimeError(
                f"inference failed for {dataset.name}; returncode={completed.returncode}; log={log_path}"
            )
        ba, tpr, tnr = pairwise_scores(gold, labels)
        in_sample = dataset.name in overlap_names
        row = {
            "dataset": dataset.name,
            "evaluation_protocol": "in_sample_diagnostic" if in_sample else "held_out",
            "training_overlap": in_sample,
            "cases": len(cases), "k": k, "num_pred_clusters": len(set(labels)),
            "BA": ba, "TPR": tpr, "TNR": tnr, "runtime_sec": runtime,
            "model_dir": str(model_dir), "pred_path": str(pred_path), "log_path": str(log_path),
        }
        rows.append(row)
        print(
            f"[heldout-eval] dataset={dataset.name} protocol={row['evaluation_protocol']} "
            f"BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f} runtime={runtime:.2f}s",
            flush=True,
        )
    write_rows(output_dir / "results.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
