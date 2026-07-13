#!/usr/bin/env python3
"""Cold-cache score/runtime evaluation for frozen Alpha and Beta v2 packages.

This runner evaluates only labeled fake datasets and the released official
benchmarks. It calls each package's required top-level CLI directly.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Sequence

import official_style_features as osf
from run_experiments import pairwise_scores, read_gold


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


def validate_output(path: Path, cases: Sequence[str]) -> list[str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Case", "bucket"]:
            return None
        rows = list(reader)
    if [str(row["Case"]) for row in rows] != [str(case) for case in cases]:
        return None
    labels = [str(row["bucket"]).strip() for row in rows]
    return labels if all(labels) else None


def terminate(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def isolate_alpha_cache(output_dir: Path) -> tuple[Path, Path | None]:
    live = Path("/tmp/regr_fail_llm_cache")
    backup = output_dir / "alpha_original_cache_backup"
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite {backup}")
    if live.exists():
        live.rename(backup)
        return live, backup
    return live, None


def restore_alpha_cache(live: Path, backup: Path | None, output_dir: Path) -> None:
    if live.exists():
        generated = output_dir / "alpha_generated_cache"
        if generated.exists():
            shutil.rmtree(generated)
        live.rename(generated)
    if backup is not None:
        backup.rename(live)


def run_one(
    method: str,
    executable: Path,
    dataset: Path,
    output_dir: Path,
    no_timeout: bool,
) -> dict[str, object]:
    cases = [str(case) for case in osf.read_cases(dataset / "input.csv")]
    gold = read_gold(osf.gold_path(dataset))
    k = len(set(gold))
    pred_path = output_dir / "preds" / f"{method}_{dataset.name}.csv"
    log_path = output_dir / "logs" / f"{method}_{dataset.name}.log"
    time_path = output_dir / "timing" / f"{method}_{dataset.name}.txt"
    for path in (pred_path, log_path, time_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if method == "beta_v2":
        env["BETA_LLM_CACHE_DIR"] = str(output_dir / "caches" / f"beta_v2_{dataset.name}")
    command = [
        "/usr/bin/time", "-f", "%e,%M,%x", "-o", str(time_path),
        str(executable), "--input", str(dataset / "input.csv"),
        "--output", str(pred_path), "--k", str(k),
    ]
    started = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        if no_timeout:
            returncode = process.wait()
        else:
            limit = 30 if len(cases) <= 30 and k <= 4 else 100
            try:
                returncode = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate(process)
                returncode = process.returncode
    observed_wall = time.monotonic() - started
    wall_sec, max_rss_kb = observed_wall, 0
    if time_path.exists():
        parts = time_path.read_text(encoding="utf-8").strip().split(",")
        if len(parts) == 3:
            wall_sec, max_rss_kb = float(parts[0]), int(parts[1])
    labels = validate_output(pred_path, cases)
    ba = tpr = tnr = float("nan")
    if labels is not None:
        ba, tpr, tnr = pairwise_scores(gold, labels)
    return {
        "method": method,
        "dataset": dataset.name,
        "cases": len(cases),
        "k": k,
        "wall_sec": wall_sec,
        "max_rss_kb": max_rss_kb,
        "returncode": returncode,
        "timed_out": timed_out,
        "valid_output": labels is not None,
        "num_pred_clusters": len(set(labels)) if labels is not None else 0,
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "pred_path": str(pred_path),
        "log_path": str(log_path),
    }


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scored Alpha vs Beta v2 evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", choices=["alpha", "beta_v2"], default=["alpha", "beta_v2"])
    parser.add_argument("--no-timeout", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [path.resolve() if path.is_absolute() else (ROOT / path).resolve() for path in args.datasets]
    executables = {
        "alpha": ROOT / "alpha_test_submission" / "regr_fail_bucketing",
        "beta_v2": ROOT / "beta_test_submission_v2" / "regr_fail_bucketing",
    }
    rows: list[dict] = []
    existing_path = output_dir / "results.csv"
    if existing_path.exists():
        with existing_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    live = backup = None
    if "alpha" in args.methods:
        live, backup = isolate_alpha_cache(output_dir)
    try:
        for method in args.methods:
            for dataset in datasets:
                if method == "alpha" and live is not None:
                    shutil.rmtree(live, ignore_errors=True)
                row = run_one(method, executables[method], dataset, output_dir, args.no_timeout)
                rows = [
                    old for old in rows
                    if not (old.get("method") == method and old.get("dataset") == dataset.name)
                ]
                rows.append(row)
                print(
                    f"[scored-eval] {method} {dataset.name} wall={row['wall_sec']:.2f}s "
                    f"BA={row['BA']:.6f} valid={row['valid_output']}",
                    flush=True,
                )
    finally:
        if live is not None:
            restore_alpha_cache(live, backup, output_dir)
    write_rows(output_dir / "results.csv", rows)
    return 0 if all(bool(row["valid_output"]) for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
