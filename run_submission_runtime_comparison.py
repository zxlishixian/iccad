#!/usr/bin/env python3
"""Cold-cache runtime comparison for frozen Alpha and Beta submissions."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

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


def final_runtime_limit(cases: int, k: int) -> int:
    return 30 if cases <= 30 and k <= 4 else 100


def validate_output(path: Path, expected_cases: Sequence[str]) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Case", "bucket"]:
            return False, []
        rows = list(reader)
    if [str(row["Case"]) for row in rows] != [str(case) for case in expected_cases]:
        return False, []
    buckets = [str(row["bucket"]).strip() for row in rows]
    return bool(all(buckets)), buckets


@contextmanager
def isolate_alpha_cache(output_dir: Path) -> Iterator[None]:
    live = Path("/tmp/regr_fail_llm_cache")
    backup = output_dir / "alpha_original_cache_backup"
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite cache backup: {backup}")
    had_live = live.exists()
    if had_live:
        live.rename(backup)
    try:
        yield
    finally:
        if live.exists():
            generated = output_dir / "alpha_last_generated_cache"
            if generated.exists():
                raise FileExistsError(f"refusing to overwrite generated cache: {generated}")
            live.rename(generated)
        if had_live and backup.exists():
            backup.rename(live)


def archive_alpha_cache(output_dir: Path, dataset_name: str) -> None:
    live = Path("/tmp/regr_fail_llm_cache")
    if not live.exists():
        return
    destination = output_dir / "caches" / f"alpha_{dataset_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite cold cache: {destination}")
    live.rename(destination)


def run_one(
    executable: Path,
    method: str,
    dataset: Path,
    output_dir: Path,
    extra_env: dict[str, str],
) -> dict:
    cases = osf.read_cases(dataset / "input.csv")
    gold = read_gold(osf.gold_path(dataset))
    k = len(set(gold))
    limit = final_runtime_limit(len(cases), k)
    pred_path = output_dir / "preds" / f"{method}_{dataset.name}.csv"
    stats_path = output_dir / "timing" / f"{method}_{dataset.name}.txt"
    log_path = output_dir / "logs" / f"{method}_{dataset.name}.log"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(extra_env)
    command = [
        "/usr/bin/time", "-f", "%e,%M,%x", "-o", str(stats_path),
        str(executable), "--input", str(dataset / "input.csv"),
        "--output", str(pred_path), "--k", str(k),
    ]
    started = time.perf_counter()
    timed_out = False
    return_code = -1
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    observed_wall = time.perf_counter() - started
    measured_wall = observed_wall
    max_rss_kb = 0
    if stats_path.exists():
        lines = [line.strip() for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            parts = lines[-1].split(",")
            if len(parts) == 3:
                measured_wall = float(parts[0])
                max_rss_kb = int(parts[1])
    valid, pred = validate_output(pred_path, cases)
    ba = tpr = tnr = float("nan")
    if valid:
        ba, tpr, tnr = pairwise_scores(gold, pred)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    route_lines = [line.strip() for line in log_text.splitlines() if "[beta-router]" in line]
    return {
        "method": method,
        "dataset": dataset.name,
        "cases": len(cases),
        "k": k,
        "final_limit_sec": limit,
        "wall_sec": measured_wall,
        "observed_wall_sec": observed_wall,
        "margin_sec": limit - observed_wall,
        "max_rss_kb": max_rss_kb,
        "return_code": return_code,
        "timed_out": timed_out,
        "valid_output": valid,
        "num_pred_clusters": len(set(pred)) if valid else 0,
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "route": " | ".join(route_lines[-4:]),
        "pred_path": str(pred_path),
        "log_path": str(log_path),
    }


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold-cache Alpha/Beta runtime comparison")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", choices=["alpha", "beta"], default=["alpha", "beta"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [(path if path.is_absolute() else ROOT / path).resolve() for path in args.datasets]
    executables = {
        "alpha": (ROOT / "alpha_test_submission/regr_fail_bucketing").resolve(),
        "beta": (ROOT / "beta_test_submission/regr_fail_bucketing").resolve(),
    }
    rows: list[dict] = []
    if "alpha" in args.methods:
        with isolate_alpha_cache(output_dir):
            for dataset in datasets:
                row = run_one(executables["alpha"], "alpha", dataset, output_dir, {})
                rows.append(row)
                archive_alpha_cache(output_dir, dataset.name)
                print(f"[runtime] alpha {dataset.name} wall={row['wall_sec']:.2f}s valid={row['valid_output']} timeout={row['timed_out']}", flush=True)
    if "beta" in args.methods:
        for dataset in datasets:
            beta_cache = output_dir / "caches" / f"beta_{dataset.name}"
            row = run_one(
                executables["beta"], "beta", dataset, output_dir,
                {"BETA_LLM_CACHE_DIR": str(beta_cache)},
            )
            rows.append(row)
            print(f"[runtime] beta {dataset.name} wall={row['wall_sec']:.2f}s valid={row['valid_output']} timeout={row['timed_out']}", flush=True)
    write_rows(output_dir / "results.csv", rows)
    return 0 if all(row["valid_output"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
