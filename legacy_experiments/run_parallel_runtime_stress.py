#!/usr/bin/env python3
"""Run unlabeled runtime-only profiles through the parallel anytime policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Sequence

import official_style_features as osf
from anytime_inference import validate_output


ROOT = Path(__file__).resolve().parent


def _read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _route(cases: int, final_limit: int) -> dict:
    if final_limit <= 30:
        baseline_timeout, expert_timeout, total_timeout = 7, 22, 26
    else:
        baseline_timeout, total_timeout = (12 if cases <= 900 else 20), 96
        expert_timeout = 72 if cases <= 100 else 70
    if cases <= 100:
        expert = ROOT / "beta_test_submission/multiview/regr_fail_bucketing_multiview"
    else:
        expert = None
    cluster = ["--llm-mode", "none", "--cluster", "agglomerative"]
    if cases > 900:
        cluster = ["--llm-mode", "none", "--cluster", "kmeans", "--cluster-factor", "1.0"]
    return {
        "baseline_timeout": baseline_timeout,
        "expert_timeout": expert_timeout,
        "total_timeout": total_timeout,
        "expert": expert,
        "baseline_extra": cluster,
    }


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_profile(dataset: Path, manifest_row: dict, output_dir: Path) -> dict:
    cases = int(manifest_row["cases"])
    k = int(manifest_row["k"])
    final_limit = int(manifest_row["final_limit_sec"])
    route = _route(cases, final_limit)
    pred_path = output_dir / "preds" / f"{dataset.name}.csv"
    diag_path = output_dir / "diagnostics" / f"{dataset.name}.json"
    log_path = output_dir / "logs" / f"{dataset.name}.log"
    time_path = output_dir / "timing" / f"{dataset.name}.txt"
    cache_path = output_dir / "caches" / dataset.name
    for path in (pred_path, diag_path, log_path, time_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/time", "-f", "%e,%M,%x", "-o", str(time_path),
        str(Path(os.environ.get("PYTHON_BIN", "/home/lishixian/miniforge3/envs/collab-overcooked/bin/python"))),
        str(ROOT / "parallel_anytime_runner.py"),
        "--input", str(dataset / "input.csv"),
        "--output", str(pred_path),
        "--k", str(k),
        "--baseline-backend", str(ROOT / "beta_test_submission/fast/regr_fail_bucketing_fast/regr_fail_bucketing_fast"),
        "--baseline-timeout", str(route["baseline_timeout"]),
        "--baseline-extra-json", json.dumps(route["baseline_extra"]),
        "--expert-timeout", str(route["expert_timeout"]),
        "--total-timeout", str(route["total_timeout"]),
        "--diagnostics", str(diag_path),
    ]
    if route["expert"] is not None:
        command.extend(["--expert-backend", str(route["expert"])])
    env = os.environ.copy()
    env["BETA_LLM_CACHE_DIR"] = str(cache_path)
    started = time.monotonic()
    timed_out = False
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
            returncode = process.wait(timeout=final_limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(process)
            returncode = process.returncode
    observed_wall = time.monotonic() - started
    wall, max_rss = observed_wall, 0
    if time_path.exists():
        lines = [line for line in time_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            parts = lines[-1].split(",")
            if len(parts) == 3:
                wall, max_rss = float(parts[0]), int(parts[1])
    expected_cases = [str(case) for case in osf.read_cases(dataset / "input.csv")]
    valid = validate_output(pred_path, expected_cases)
    diagnostics = {}
    if diag_path.exists():
        diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset.name,
        "cases": cases,
        "k": k,
        "target_max_lines": int(manifest_row["max_lines"]),
        "materialized_lines": int(manifest_row["materialized_lines"]),
        "final_limit_sec": final_limit,
        "wall_sec": wall,
        "margin_sec": final_limit - observed_wall,
        "max_rss_kb": max_rss,
        "timed_out": timed_out,
        "returncode": returncode,
        "valid_output": valid,
        "selected": diagnostics.get("selected", "unknown"),
        "expert_enabled": route["expert"] is not None,
        "log_path": str(log_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel anytime runtime-only stress runner")
    parser.add_argument("--runtime-root", type=Path, default=Path("runtime_only_benchmarks"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_root = args.runtime_root.resolve() if args.runtime_root.is_absolute() else (ROOT / args.runtime_root).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest_row in _read_manifest(runtime_root / "manifest.csv"):
        dataset = runtime_root / manifest_row["dataset"]
        row = run_profile(dataset, manifest_row, output_dir)
        rows.append(row)
        print(
            f"[runtime-stress] dataset={row['dataset']} wall={row['wall_sec']:.2f}s "
            f"selected={row['selected']} valid={row['valid_output']}",
            flush=True,
        )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["valid_output"] and not row["timed_out"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
