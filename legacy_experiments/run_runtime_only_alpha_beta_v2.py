#!/usr/bin/env python3
"""Compare frozen Alpha and Beta v2 packages on unlabeled runtime profiles.

This harness deliberately never discovers or opens gold, golden, or meta files.
It consumes the generated manifest only for case count, reference k, and the
Final runtime limit, then invokes each package through its public CLI.
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


ROOT = Path(__file__).resolve().parent


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_output(path: Path, expected_cases: Sequence[str]) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Case", "bucket"]:
            return False
        rows = list(reader)
    return (
        [str(row.get("Case", "")) for row in rows] == [str(case) for case in expected_cases]
        and all(str(row.get("bucket", "")).strip() for row in rows)
    )


def terminate(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def isolate_alpha_cache(work_dir: Path) -> tuple[Path, Path | None]:
    live = Path("/tmp/regr_fail_llm_cache")
    backup = work_dir / "alpha_cache_original"
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite {backup}")
    if live.exists():
        live.rename(backup)
        return live, backup
    return live, None


def restore_alpha_cache(live: Path, backup: Path | None, work_dir: Path) -> None:
    if live.exists():
        generated = work_dir / "alpha_cache_generated"
        if generated.exists():
            shutil.rmtree(generated)
        live.rename(generated)
    if backup is not None:
        backup.rename(live)


def run_one(
    method: str,
    executable: Path,
    dataset: Path,
    k: int,
    final_limit: int,
    output_dir: Path,
    no_timeout: bool,
) -> dict[str, object]:
    cases = [str(case) for case in osf.read_cases(dataset / "input.csv")]
    pred_path = output_dir / "preds" / f"{method}_{dataset.name}.csv"
    log_path = output_dir / "logs" / f"{method}_{dataset.name}.log"
    timing_path = output_dir / "timing" / f"{method}_{dataset.name}.txt"
    for path in (pred_path, log_path, timing_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if method == "beta_v2":
        env["BETA_LLM_CACHE_DIR"] = str(output_dir / "caches" / f"beta_v2_{dataset.name}")
    command = [
        "/usr/bin/time", "-f", "%e,%M,%x", "-o", str(timing_path),
        str(executable), "--input", str(dataset / "input.csv"),
        "--output", str(pred_path), "--k", str(k),
    ]
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
        if no_timeout:
            returncode = process.wait()
        else:
            try:
                returncode = process.wait(timeout=final_limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate(process)
                returncode = process.returncode
    observed_wall = time.monotonic() - started
    wall_sec, max_rss_kb = observed_wall, 0
    if timing_path.exists():
        parts = timing_path.read_text(encoding="utf-8").strip().split(",")
        if len(parts) == 3:
            wall_sec, max_rss_kb = float(parts[0]), int(parts[1])
    return {
        "method": method,
        "dataset": dataset.name,
        "cases": len(cases),
        "k": k,
        "final_limit_sec": final_limit,
        "wall_sec": wall_sec,
        "observed_wall_sec": observed_wall,
        "margin_sec": final_limit - observed_wall,
        "max_rss_kb": max_rss_kb,
        "returncode": returncode,
        "timed_out": timed_out,
        "valid_output": validate_output(pred_path, cases),
        "pred_path": str(pred_path),
        "log_path": str(log_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unlabeled Alpha vs Beta v2 runtime comparison")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["alpha", "beta_v2"], default=["alpha", "beta_v2"])
    parser.add_argument("--no-timeout", action="store_true", help="Measure natural completion time; never terminate a package at the Final limit.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_root = args.runtime_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    executables = {
        "alpha": ROOT / "alpha_test_submission" / "regr_fail_bucketing",
        "beta_v2": ROOT / "beta_test_submission_v2" / "regr_fail_bucketing",
    }
    rows: list[dict[str, object]] = []
    alpha_live = alpha_backup = None
    if "alpha" in args.methods:
        alpha_live, alpha_backup = isolate_alpha_cache(output_dir)
    try:
        for manifest_row in read_manifest(runtime_root / "manifest.csv"):
            dataset = runtime_root / manifest_row["dataset"]
            for method in args.methods:
                if method == "alpha" and alpha_live is not None:
                    shutil.rmtree(alpha_live, ignore_errors=True)
                row = run_one(
                    method, executables[method], dataset,
                    int(manifest_row["k"]), int(manifest_row["final_limit_sec"]), output_dir,
                    args.no_timeout,
                )
                rows.append(row)
                print(
                    f"[runtime-only] {method} {dataset.name} wall={row['wall_sec']:.2f}s "
                    f"valid={row['valid_output']} timeout={row['timed_out']}",
                    flush=True,
                )
    finally:
        if alpha_live is not None:
            restore_alpha_cache(alpha_live, alpha_backup, output_dir)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(bool(row["valid_output"]) for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
