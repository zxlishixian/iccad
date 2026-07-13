#!/usr/bin/env python3
"""Experimental anytime wrapper for regression-failure bucketing backends.

The wrapper writes a valid singleton result before starting expensive work.
Backends write to sibling temporary files and only replace the public output
after strict validation.  Gold, meta, and trace files are never discovered.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


_ACTIVE_PROCESS: subprocess.Popen | None = None


@dataclass
class BackendResult:
    name: str
    status: str
    returncode: int | None
    runtime_sec: float
    output_valid: bool


def _normalized(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def read_case_ids(input_csv: Path) -> list[str]:
    with input_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields:
            return []
        case_col = next((name for name in fields if _normalized(name) == "case"), fields[0])
        return [str(row.get(case_col, "")).strip() for row in reader]


def _atomic_write_rows(path: Path, rows: Sequence[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Case", "bucket"])
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def write_singleton_output(path: Path, cases: Sequence[str]) -> None:
    _atomic_write_rows(
        path,
        [(case, f"bucket_emergency_{idx:06d}") for idx, case in enumerate(cases)],
    )


def validate_output(path: Path, expected_cases: Sequence[str]) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or []) != ["Case", "bucket"]:
                return False
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError):
        return False
    if len(rows) != len(expected_cases):
        return False
    actual_cases = [str(row.get("Case", "")).strip() for row in rows]
    if actual_cases != list(expected_cases):
        return False
    return all(str(row.get("bucket", "")).strip() for row in rows)


def _terminate_process(process: subprocess.Popen, grace_sec: float = 0.5) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        process.kill()
    process.wait()


def run_backend(
    name: str,
    command_prefix: Sequence[str],
    input_csv: Path,
    candidate_output: Path,
    k: int,
    timeout_sec: float,
    expected_cases: Sequence[str],
) -> BackendResult:
    global _ACTIVE_PROCESS
    command = [
        *map(str, command_prefix),
        "--input", str(input_csv),
        "--output", str(candidate_output),
        "--k", str(k),
    ]
    try:
        candidate_output.unlink(missing_ok=True)
    except OSError:
        pass
    started = time.perf_counter()
    try:
        process = subprocess.Popen(command, start_new_session=True)
    except OSError:
        return BackendResult(name, "start_error", None, time.perf_counter() - started, False)
    _ACTIVE_PROCESS = process

    try:
        returncode = process.wait(timeout=max(0.01, timeout_sec))
        status = "completed" if returncode == 0 else "backend_error"
    except subprocess.TimeoutExpired:
        _terminate_process(process)
        returncode = process.returncode
        status = "timeout"
    finally:
        if process.poll() is not None:
            _ACTIVE_PROCESS = None
    runtime = time.perf_counter() - started
    valid = returncode == 0 and validate_output(candidate_output, expected_cases)
    if returncode == 0 and not valid:
        status = "invalid_output"
    return BackendResult(name, status, returncode, runtime, valid)


def _candidate_path(output: Path, name: str) -> Path:
    return output.with_name(f".{output.name}.{name}.{os.getpid()}.candidate")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental anytime bucketing wrapper")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--primary-backend", type=Path, required=True)
    parser.add_argument("--primary-timeout", type=float, required=True)
    parser.add_argument("--fallback-backend", type=Path)
    parser.add_argument("--fallback-timeout", type=float, default=10.0)
    parser.add_argument("--diagnostics", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = read_case_ids(args.input)
    started = time.perf_counter()
    write_singleton_output(args.output, cases)
    diagnostics = {
        "cases": len(cases),
        "k": int(args.k),
        "initial_output_sec": time.perf_counter() - started,
        "backends": [],
        "selected": "singleton",
    }

    def stop_handler(_signum, _frame):
        if _ACTIVE_PROCESS is not None:
            _terminate_process(_ACTIVE_PROCESS)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    attempts = [("primary", args.primary_backend, args.primary_timeout)]
    if args.fallback_backend is not None:
        attempts.append(("fallback", args.fallback_backend, args.fallback_timeout))
    for name, backend, timeout_sec in attempts:
        candidate = _candidate_path(args.output, name)
        result = run_backend(
            name,
            [str(backend)],
            args.input,
            candidate,
            args.k,
            timeout_sec,
            cases,
        )
        diagnostics["backends"].append(asdict(result))
        if result.output_valid:
            os.replace(candidate, args.output)
            diagnostics["selected"] = name
            break
        candidate.unlink(missing_ok=True)

    diagnostics["total_sec"] = time.perf_counter() - started
    diagnostics["final_output_valid"] = validate_output(args.output, cases)
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[anytime] cases={len(cases)} selected={diagnostics['selected']} "
        f"initial={diagnostics['initial_output_sec']:.4f}s total={diagnostics['total_sec']:.3f}s",
        file=sys.stderr,
    )
    return 0 if diagnostics["final_output_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
