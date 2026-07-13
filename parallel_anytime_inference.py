#!/usr/bin/env python3
"""Experimental parallel baseline/expert anytime inference supervisor.

The public output is valid from the beginning.  Baseline and expert backends
run concurrently in separate process groups and write private candidates.  A
validated baseline replaces the singleton as soon as it finishes; a validated
expert may then replace the baseline.  Expert failure never removes a
previously published result.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from anytime_inference import (
    _candidate_path,
    _terminate_process,
    read_case_ids,
    validate_output,
    write_singleton_output,
)


@dataclass
class RunningBackend:
    name: str
    process: subprocess.Popen
    candidate: Path
    started: float
    timeout_sec: float
    handled: bool = False


@dataclass
class BackendOutcome:
    name: str
    status: str
    returncode: int | None
    runtime_sec: float
    output_valid: bool


def _parse_extra(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("backend extra arguments must be a JSON string list")
    return parsed


def start_backend(
    name: str,
    executable: Path,
    extra_args: Sequence[str],
    input_csv: Path,
    candidate: Path,
    k: int,
    timeout_sec: float,
) -> RunningBackend:
    candidate.unlink(missing_ok=True)
    command = [
        str(executable),
        "--input", str(input_csv),
        "--output", str(candidate),
        "--k", str(k),
        *extra_args,
    ]
    process = subprocess.Popen(command, start_new_session=True)
    return RunningBackend(name, process, candidate, time.monotonic(), timeout_sec)


def poll_backend(backend: RunningBackend, expected_cases: Sequence[str]) -> BackendOutcome | None:
    if backend.handled:
        return None
    now = time.monotonic()
    returncode = backend.process.poll()
    if returncode is None and now - backend.started < backend.timeout_sec:
        return None
    if returncode is None:
        _terminate_process(backend.process)
        returncode = backend.process.returncode
        status = "timeout"
    else:
        status = "completed" if returncode == 0 else "backend_error"
    runtime = time.monotonic() - backend.started
    valid = returncode == 0 and validate_output(backend.candidate, expected_cases)
    if returncode == 0 and not valid:
        status = "invalid_output"
    backend.handled = True
    return BackendOutcome(backend.name, status, returncode, runtime, valid)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel baseline/expert anytime inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--baseline-backend", type=Path, required=True)
    parser.add_argument("--expert-backend", type=Path)
    parser.add_argument("--baseline-timeout", type=float, default=12.0)
    parser.add_argument("--expert-timeout", type=float, default=70.0)
    parser.add_argument("--total-timeout", type=float, default=96.0)
    parser.add_argument("--baseline-extra-json", default='["--llm-mode", "none", "--cluster", "agglomerative"]')
    parser.add_argument("--expert-extra-json", default="[]")
    parser.add_argument("--diagnostics", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = read_case_ids(args.input)
    started = time.monotonic()
    write_singleton_output(args.output, cases)
    selected = "singleton"
    outcomes: list[BackendOutcome] = []
    active: list[RunningBackend] = []

    def stop_handler(_signum, _frame):
        for backend in active:
            if backend.process.poll() is None:
                _terminate_process(backend.process)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    try:
        backend_specs = [(
            "baseline",
            args.baseline_backend,
            args.baseline_extra_json,
            args.baseline_timeout,
        )]
        if args.expert_backend is not None and args.expert_timeout > 0:
            backend_specs.append((
                "expert",
                args.expert_backend,
                args.expert_extra_json,
                args.expert_timeout,
            ))
        for name, executable, extra_json, timeout_sec in backend_specs:
            backend_started = time.monotonic()
            try:
                active.append(start_backend(
                    name,
                    executable,
                    _parse_extra(extra_json),
                    args.input,
                    _candidate_path(args.output, f"parallel_{name}"),
                    args.k,
                    timeout_sec,
                ))
            except (OSError, ValueError) as exc:
                outcomes.append(BackendOutcome(
                    name=name,
                    status=f"startup_error:{type(exc).__name__}",
                    returncode=None,
                    runtime_sec=time.monotonic() - backend_started,
                    output_valid=False,
                ))
                print(
                    f"[parallel-anytime] {name} startup failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        while active and time.monotonic() - started < args.total_timeout:
            progressed = False
            for backend in active:
                outcome = poll_backend(backend, cases)
                if outcome is None:
                    continue
                progressed = True
                outcomes.append(outcome)
                if outcome.output_valid:
                    os.replace(backend.candidate, args.output)
                    selected = backend.name
                    print(
                        f"[parallel-anytime] published {backend.name} at "
                        f"{time.monotonic() - started:.3f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    if backend.name == "expert":
                        for other in active:
                            if other is not backend and other.process.poll() is None:
                                _terminate_process(other.process)
                        active = []
                        break
                backend.candidate.unlink(missing_ok=True)
            else:
                active = [backend for backend in active if not backend.handled]
                if not progressed:
                    time.sleep(0.02)
                continue
            break

        for backend in active:
            if backend.process.poll() is None:
                _terminate_process(backend.process)
            outcome = poll_backend(backend, cases)
            if outcome is not None:
                outcomes.append(outcome)
            backend.candidate.unlink(missing_ok=True)
    finally:
        for backend in active:
            if backend.process.poll() is None:
                _terminate_process(backend.process)
            backend.candidate.unlink(missing_ok=True)

    total = time.monotonic() - started
    final_valid = validate_output(args.output, cases)
    diagnostics = {
        "cases": len(cases),
        "k": int(args.k),
        "selected": selected,
        "total_sec": total,
        "final_output_valid": final_valid,
        "outcomes": [asdict(outcome) for outcome in outcomes],
    }
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[parallel-anytime] selected={selected} total={total:.3f}s valid={final_valid}",
        file=sys.stderr,
    )
    return 0 if final_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
