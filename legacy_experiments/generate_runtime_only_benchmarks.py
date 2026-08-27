#!/usr/bin/env python3
"""Generate unlabeled runtime-only benchmarks from public official logs.

The generated datasets are intentionally unsuitable for training or score
reporting.  Cases recombine public sim/regr evidence and add unique neutral
padding so embedding deduplication cannot make large-case timing unrealistically
cheap.  No gold, golden, meta, or bug labels are written.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import regr_fail_bucketing as rfb


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = [
    Path("test_case/problem/benchmark_set_1"),
    Path("test_case/problem/benchmark_set_2"),
]


@dataclass(frozen=True)
class RuntimeProfile:
    benchmark_id: int
    max_lines: int
    cases: int
    k: int
    final_limit_sec: int


PROFILES = {
    1: RuntimeProfile(1, 1_000_000, 10, 2, 30),
    2: RuntimeProfile(2, 1_000_000, 30, 4, 30),
    3: RuntimeProfile(3, 10_000_000, 100, 8, 100),
    4: RuntimeProfile(4, 10_000_000, 300, 16, 100),
    5: RuntimeProfile(5, 10_000_000, 1000, 32, 100),
    6: RuntimeProfile(6, 10_000_000, 3000, 64, 100),
    7: RuntimeProfile(7, 100_000_000, 100, 8, 300),
    8: RuntimeProfile(8, 100_000_000, 300, 16, 300),
    9: RuntimeProfile(9, 100_000_000, 1000, 32, 300),
    10: RuntimeProfile(10, 100_000_000, 3000, 64, 300),
}


@dataclass
class SourceEvidence:
    source_dataset: str
    source_case: str
    sim_lines: list[str]
    regr_lines: list[str]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _compact_lines(text: str, max_lines: int = 32) -> list[str]:
    selected = rfb.select_lines(text, mode="signal_window")
    if not selected:
        selected = text.splitlines()[:max_lines]
    output = []
    for line in selected:
        compact = " ".join(line.strip().split())
        if compact and compact not in output:
            output.append(compact[:300])
        if len(output) >= max_lines:
            break
    return output or ["RUNTIME_ONLY_NO_SIGNAL"]


def load_source_evidence(source_dirs: Sequence[Path]) -> list[SourceEvidence]:
    output: list[SourceEvidence] = []
    for source_dir in source_dirs:
        input_csv = source_dir / "input.csv"
        rows, fields = rfb.read_csv_rows(input_csv)
        sim_col = rfb.pick_column(fields, "sim")
        regr_col = rfb.pick_column(fields, "regr")
        case_col = next((field for field in fields if "".join(ch for ch in field.lower() if ch.isalnum()) == "case"), None)
        for index, row in enumerate(rows, 1):
            sim_path = rfb.resolve_log_path(input_csv, row.get(sim_col) if sim_col else None)
            regr_path = rfb.resolve_log_path(input_csv, row.get(regr_col) if regr_col else None)
            sim_text, _sim_status = rfb.read_log_sample(sim_path)
            regr_text, _regr_status = rfb.read_log_sample(regr_path)
            output.append(SourceEvidence(
                source_dataset=source_dir.name,
                source_case=str(row.get(case_col, index)) if case_col else str(index),
                sim_lines=_compact_lines(sim_text),
                regr_lines=_compact_lines(regr_text),
            ))
    if not output:
        raise ValueError("no public source evidence was loaded")
    return output


def _combined_lines(
    left: SourceEvidence,
    right: SourceEvidence,
    kind: str,
    case_index: int,
) -> list[str]:
    left_lines = left.sim_lines if kind == "sim" else left.regr_lines
    right_lines = right.sim_lines if kind == "sim" else right.regr_lines
    split_left = max(1, len(left_lines) // 2)
    split_right = max(1, len(right_lines) // 2)
    return [
        f"RUNTIME_ONLY_CASE={case_index} SOURCE_A={left.source_dataset}:{left.source_case}",
        *left_lines[:split_left],
        f"RUNTIME_ONLY_COMBINE SOURCE_B={right.source_dataset}:{right.source_case}",
        *right_lines[-split_right:],
    ]


def _padding_line(case_index: int, line_index: int) -> str:
    pc = 0x80000000 + ((case_index * 131 + line_index * 4) & 0xFFFFC)
    reg = (case_index + line_index) % 32
    opcode = ("addi", "lw", "sw", "beq", "csrrw", "xor")[(case_index + line_index) % 6]
    return (
        f"{line_index:09d} RUNTIME_ONLY neutral retire case={case_index:05d} "
        f"pc=0x{pc:08x} opcode={opcode} x{reg}=0x{(pc ^ line_index):08x}"
    )


def _write_case(
    case_dir: Path,
    case_index: int,
    left: SourceEvidence,
    right: SourceEvidence,
    lines_per_case: int,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=False)
    regr_lines = _combined_lines(left, right, "regr", case_index)
    sim_lines = _combined_lines(left, right, "sim", case_index)
    padding_needed = max(0, lines_per_case - len(regr_lines) - len(sim_lines))
    regr_padding = padding_needed // 4
    sim_padding = padding_needed - regr_padding
    with (case_dir / "regr.log").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(regr_lines))
        handle.write("\n")
        for line_index in range(regr_padding):
            handle.write(_padding_line(case_index, line_index) + "\n")
    with gzip.open(case_dir / "sim.log.gz", "wt", encoding="utf-8", compresslevel=3) as handle:
        handle.write("\n".join(sim_lines))
        handle.write("\n")
        for line_index in range(sim_padding):
            handle.write(_padding_line(case_index, regr_padding + line_index) + "\n")
    with gzip.open(case_dir / "trace.log.gz", "wt", encoding="utf-8", compresslevel=3) as handle:
        handle.write(
            f"RUNTIME_ONLY trace intentionally minimal case={case_index}; "
            "formal no-trace routes must ignore this file\n"
        )


def generate_profile(
    output_root: Path,
    profile: RuntimeProfile,
    evidence: Sequence[SourceEvidence],
    max_materialized_lines: int,
) -> dict:
    dataset = output_root / (
        f"benchmark_{profile.benchmark_id:02d}_n{profile.cases}_k{profile.k}_runtime_only"
    )
    if dataset.exists():
        raise FileExistsError(f"refusing to overwrite generated dataset: {dataset}")
    dataset.mkdir(parents=True)
    materialized_total = profile.max_lines
    if max_materialized_lines > 0:
        materialized_total = min(materialized_total, max_materialized_lines)
    lines_per_case = max(32, math.ceil(materialized_total / profile.cases))
    materialized_total = lines_per_case * profile.cases
    input_path = dataset / "input.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Case", "Regr Log", "Sim Log", "Trace Log"])
        for case_index in range(1, profile.cases + 1):
            left = evidence[(case_index - 1) % len(evidence)]
            right = evidence[(case_index * 7 + profile.benchmark_id * 3) % len(evidence)]
            _write_case(dataset / f"case_{case_index}", case_index, left, right, lines_per_case)
            writer.writerow([
                case_index,
                f"case_{case_index}/regr.log",
                f"case_{case_index}/sim.log.gz",
                f"case_{case_index}/trace.log.gz",
            ])
    metadata = {
        **asdict(profile),
        "runtime_only": True,
        "labels_available": False,
        "training_allowed": False,
        "score_reporting_allowed": False,
        "materialized_lines": materialized_total,
        "materialization_ratio": materialized_total / profile.max_lines,
        "lines_per_case": lines_per_case,
        "source_datasets": sorted({item.source_dataset for item in evidence}),
    }
    (dataset / "RUNTIME_ONLY.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"dataset": dataset.name, **metadata}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unlabeled official-style runtime stress datasets")
    parser.add_argument("--output-root", type=Path, default=Path("runtime_only_benchmarks"))
    parser.add_argument("--source-datasets", nargs="+", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--profiles", nargs="+", type=int, choices=sorted(PROFILES), default=[1, 2, 3, 4, 5, 6])
    parser.add_argument(
        "--max-materialized-lines", type=int, default=100_000,
        help="Per-benchmark line cap; 0 materializes the full official Max Lines target.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = resolve(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_dirs = [resolve(path) for path in args.source_datasets]
    evidence = load_source_evidence(source_dirs)
    rows = [
        generate_profile(output_root, PROFILES[profile_id], evidence, args.max_materialized_lines)
        for profile_id in args.profiles
    ]
    fields = list(rows[0])
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "README.md").write_text(
        "# Runtime-Only Benchmarks\n\n"
        "These datasets are derived from public official logs exclusively for runtime, "
        "watchdog, memory, and output-format testing. They contain no labels and MUST NOT "
        "be used for training, model selection, calibration, or score reporting.\n\n"
        "`max_lines` is the official target. `materialized_lines` records the actual local "
        "stress size; use `--max-materialized-lines 0` only when intentionally generating "
        "the full, potentially very large context workload.\n",
        encoding="utf-8",
    )
    print(f"[runtime-generator] datasets={len(rows)} evidence={len(evidence)} root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
