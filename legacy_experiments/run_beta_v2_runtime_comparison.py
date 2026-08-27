#!/usr/bin/env python3
"""Run the experimental Beta v2 router under the existing cold-cache harness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from run_submission_runtime_comparison import (
    DEFAULT_DATASETS,
    ROOT,
    run_one,
    write_rows,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold-cache Beta v2 runtime comparison")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        (path if path.is_absolute() else ROOT / path).resolve()
        for path in args.datasets
    ]
    executable = (ROOT / "beta_v2_router.sh").resolve()
    rows = []
    for dataset in datasets:
        cache_dir = output_dir / "caches" / f"beta_v2_{dataset.name}"
        row = run_one(
            executable,
            "beta_v2",
            dataset,
            output_dir,
            {"BETA_LLM_CACHE_DIR": str(cache_dir)},
        )
        rows.append(row)
        print(
            f"[runtime] beta_v2 {dataset.name} wall={row['wall_sec']:.2f}s "
            f"valid={row['valid_output']} timeout={row['timed_out']}",
            flush=True,
        )
    write_rows(output_dir / "results.csv", rows)
    return 0 if all(row["valid_output"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
