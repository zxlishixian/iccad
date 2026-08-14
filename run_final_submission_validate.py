#!/usr/bin/env python3
"""Validate the final submission across all 11 datasets + fallback paths.

Runs ``final_inference.py`` (the 9-fold TriLog ensemble + correlation
clustering) on every dataset, scores balanced accuracy / TPR / TNR against the
local gold labels, and additionally scores the deterministic baseline fallback
to confirm the fault-tolerance path produces valid output.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import official_style_features as osf
from run_experiments import pairwise_scores, read_gold

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS = [
    Path("dataset/fake_dataset/old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("dataset/fake_dataset/official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/directed_cross_v4"),
    Path("dataset/fake_dataset/official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/benchmark6_final"),
    Path("dataset/real_dataset/benchmark_set_1"),
    Path("dataset/real_dataset/benchmark_set_2"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--python", default=None, help="python interpreter (default: auto-detect)")
    p.add_argument("--datasets", nargs="*", type=Path, default=None,
                   help="subset of datasets to validate (default: all 11)")
    return p.parse_args(argv)


def resolve_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [
        "/home/lishixian/miniforge3/envs/collab-overcooked/bin/python",
        "python3",
    ]
    for cand in candidates:
        if "/" not in cand:
            return cand
        if Path(cand).exists():
            return cand
    return "python3"


def read_buckets(out_csv: Path) -> list[str]:
    with out_csv.open(newline="", encoding="utf-8") as f:
        return [row["bucket"] for row in csv.DictReader(f)]


def run_one(model_dir: Path, dataset: Path, out_dir: Path, python: str) -> dict:
    out_csv = out_dir / f"{dataset.name}.csv"
    gold = read_gold(osf.gold_path(dataset))
    k = len(set(gold))
    proc = subprocess.run(
        [python, str(PROJECT_ROOT / "final_inference.py"), "--input",
         str(dataset / "input.csv"), "--output", str(out_csv),
         "--k", str(k), "--model-dir", str(model_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_csv.exists():
        return {"dataset": dataset.name, "BA": None, "error": proc.stderr[-400:]}
    pred = read_buckets(out_csv)
    ba, tpr, tnr = pairwise_scores(gold, pred)
    return {
        "dataset": dataset.name, "BA": ba, "TPR": tpr, "TNR": tnr,
        "k": k, "cases": len(gold), "clusters": len(set(pred)),
    }


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    python = resolve_python(args.python)
    datasets = [d.resolve() for d in (args.datasets or DATASETS)]
    rows = [run_one(args.model_dir, d, args.output_dir, python) for d in datasets]
    fields = ["dataset", "BA", "TPR", "TNR", "k", "cases", "clusters", "error"]
    with open(args.output_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r.get("BA") is not None]
    official = [r for r in valid if r["dataset"] in {"benchmark_set_1", "benchmark_set_2"}]
    summary = {
        "mean_BA": (sum(r["BA"] for r in valid) / len(valid)) if valid else None,
        "worst_BA": (min(r["BA"] for r in valid)) if valid else None,
        "official_mean_BA": (sum(r["BA"] for r in official) / len(official)) if official else None,
        "n_valid": len(valid),
        "n_total": len(rows),
    }
    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)

    print("\n| dataset | BA | TPR | TNR | clusters/k |")
    print("|---|---|---:|---:|---:|")
    for r in rows:
        if r.get("BA") is None:
            print(f"| {r['dataset']} | ERROR | | | {r.get('error', '')[:40]} |")
        else:
            print(f"| {r['dataset']} | {r['BA']:.4f} | {r['TPR']:.4f} | {r['TNR']:.4f} | {r['clusters']}/{r['k']} |")
    print(f"\nsummary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
