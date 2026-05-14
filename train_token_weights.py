#!/usr/bin/env python3
"""Train supervised token weights from local datasets.

This script is the only place, together with evaluation scripts, where gold.csv
is read. The official predictor regr_fail_bucketing.py only loads the produced
token_weights.json and never reads gold.csv/meta.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import regr_fail_bucketing as rfb


def pick_col(fieldnames: Sequence[str], names: Sequence[str], fallback: int = 0) -> str:
    normalized = {rfb.norm_col(name): name for name in fieldnames}
    for name in names:
        key = rfb.norm_col(name)
        if key in normalized:
            return normalized[key]
    for name in fieldnames:
        key = rfb.norm_col(name)
        if any(rfb.norm_col(candidate) in key for candidate in names):
            return name
    return fieldnames[min(fallback, len(fieldnames) - 1)]


def load_gold(dataset_dir: Path, input_rows: Sequence[dict], sim_col: str | None, regr_col: str | None) -> list[str]:
    gold_csv = dataset_dir / "gold.csv"
    rows, fields = rfb.read_csv_rows(gold_csv)
    if len(rows) != len(input_rows):
        raise ValueError(f"{gold_csv}: row count mismatch input={len(input_rows)} gold={len(rows)}")
    bug_col = pick_col(fields, ("bug_id", "bug", "gold", "label"), 1 if len(fields) > 1 else 0)
    case_col = pick_col(fields, ("case_id", "case", "id"), 0) if fields else None

    if case_col:
        by_case = {str(row.get(case_col, "")): str(row.get(bug_col, "")) for row in rows}
        input_case_ids = [
            rfb.case_id_from_row(row, idx, [c for c in (sim_col, regr_col) if c])
            for idx, row in enumerate(input_rows)
        ]
        if all(case_id in by_case for case_id in input_case_ids):
            return [by_case[case_id] for case_id in input_case_ids]
    return [str(row.get(bug_col, "")) for row in rows]


def build_training_docs(dataset_dir: Path, parser_args: argparse.Namespace) -> tuple[list[Counter], list[str]]:
    input_csv = dataset_dir / "input.csv"
    rows, fields = rfb.read_csv_rows(input_csv)
    sim_col = rfb.pick_column(fields, "sim")
    regr_col = rfb.pick_column(fields, "regr")
    if sim_col is None and regr_col is None:
        raise ValueError(f"{input_csv}: no sim/regr log columns")
    labels = load_gold(dataset_dir, rows, sim_col, regr_col)
    base, lines = rfb.collect_case_inputs(input_csv, rows, sim_col, regr_col, parser_args.parser)
    docs, _ = rfb.build_feature_counters(
        parser_args,
        base,
        lines,
        token_weights={},
        token_weight_mode="none",
    )
    return docs, labels


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def train_weights(args: argparse.Namespace) -> dict:
    parser_args = argparse.Namespace(
        parser=args.parser,
        drain_depth=args.drain_depth,
        drain_st=args.drain_st,
        drain_max_children=args.drain_max_children,
    )
    all_docs: list[Counter] = []
    all_labels: list[str] = []
    for dataset in args.datasets:
        docs, labels = build_training_docs(Path(dataset), parser_args)
        all_docs.extend(docs)
        all_labels.extend(labels)
        print(f"loaded {dataset}: cases={len(docs)}", file=sys.stderr)

    token_df: Counter = Counter()
    token_bug_counts: dict[str, Counter] = defaultdict(Counter)
    for doc, bug in zip(all_docs, all_labels):
        for token in set(doc.keys()):
            token_df[token] += 1
            token_bug_counts[token][bug] += 1

    num_cases = len(all_docs)
    weights: dict[str, float] = {}
    for token, df in token_df.items():
        is_primary = token.startswith("PRIMARY_")
        if df < args.min_df and not is_primary:
            continue
        idf = math.log((1 + num_cases) / (1 + df)) + 1
        purity = max(token_bug_counts[token].values()) / df if df else 0.0
        coverage_bonus = min(1.0, math.log(1 + df) / math.log(1 + 5))
        raw_weight = idf * purity * coverage_bonus
        if is_primary:
            raw_weight *= args.primary_boost
        weights[token] = clamp(raw_weight, args.min_weight, args.max_weight)

    return {
        "__meta__": {
            "num_cases": num_cases,
            "num_tokens": len(weights),
            "formula": "idf * purity * coverage_bonus",
            "min_df": args.min_df,
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
            "primary_boost": args.primary_boost,
            "datasets": [str(Path(d)) for d in args.datasets],
            "parser": args.parser,
            "drain_depth": args.drain_depth,
            "drain_st": args.drain_st,
            "drain_max_children": args.drain_max_children,
        },
        "weights": dict(sorted(weights.items())),
        "_debug": {
            "df": token_df,
        },
    }


def print_top_tokens(result: dict, top_n: int = 30) -> None:
    weights = result["weights"]
    df = result["_debug"]["df"]
    highest = sorted(weights.items(), key=lambda kv: (kv[1], df[kv[0]]), reverse=True)[:top_n]
    lowest = sorted(weights.items(), key=lambda kv: (kv[1], -df[kv[0]]))[:top_n]
    print("Top 30 highest-weight tokens", file=sys.stderr)
    for token, weight in highest:
        print(f"  {weight:.4f} df={df[token]} {token}", file=sys.stderr)
    print("Top 30 lowest-weight frequent tokens", file=sys.stderr)
    for token, weight in lowest:
        print(f"  {weight:.4f} df={df[token]} {token}", file=sys.stderr)
    print(f"Number of tokens saved: {len(weights)}", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised token weights.")
    parser.add_argument("--datasets", nargs="+", required=True, help="dataset directories containing input.csv/gold.csv")
    parser.add_argument("--output", required=True, type=Path, help="output token_weights.json")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--primary-boost", type=float, default=1.5)
    parser.add_argument("--parser", choices=("simple", "drain"), default="drain")
    parser.add_argument("--drain-depth", type=int, default=4)
    parser.add_argument("--drain-st", type=float, default=0.45)
    parser.add_argument("--drain-max-children", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = train_weights(args)
    print_top_tokens(result)
    debug = result.pop("_debug")
    del debug
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
