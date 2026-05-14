#!/usr/bin/env python3
"""Error analysis for regression-failure bucketing outputs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def norm_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def pick_column(fieldnames: Sequence[str], candidates: Sequence[str], fallback_idx: int = 0) -> str:
    normalized = {norm_col(name): name for name in fieldnames}
    for candidate in candidates:
        key = norm_col(candidate)
        if key in normalized:
            return normalized[key]
    for name in fieldnames:
        key = norm_col(name)
        if any(norm_col(candidate) in key for candidate in candidates):
            return name
    if not fieldnames:
        raise ValueError("CSV has no columns")
    return fieldnames[min(fallback_idx, len(fieldnames) - 1)]


def load_gold(path: Path) -> Tuple[List[str], List[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    case_col = pick_column(fields, ("case_id", "case", "id"), 0)
    bug_col = pick_column(fields, ("bug_id", "bug", "label", "gold"), 1 if len(fields) > 1 else 0)
    return [str(row.get(case_col, "") or f"row_{i}") for i, row in enumerate(rows)], [
        str(row.get(bug_col, "") or "UNKNOWN_BUG") for row in rows
    ]


def load_pred(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    bucket_col = pick_column(fields, ("bucket", "pred", "cluster", "label"), 0)
    return [str(row.get(bucket_col, "") or "UNKNOWN_BUCKET") for row in rows]


def choose(n: int) -> int:
    return comb(n, 2) if n >= 2 else 0


def pairwise_metrics(gold: Sequence[str], pred: Sequence[str]) -> Dict[str, float]:
    bug_counts = Counter(gold)
    bucket_counts = Counter(pred)
    joint_counts = Counter(zip(gold, pred))

    positive = sum(choose(n) for n in bug_counts.values())
    pred_positive = sum(choose(n) for n in bucket_counts.values())
    tp = sum(choose(n) for n in joint_counts.values())
    fn = positive - tp
    fp = pred_positive - tp
    total_pairs = choose(len(gold))
    negative = total_pairs - positive
    tn = negative - fp

    tpr = tp / positive if positive else 0.0
    tnr = tn / negative if negative else 0.0
    return {
        "cases": len(gold),
        "gold_bugs": len(bug_counts),
        "pred_buckets": len(bucket_counts),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tpr,
        "tnr": tnr,
        "ba": (tpr + tnr) / 2.0,
    }


def bug_fragmentation(gold: Sequence[str], pred: Sequence[str]) -> List[dict]:
    by_bug: Dict[str, Counter] = defaultdict(Counter)
    for bug, bucket in zip(gold, pred):
        by_bug[bug][bucket] += 1

    rows = []
    for bug, buckets in by_bug.items():
        size = sum(buckets.values())
        same_bug_pairs = choose(size)
        kept_pairs = sum(choose(n) for n in buckets.values())
        fn_pairs = same_bug_pairs - kept_pairs
        largest_bucket, largest_count = buckets.most_common(1)[0]
        rows.append(
            {
                "bug": bug,
                "size": size,
                "pred_bucket_count": len(buckets),
                "largest_bucket": largest_bucket,
                "largest_count": largest_count,
                "largest_share": largest_count / size if size else 0.0,
                "fn_pairs": fn_pairs,
                "fragmentation_rate": fn_pairs / same_bug_pairs if same_bug_pairs else 0.0,
                "distribution": buckets,
            }
        )
    rows.sort(key=lambda r: (r["fn_pairs"], r["pred_bucket_count"], -r["largest_share"]), reverse=True)
    return rows


def bucket_purity(gold: Sequence[str], pred: Sequence[str]) -> List[dict]:
    by_bucket: Dict[str, Counter] = defaultdict(Counter)
    for bug, bucket in zip(gold, pred):
        by_bucket[bucket][bug] += 1

    rows = []
    for bucket, bugs in by_bucket.items():
        size = sum(bugs.values())
        bucket_pairs = choose(size)
        pure_pairs = sum(choose(n) for n in bugs.values())
        fp_pairs = bucket_pairs - pure_pairs
        dominant_bug, dominant_count = bugs.most_common(1)[0]
        rows.append(
            {
                "bucket": bucket,
                "size": size,
                "gold_bug_count": len(bugs),
                "dominant_bug": dominant_bug,
                "dominant_count": dominant_count,
                "purity": dominant_count / size if size else 0.0,
                "fp_pairs": fp_pairs,
                "impurity_pair_rate": fp_pairs / bucket_pairs if bucket_pairs else 0.0,
                "distribution": bugs,
            }
        )
    rows.sort(key=lambda r: (r["fp_pairs"], r["gold_bug_count"], -r["purity"]), reverse=True)
    return rows


def fp_bug_pairs_by_bucket(gold: Sequence[str], pred: Sequence[str]) -> List[dict]:
    by_bucket: Dict[str, Counter] = defaultdict(Counter)
    for bug, bucket in zip(gold, pred):
        by_bucket[bucket][bug] += 1

    rows = []
    for bucket, bugs in by_bucket.items():
        items = sorted(bugs.items())
        for i, (bug_a, count_a) in enumerate(items):
            for bug_b, count_b in items[i + 1 :]:
                rows.append(
                    {
                        "bucket": bucket,
                        "bug_a": bug_a,
                        "bug_b": bug_b,
                        "count_a": count_a,
                        "count_b": count_b,
                        "fp_pairs": count_a * count_b,
                    }
                )
    rows.sort(key=lambda r: r["fp_pairs"], reverse=True)
    return rows


def format_counter(counter: Counter, limit: int = 8) -> str:
    parts = [f"{key}:{value}" for key, value in counter.most_common(limit)]
    if len(counter) > limit:
        parts.append(f"...(+{len(counter) - limit})")
    return ", ".join(parts)


def make_report(gold_path: Path, pred_path: Path, top: int) -> str:
    cases, gold = load_gold(gold_path)
    pred = load_pred(pred_path)
    if len(gold) != len(pred):
        raise ValueError(f"row count mismatch: gold={len(gold)} pred={len(pred)}")

    metrics = pairwise_metrics(gold, pred)
    frag = bug_fragmentation(gold, pred)
    purity = bucket_purity(gold, pred)
    fp_pairs = fp_bug_pairs_by_bucket(gold, pred)

    lines = []
    lines.append("# Error Analysis")
    lines.append("")
    lines.append(f"- gold: `{gold_path}`")
    lines.append(f"- pred: `{pred_path}`")
    lines.append(f"- cases: {metrics['cases']}")
    lines.append(f"- gold bugs: {metrics['gold_bugs']}")
    lines.append(f"- predicted buckets: {metrics['pred_buckets']}")
    lines.append(
        "- pairwise: "
        f"BA={metrics['ba']:.6f}, TPR={metrics['tpr']:.6f}, TNR={metrics['tnr']:.6f}, "
        f"TP={int(metrics['tp'])}, FN={int(metrics['fn'])}, FP={int(metrics['fp'])}, TN={int(metrics['tn'])}"
    )
    lines.append("")

    lines.append("## Gold Bug Fragmentation")
    lines.append("")
    lines.append("| bug | size | predicted buckets | largest bucket | largest share | FN pairs | fragmentation | distribution |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---|")
    for row in frag[:top]:
        lines.append(
            f"| {row['bug']} | {row['size']} | {row['pred_bucket_count']} | {row['largest_bucket']} "
            f"| {row['largest_share']:.3f} | {row['fn_pairs']} | {row['fragmentation_rate']:.3f} "
            f"| {format_counter(row['distribution'])} |"
        )
    lines.append("")

    lines.append("## Predicted Bucket Purity")
    lines.append("")
    lines.append("| bucket | size | gold bugs | dominant bug | purity | FP pairs | impurity pairs | distribution |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---|")
    for row in purity[:top]:
        lines.append(
            f"| {row['bucket']} | {row['size']} | {row['gold_bug_count']} | {row['dominant_bug']} "
            f"| {row['purity']:.3f} | {row['fp_pairs']} | {row['impurity_pair_rate']:.3f} "
            f"| {format_counter(row['distribution'])} |"
        )
    lines.append("")

    lines.append("## Top FN Bugs")
    lines.append("")
    lines.append("| bug | FN pairs | size | predicted buckets | largest share | distribution |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in frag[:top]:
        lines.append(
            f"| {row['bug']} | {row['fn_pairs']} | {row['size']} | {row['pred_bucket_count']} "
            f"| {row['largest_share']:.3f} | {format_counter(row['distribution'])} |"
        )
    lines.append("")

    lines.append("## Top FP Bucket Pairs")
    lines.append("")
    lines.append("| predicted bucket | bug A | bug B | count A | count B | FP pairs |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in fp_pairs[:top]:
        lines.append(
            f"| {row['bucket']} | {row['bug_a']} | {row['bug_b']} | {row['count_a']} "
            f"| {row['count_b']} | {row['fp_pairs']} |"
        )
    lines.append("")

    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze bucketing errors against gold.csv.")
    parser.add_argument("--gold", required=True, type=Path, help="gold.csv with case_id and bug_id")
    parser.add_argument("--pred", required=True, type=Path, help="predicted output.csv with bucket column")
    parser.add_argument("--top", type=int, default=12, help="number of rows per section")
    parser.add_argument("--out", type=Path, help="optional markdown report path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = make_report(args.gold, args.pred, args.top)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
