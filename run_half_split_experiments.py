#!/usr/bin/env python3
"""Experimental half-split cross-scale validation experiments.

These scripts are research utilities for supervised token weighting experiments.
Current validation showed that learned token weights did not consistently
outperform the no-weight baseline, so they are not enabled by default.

This is an evaluation/training utility, so it may read gold.csv. The official
predictor regr_fail_bucketing.py must remain gold/meta-free.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

import regr_fail_bucketing as rfb
import train_token_weights as ttw
from run_experiments import pairwise_scores, read_gold, read_pred


DEFAULT_DATASETS = [
    Path("dataset/first_batch_dataset"),
    Path("dataset/stage2_dataset_working"),
    Path("dataset/stage3_dataset_32bugs_640cases"),
]
PARTS = ("part1", "part2")


def dataset_name(path: Path) -> str:
    return path.name


def read_csv_dicts(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv_dicts(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def align_gold(dataset_dir: Path, input_rows: Sequence[dict], input_fields: Sequence[str]) -> tuple[list[str], list[str]]:
    gold_rows, gold_fields = read_csv_dicts(dataset_dir / "gold.csv")
    if len(gold_rows) != len(input_rows):
        raise ValueError(f"{dataset_dir}: input/gold row mismatch")
    sim_col = rfb.pick_column(input_fields, "sim")
    regr_col = rfb.pick_column(input_fields, "regr")
    bug_col = ttw.pick_col(gold_fields, ("bug_id", "bug", "gold", "label"), 1 if len(gold_fields) > 1 else 0)
    case_col = ttw.pick_col(gold_fields, ("case_id", "case", "id"), 0)
    case_ids = [
        rfb.case_id_from_row(row, idx, [c for c in (sim_col, regr_col) if c])
        for idx, row in enumerate(input_rows)
    ]
    by_case = {str(row.get(case_col, "")): str(row.get(bug_col, "")) for row in gold_rows}
    if all(case_id in by_case for case_id in case_ids):
        return case_ids, [by_case[case_id] for case_id in case_ids]
    return [str(row.get(case_col, f"case_{idx + 1:06d}")) for idx, row in enumerate(gold_rows)], [
        str(row.get(bug_col, "")) for row in gold_rows
    ]


def absolutize_input_row(dataset_dir: Path, row: dict, fields: Sequence[str]) -> dict:
    out = dict(row)
    for wanted in ("sim", "regr", "trace"):
        col = rfb.pick_column(fields, wanted)
        if not col or not out.get(col):
            continue
        path = Path(str(out[col]))
        out[col] = str(path if path.is_absolute() else (dataset_dir / path).resolve())
    return out


def stratified_half_split(dataset_dir: Path, seed: int, output_dir: Path) -> dict:
    """
    Return:
    {
      "part1": {"input": Path, "gold": Path, "k": int, "num_cases": int},
      "part2": {"input": Path, "gold": Path, "k": int, "num_cases": int},
    }
    """
    input_rows, input_fields = read_csv_dicts(dataset_dir / "input.csv")
    case_ids, bug_ids = align_gold(dataset_dir, input_rows, input_fields)
    rng = random.Random(seed)
    by_bug: Dict[str, list[int]] = defaultdict(list)
    for idx, bug_id in enumerate(bug_ids):
        by_bug[bug_id].append(idx)

    part_indices = {"part1": [], "part2": []}
    for bug_id, indices in sorted(by_bug.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            part_indices[rng.choice(PARTS)].append(shuffled[0])
            continue
        cut = (len(shuffled) + 1) // 2
        part_indices["part1"].extend(shuffled[:cut])
        part_indices["part2"].extend(shuffled[cut:])

    result = {}
    for part in PARTS:
        indices = sorted(part_indices[part])
        part_dir = output_dir / dataset_name(dataset_dir) / part
        part_input_rows = [absolutize_input_row(dataset_dir, input_rows[i], input_fields) for i in indices]
        part_gold_rows = [{"case_id": case_ids[i], "bug_id": bug_ids[i]} for i in indices]
        input_path = part_dir / "input.csv"
        gold_path = part_dir / "gold.csv"
        write_csv_dicts(input_path, input_fields, part_input_rows)
        write_csv_dicts(gold_path, ["case_id", "bug_id"], part_gold_rows)
        unique_bugs = {row["bug_id"] for row in part_gold_rows}
        result[part] = {
            "input": input_path,
            "gold": gold_path,
            "k": max(1, len(unique_bugs)),
            "num_cases": len(part_gold_rows),
        }
    return result


def train_repeat_weights(python: str, train_dirs: Sequence[Path], output_path: Path, args: argparse.Namespace) -> Path:
    cmd = [
        python,
        "train_token_weights.py",
        "--datasets",
        *[str(path) for path in train_dirs],
        "--output",
        str(output_path),
        "--min-df",
        str(args.train_min_df),
        "--max-weight",
        str(args.max_weight),
        "--min-weight",
        str(args.min_weight),
        "--primary-boost",
        str(args.primary_boost),
        "--parser",
        args.parser,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return output_path


def make_conservative_weights(repeat_path: Path, output_path: Path) -> Path:
    data = json.loads(repeat_path.read_text(encoding="utf-8"))
    weights = data.get("weights", {})
    conservative = {}
    for token, weight in weights.items():
        value = max(0.5, min(2.0, float(weight)))
        if token.startswith("PRIMARY_"):
            value = min(value * 1.1, 2.0)
        conservative[token] = value
    out = {
        "__meta__": {
            **data.get("__meta__", {}),
            "mode": "conservative",
            "description": "Clipped repeat weights to [0.5, 2.0], with primary tokens capped after a small boost.",
        },
        "weights": conservative,
    }
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def is_noise_token(token: str) -> bool:
    common_noise = {
        "parser:drain",
        "regr:basename:regr.log",
        "sim:basename:sim.log",
        "regr:file_status:ok",
        "sim:file_status:ok",
    }
    if token in common_noise:
        return True
    if token.startswith("PRIMARY_"):
        return False
    low = token.lower()
    if "<num>" in low or "<n>" in low:
        meaningful = any(word in low for word in ("mismatch", "fatal", "error", "trap", "pc", "register", "cosim"))
        return not meaningful
    return False


def make_blacklist_weights(train_dirs: Sequence[Path], output_path: Path, args: argparse.Namespace) -> Path:
    parser_args = argparse.Namespace(
        parser=args.parser,
        drain_depth=4,
        drain_st=0.45,
        drain_max_children=100,
    )
    docs = []
    labels = []
    for dataset_dir in train_dirs:
        dataset_docs, dataset_labels = ttw.build_training_docs(dataset_dir, parser_args)
        docs.extend(dataset_docs)
        labels.extend(dataset_labels)

    token_df: Counter = Counter()
    token_bug_counts: dict[str, Counter] = defaultdict(Counter)
    for doc, bug_id in zip(docs, labels):
        for token in set(doc.keys()):
            token_df[token] += 1
            token_bug_counts[token][bug_id] += 1

    num_bugs = len(set(labels))
    blacklist = {}
    for token, df in token_df.items():
        if token.startswith("PRIMARY_") or df < args.train_min_df:
            continue
        bug_counts = token_bug_counts[token]
        purity = max(bug_counts.values()) / df if df else 0.0
        appears_in_many_bugs = len(bug_counts) >= max(4, int(0.35 * max(1, num_bugs)))
        if purity < 0.35 or appears_in_many_bugs or is_noise_token(token):
            blacklist[token] = 0.0

    out = {
        "__meta__": {
            "mode": "blacklist",
            "description": "Only low-value tokens are assigned weight 0.0. Missing tokens keep default repeat.",
            "num_cases": len(docs),
            "num_blacklisted_tokens": len(blacklist),
            "min_df": args.train_min_df,
        },
        "weights": dict(sorted(blacklist.items())),
    }
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def ensure_weights(
    mode: str,
    python: str,
    train_dirs: Sequence[Path],
    weights_dir: Path,
    key: str,
    args: argparse.Namespace,
) -> Path | None:
    if mode == "none":
        return None
    weights_dir.mkdir(parents=True, exist_ok=True)
    repeat_path = weights_dir / f"{key}_repeat.json"
    if mode in {"repeat", "conservative"} and not repeat_path.exists():
        train_repeat_weights(python, train_dirs, repeat_path, args)
    if mode == "repeat":
        return repeat_path
    if mode == "conservative":
        out = weights_dir / f"{key}_conservative.json"
        if not out.exists():
            make_conservative_weights(repeat_path, out)
        return out
    out = weights_dir / f"{key}_blacklist.json"
    if not out.exists():
        make_blacklist_weights(train_dirs, out, args)
    return out


def run_predict(
    python: str,
    input_csv: Path,
    output_csv: Path,
    k: int,
    weight_mode: str,
    token_weights: Path | None,
    cluster_factor: float,
    args: argparse.Namespace,
) -> float:
    cmd = [
        python,
        "regr_fail_bucketing.py",
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
        "--k",
        str(k),
        "--parser",
        args.parser,
        "--cluster",
        args.cluster,
        "--cluster-factor",
        str(cluster_factor),
        "--svd-dim",
        str(args.svd_dim),
        "--feature-level",
        args.feature_level,
        "--normalizer",
        args.normalizer,
        "--line-mode",
        args.line_mode,
        "--template-weighting",
        args.template_weighting,
        "--llm-mode",
        args.llm_mode,
        "--llm-weight",
        str(args.llm_weight),
        "--llm-cache-dir",
        str(args.llm_cache_dir),
        "--token-weight-mode",
        "none" if weight_mode == "none" else "repeat",
    ]
    if token_weights:
        cmd.extend(["--token-weights", str(token_weights)])
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    runtime = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return runtime


def write_results_csv(path: Path, rows: Sequence[dict]) -> None:
    header = [
        "seed",
        "combo",
        "dataset",
        "val_part",
        "train_part_for_dataset",
        "val_part_for_dataset",
        "weight_mode",
        "cluster_factor",
        "feature_level",
        "svd_dim",
        "normalizer",
        "line_mode",
        "template_weighting",
        "llm_mode",
        "llm_weight",
        "parser",
        "cluster",
        "k",
        "num_cases",
        "num_pred_clusters",
        "BA",
        "TPR",
        "TNR",
        "runtime_sec",
        "token_weights_path",
        "pred_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: Dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["weight_mode"], row["cluster_factor"], row["dataset"])].append(row)
    summary = []
    for (weight_mode, cluster_factor, dataset), items in sorted(groups.items()):
        bas = [float(row["BA"]) for row in items]
        tprs = [float(row["TPR"]) for row in items]
        tnrs = [float(row["TNR"]) for row in items]
        clusters = [float(row["num_pred_clusters"]) for row in items]
        summary.append(
            {
                "weight_mode": weight_mode,
                "cluster_factor": cluster_factor,
                "dataset": dataset,
                "mean_BA": mean(bas),
                "std_BA": std(bas),
                "mean_TPR": mean(tprs),
                "std_TPR": std(tprs),
                "mean_TNR": mean(tnrs),
                "std_TNR": std(tnrs),
                "mean_num_pred_clusters": mean(clusters),
                "num_runs": len(items),
            }
        )
    return summary


def write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    header = [
        "weight_mode",
        "cluster_factor",
        "dataset",
        "mean_BA",
        "std_BA",
        "mean_TPR",
        "std_TPR",
        "mean_TNR",
        "std_TNR",
        "mean_num_pred_clusters",
        "num_runs",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown_summary(rows: Sequence[dict]) -> None:
    print("| weight_mode | cluster_factor | dataset | mean_BA | mean_TPR | mean_TNR | runs |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['weight_mode']} | {float(row['cluster_factor']):.2f} | {row['dataset']} "
            f"| {float(row['mean_BA']):.6f} | {float(row['mean_TPR']):.6f} "
            f"| {float(row['mean_TNR']):.6f} | {row['num_runs']} |"
        )


def part_for_bit(bit: int) -> str:
    return "part2" if bit else "part1"


def opposite_part(part: str) -> str:
    return "part1" if part == "part2" else "part2"


def run_experiments(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    output_dir = args.output_dir
    splits_root = output_dir / "splits"
    weights_dir = output_dir / "weights"
    preds_dir = output_dir / "preds"
    dataset_dirs = [Path(path) for path in args.datasets]
    results = []

    for seed in args.seeds:
        seed_splits = {}
        for dataset_dir in dataset_dirs:
            seed_splits[dataset_name(dataset_dir)] = stratified_half_split(
                dataset_dir,
                seed,
                splits_root / f"seed_{seed}",
            )

        for combo in range(1 << len(dataset_dirs)):
            train_dirs_by_dataset = {}
            val_infos = []
            for idx, dataset_dir in enumerate(dataset_dirs):
                name = dataset_name(dataset_dir)
                train_part = part_for_bit((combo >> idx) & 1)
                val_part = opposite_part(train_part)
                train_dirs_by_dataset[name] = (train_part, seed_splits[name][train_part]["input"].parent)
                val_infos.append((name, train_part, val_part, seed_splits[name][val_part]))

            train_dirs = [part_dir for _, part_dir in train_dirs_by_dataset.values()]
            combo_key = f"seed_{seed}_combo_{combo:03b}"

            for weight_mode in args.weight_modes:
                token_weights = ensure_weights(weight_mode, args.python, train_dirs, weights_dir, combo_key, args)
                for cluster_factor in args.cluster_factors:
                    factor_name = str(cluster_factor).replace(".", "p")
                    for name, train_part, val_part, val_info in val_infos:
                        pred_path = preds_dir / (
                            f"{combo_key}_{weight_mode}_{args.llm_mode}_factor_{factor_name}_{name}_{val_part}.csv"
                        )
                        runtime = run_predict(
                            args.python,
                            val_info["input"],
                            pred_path,
                            val_info["k"],
                            weight_mode,
                            token_weights,
                            cluster_factor,
                            args,
                        )
                        gold = read_gold(val_info["gold"])
                        pred = read_pred(pred_path)
                        ba, tpr, tnr = pairwise_scores(gold, pred)
                        results.append(
                            {
                                "seed": seed,
                                "combo": f"{combo:03b}",
                                "dataset": name,
                                "val_part": val_part,
                                "train_part_for_dataset": train_part,
                                "val_part_for_dataset": val_part,
                                "weight_mode": weight_mode,
                                "cluster_factor": cluster_factor,
                                "feature_level": args.feature_level,
                                "svd_dim": args.svd_dim,
                                "normalizer": args.normalizer,
                                "line_mode": args.line_mode,
                                "template_weighting": args.template_weighting,
                                "llm_mode": args.llm_mode,
                                "llm_weight": args.llm_weight,
                                "parser": args.parser,
                                "cluster": args.cluster,
                                "k": val_info["k"],
                                "num_cases": val_info["num_cases"],
                                "num_pred_clusters": len(set(pred)),
                                "BA": ba,
                                "TPR": tpr,
                                "TNR": tnr,
                                "runtime_sec": runtime,
                                "token_weights_path": str(token_weights) if token_weights else "",
                                "pred_path": str(pred_path),
                            }
                        )
                        print(
                            f"done seed={seed} combo={combo:03b} dataset={name} val={val_part} "
                            f"mode={weight_mode} cf={cluster_factor} BA={ba:.6f}",
                            file=sys.stderr,
                        )

    summary = summarize(results)
    write_results_csv(output_dir / "results.csv", results)
    write_summary_csv(output_dir / "summary.csv", summary)
    return results, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run half-split cross-scale validation.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--cluster-factors", nargs="+", type=float, default=[0.875])
    parser.add_argument(
        "--weight-modes",
        nargs="+",
        choices=("none", "repeat", "conservative", "blacklist"),
        default=["none", "repeat", "conservative", "blacklist"],
    )
    parser.add_argument("--parser", choices=("simple", "drain"), default="drain")
    parser.add_argument("--cluster", choices=("kmeans", "agglomerative", "hdbscan"), default="agglomerative")
    parser.add_argument("--feature-level", choices=("baseline", "structured"), default="baseline")
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--normalizer", choices=("v1", "semantic"), default="v1")
    parser.add_argument("--line-mode", choices=("default", "signal_window"), default="default")
    parser.add_argument("--template-weighting", choices=("none", "quality"), default="quality")
    parser.add_argument("--llm-mode", choices=("none", "embedding", "auto"), default="none")
    parser.add_argument("--llm-weight", type=float, default=4.0)
    parser.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    parser.add_argument("--train-min-df", type=int, default=2)
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--primary-boost", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/half_split_exp"))
    parser.add_argument("--keep-temp", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, summary = run_experiments(args)
    print_markdown_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
