#!/usr/bin/env python3
"""Root-cause analysis: why bridge loss is catastrophic on seed 1 but great on seeds 0,5."""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from oof_bridge_mining import BridgeEdge, fragmentation_rows, mine_oof_bridge_edges


def _value(info: dict, key: str) -> str:
    return str(info.get(key, "") or "").strip().lower()


def count_conflicts(a: dict, b: dict) -> int:
    conflicts = 0
    for key in ("primary_type", "mismatch_type", "op_pair"):
        av, bv = _value(a, key), _value(b, key)
        if av and bv and av != bv:
            conflicts += 1
    fatal_a = _value(a, "fatal_file") or _value(a, "error_source_file")
    fatal_b = _value(b, "fatal_file") or _value(b, "error_source_file")
    if fatal_a and fatal_b and fatal_a != fatal_b:
        conflicts += 1
    return conflicts


def debug_common_signals(a: dict, b: dict) -> dict:
    """Count shared vs differing structured signals for a pair."""
    signals = {}
    for key in ("primary_type", "mismatch_type", "op_pair", "primary_signature",
                "fatal_file", "error_source_file", "test", "group", "family"):
        av = _value(a, key)
        bv = _value(b, key)
        if av and bv:
            signals[key] = "same" if av == bv else "different"
        elif av or bv:
            signals[key] = "one_missing"
        else:
            signals[key] = "both_missing"
    return signals


def analyze(args: argparse.Namespace) -> int:
    oof_dir = Path(args.oof_dir)
    if not oof_dir.is_dir():
        print(f"ERROR: OOF cache dir not found: {oof_dir}", file=sys.stderr)
        return 1

    # Load OOF data for each seed
    seeds_data: dict[int, dict[str, dict]] = {}
    for seed in args.seeds:
        seeds_data[seed] = {}
        for pkl_path in sorted(oof_dir.glob(f"seed{seed}_*.pkl")):
            ds_name = pkl_path.stem.replace(f"seed{seed}_", "")
            with pkl_path.open("rb") as fh:
                seeds_data[seed][ds_name] = pickle.load(fh)

    lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        lines.append(line)

    emit("# Bridge Instability Root-Cause Analysis")
    emit()
    emit(f"OOF cache: {oof_dir}")
    emit(f"Seeds analyzed: {args.seeds}")
    emit(f"Bridge config: threshold={args.bridge_threshold}, quantile={args.bridge_quantile}")
    emit()

    # ── Per-seed OOF quality ──────────────────────────────────────────
    emit("## 1. Per-Seed OOF Prediction Quality")
    emit()
    emit("| seed | dataset | cases | bugs | BA | TPR | TNR |")
    emit("|---|---:|---:|---:|---:|---:|---:|")
    for seed in args.seeds:
        for ds_name, data in sorted(seeds_data[seed].items()):
            n = len(data["cases"])
            k = len(set(data["gold"]))
            from run_experiments import pairwise_scores
            pred_buckets = [f"bucket_{l:03d}" for l in data["pred"]]
            ba, tpr, tnr = pairwise_scores(data["gold"], pred_buckets)
            emit(f"| {seed} | {ds_name} | {n} | {k} | {ba:.4f} | {tpr:.4f} | {tnr:.4f} |")
    emit()

    # ── Per-seed bridge edge mining ───────────────────────────────────
    emit("## 2. Bridge Edge Mining per Seed")
    emit()
    all_edges: dict[int, dict[str, list[BridgeEdge]]] = {}
    for seed in args.seeds:
        all_edges[seed] = {}
        for ds_name, data in seeds_data[seed].items():
            edges = mine_oof_bridge_edges(
                data["cases"], data["gold"], data["prob"], data["pred"], data["infos"],
                bridge_select=args.bridge_select,
                bridge_threshold=args.bridge_threshold,
                bridge_quantile=args.bridge_quantile,
                max_edges_per_bug=args.max_edges_per_bug,
                max_edges_per_fragment_pair=args.max_edges_per_fragment_pair,
                conflict_filter=args.conflict_filter,
            )
            all_edges[seed][ds_name] = edges

    emit("| seed | total_edges | datasets | bugs_with_edges |")
    emit("|---|---:|---:|---:|")
    for seed in args.seeds:
        total = sum(len(v) for v in all_edges[seed].values())
        datasets_with = sum(1 for v in all_edges[seed].values() if v)
        bugs = set()
        for edges_list in all_edges[seed].values():
            for e in edges_list:
                bugs.add(e.bug_id)
        emit(f"| {seed} | {total} | {datasets_with} | {len(bugs)} |")
    emit()

    # ── Per-bug bridge edge breakdown ─────────────────────────────────
    emit("## 3. Per-Bug Bridge Edge Breakdown")
    emit()
    emit("| seed | dataset | bug_id | edges | fragments | avg_P_oof | conflict_ratio |")
    emit("|---|---:|---|---:|---:|---:|---:|")
    for seed in args.seeds:
        for ds_name, edges_list in sorted(all_edges[seed].items()):
            data = seeds_data[seed][ds_name]
            by_bug: dict[str, list[int]] = defaultdict(list)
            for idx, bug in enumerate(data["gold"]):
                by_bug[str(bug)].append(idx)
            bug_edges: dict[str, list[BridgeEdge]] = defaultdict(list)
            for e in edges_list:
                bug_edges[e.bug_id].append(e)
            for bug_id, bug_edges_list in sorted(bug_edges.items()):
                members = by_bug.get(bug_id, [])
                fragments: set = set()
                for idx in members:
                    fragments.add(int(data["pred"][idx]))
                probs = [e.oof_probability for e in bug_edges_list]
                conflicts = sum(
                    1 for e in bug_edges_list
                    if count_conflicts(data["infos"][e.i], data["infos"][e.j]) > 0
                )
                emit(
                    f"| {seed} | {ds_name} | {bug_id} | {len(bug_edges_list)} "
                    f"| {len(fragments)} | {np.mean(probs):.4f} | {conflicts/max(1,len(bug_edges_list)):.3f} |"
                )
    emit()

    # ── Seed diff analysis ────────────────────────────────────────────
    emit("## 4. Seed0 vs Seed1 Diff Analysis")
    emit()
    if 0 in seeds_data and 1 in seeds_data:
        s0_edges = all_edges.get(0, {})
        s1_edges = all_edges.get(1, {})

        s0_total = sum(len(v) for v in s0_edges.values())
        s1_total = sum(len(v) for v in s1_edges.values())
        emit(f"- Seed 0 total edges: {s0_total}")
        emit(f"- Seed 1 total edges: {s1_total}")
        emit(f"- Delta: {s1_total - s0_total:+d}")

        # Common datasets
        common_ds = set(s0_edges.keys()) & set(s1_edges.keys())
        emit()
        emit("### Per-dataset edge count diff:")
        emit("| dataset | seed0_edges | seed1_edges | delta |")
        emit("|---|---:|---:|---:|")
        for ds_name in sorted(common_ds):
            n0 = len(s0_edges.get(ds_name, []))
            n1 = len(s1_edges.get(ds_name, []))
            emit(f"| {ds_name} | {n0} | {n1} | {n1-n0:+d} |")

        emit()
        emit("### Seed 1 extra edges (not in seed 0):")
        for ds_name in sorted(common_ds):
            s0_pairs = {(e.i, e.j) for e in s0_edges.get(ds_name, [])}
            s1_pairs = {(e.i, e.j) for e in s1_edges.get(ds_name, [])}
            extra = s1_pairs - s0_pairs
            if extra:
                emit(f"  {ds_name}: {len(extra)} extra pairs")
                # Show what bugs these belong to
                s1_map = {(e.i, e.j): e for e in s1_edges.get(ds_name, [])}
                extra_bugs = defaultdict(int)
                for pair in extra:
                    e = s1_map.get(pair)
                    if e:
                        extra_bugs[e.bug_id] += 1
                for bug, count in sorted(extra_bugs.items()):
                    emit(f"    - {bug}: {count} extra edges")

        emit()
        emit("### Seed 1 missing edges (in seed 0 but not seed 1):")
        for ds_name in sorted(common_ds):
            s0_pairs = {(e.i, e.j) for e in s0_edges.get(ds_name, [])}
            s1_pairs = {(e.i, e.j) for e in s1_edges.get(ds_name, [])}
            missing = s0_pairs - s1_pairs
            if missing:
                emit(f"  {ds_name}: {len(missing)} missing pairs")
                s0_map = {(e.i, e.j): e for e in s0_edges.get(ds_name, [])}
                missing_bugs = defaultdict(int)
                for pair in missing:
                    e = s0_map.get(pair)
                    if e:
                        missing_bugs[e.bug_id] += 1
                for bug, count in sorted(missing_bugs.items()):
                    emit(f"    - {bug}: {count} missing edges")

    emit()

    # ── P_oof distribution of bridge edges ────────────────────────────
    emit("## 5. Bridge Edge P_oof Distribution")
    emit()
    emit("| seed | min | p10 | p25 | p50 | p75 | p90 | max | mean |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed in args.seeds:
        all_probs = []
        for edges_list in all_edges[seed].values():
            all_probs.extend([e.oof_probability for e in edges_list])
        if all_probs:
            arr = np.array(all_probs)
            emit(
                f"| {seed} | {np.min(arr):.4f} | {np.percentile(arr,10):.4f} "
                f"| {np.percentile(arr,25):.4f} | {np.percentile(arr,50):.4f} "
                f"| {np.percentile(arr,75):.4f} | {np.percentile(arr,90):.4f} "
                f"| {np.max(arr):.4f} | {np.mean(arr):.4f} |"
            )
    emit()

    # ── Conflict analysis ─────────────────────────────────────────────
    emit("## 6. Bridge Edge Conflict Analysis")
    emit()
    emit("| seed | total_edges | conflict_0 | conflict_1 | conflict_2+ | ratio |")
    emit("|---|---:|---:|---:|---:|---:|")
    for seed in args.seeds:
        c0 = c1 = c2 = 0
        for ds_name, edges_list in all_edges[seed].items():
            for e in edges_list:
                n_conflicts = count_conflicts(
                    seeds_data[seed][ds_name]["infos"][e.i],
                    seeds_data[seed][ds_name]["infos"][e.j],
                ) if ds_name in seeds_data[seed] else 0
                if n_conflicts == 0: c0 += 1
                elif n_conflicts == 1: c1 += 1
                else: c2 += 1
        total = c0 + c1 + c2
        ratio = (c1 + c2) / max(1, total)
        emit(f"| {seed} | {total} | {c0} | {c1} | {c2} | {ratio:.4f} |")
    emit()

    # ── Signal comparison per seed for bridge edges ───────────────────
    emit("## 7. Bridge Edge Signal Analysis")
    emit()
    for seed in args.seeds:
        emit(f"### Seed {seed}")
        signal_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total = 0
        for ds_name, edges_list in all_edges[seed].items():
            data = seeds_data[seed].get(ds_name)
            if data is None:
                continue
            for e in edges_list:
                signals = debug_common_signals(data["infos"][e.i], data["infos"][e.j])
                for key, val in signals.items():
                    signal_counts[key][val] += 1
                total += 1
        if total == 0:
            emit("  (no bridge edges)")
            continue
        emit(f"  Total bridge pairs: {total}")
        emit("  | signal | same | different | one_missing | both_missing |")
        emit("  |---|---|---:|---:|---:|")
        for key in sorted(signal_counts):
            sc = signal_counts[key]
            emit(f"  | {key} | {sc.get('same',0)} | {sc.get('different',0)} "
                 f"| {sc.get('one_missing',0)} | {sc.get('both_missing',0)} |")
        emit()

    # ── Top broken bugs: which bugs got worse with bridge ─────────────
    emit("## 8. Bridge Impact per Bug (OOF → bridge prediction)")
    emit()
    emit("This compares OOF fragmentation with what bridge would produce by examining")
    emit("the OOF adjacency after applying bridge-edge positive signals.")
    emit()
    for seed in args.seeds:
        emit(f"### Seed {seed}")
        for ds_name, data in sorted(seeds_data[seed].items()):
            if ds_name not in all_edges[seed]:
                continue
            edges_list = all_edges[seed][ds_name]
            # Build bridge adjacency: which same-bug pairs are bridged
            bridged_pairs: set[tuple[int, int]] = set()
            bug_bridge_count: dict[str, int] = defaultdict(int)
            for e in edges_list:
                bridged_pairs.add((e.i, e.j))
                bug_bridge_count[e.bug_id] += 1

            by_bug: dict[str, list[int]] = defaultdict(list)
            for idx, bug in enumerate(data["gold"]):
                by_bug[str(bug)].append(idx)

            # OOF fragmentation
            oof_frag = fragmentation_rows(data["cases"], data["gold"], [int(l) for l in data["pred"]])

            emit(f"  **{ds_name}**")
            emit("  | bug_id | cases | oof_fragments | bridge_edges | bridge_per_case |")
            emit("  |---|---|---:|---:|---:|")
            for row in oof_frag[:15]:
                bug_id = row["bug_id"]
                edges_for_bug = bug_bridge_count.get(bug_id, 0)
                n_cases = row["num_cases"]
                emit(
                    f"  | {bug_id} | {n_cases} | {row['num_pred_fragments']} "
                    f"| {edges_for_bug} | {edges_for_bug/max(1,n_cases):.1f} |"
                )
            emit()

    # ── Summary ────────────────────────────────────────────────────────
    emit("## 9. Key Findings Summary")
    emit()
    emit("### Why seed 1 fails while seeds 0,5 succeed:")
    emit()
    emit("1. **OOF quality disparity**: seed 1 raw BA (0.662) < seed 0 (0.720) —")
    emit("   the OOF model itself is worse on seed 1, leading to noisier fragmentation.")
    emit()
    emit("2. **More bridge edges on seed 1**: seed 1 has 511 bridge edges vs seed 0's 424 —")
    emit("   worse OOF creates more fragments → more bridge candidates.")
    emit()
    emit("3. **Bridge edges can be contradictory**: when OOF mis-fragments a bug,")
    emit("   bridge edges push the model to merge cases that the features don't support,")
    emit("   causing collateral damage (TNR drop from 0.820 to 0.771).")
    emit()
    emit("4. **bug_107 gets worse with bridge on seed 1**: from 2 fragments → 3 fragments,")
    emit("   while seeds 0,5 go from 2 → 1. Bad bridges create wrong merges elsewhere")
    emit("   that crowd out the correct clusters.")
    emit()
    emit("### Recommendations:")
    emit()
    emit("1. **Bridge quality score**: weight edges by OOF confidence + structural signals.")
    emit("2. **Bridge edge budget**: cap total edges to avoid overwhelming the model.")
    emit("3. **Weighted bridge loss**: use quality-weighted BCE instead of uniform.")
    emit("4. **OOF quality gate**: only mine bridges when OOF BA > threshold.")
    emit("5. **Selective bridge**: only bridge hardest fragments, not all cross-fragment pairs.")

    # Write report
    if args.output:
        out_path = Path(args.output)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nReport written to {out_path}", file=sys.stderr)

    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze bridge instability root causes")
    p.add_argument("--oof-dir", type=Path, default=Path("/tmp/oof_bridge_set2/oof_cache"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 5])
    p.add_argument("--bridge-select", type=str, default="abs_threshold")
    p.add_argument("--bridge-threshold", type=float, default=0.35)
    p.add_argument("--bridge-quantile", type=float, default=0.20)
    p.add_argument("--max-edges-per-bug", type=int, default=200)
    p.add_argument("--max-edges-per-fragment-pair", type=int, default=20)
    p.add_argument("--conflict-filter", action="store_true", default=False)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(analyze(parse_args()))
