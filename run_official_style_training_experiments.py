#!/usr/bin/env python3
"""Official-style root-cause training experiments.

Experimental only. This script may read official ``golden.csv`` labels for
leave-one-benchmark-out research. It does not alter the formal prediction path.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
import trace_anchor as ta
from run_experiments import pairwise_scores, read_gold
from run_official_directed_trace_eval import build_current_best_probability


PROJECT_ROOT = Path(__file__).resolve().parent


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_pred(path: Path, cases: Sequence[str], labels: Sequence[int]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = [f"bucket_{int(x):03d}" for x in labels]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "bucket"])
        writer.writerows(zip(cases, pred))
    return pred


def prob_from_pair_scores(n: int, pairs: Sequence[tuple[int, int]], scores: np.ndarray) -> np.ndarray:
    prob = np.eye(n, dtype=np.float32)
    for (i, j), p in zip(pairs, scores):
        prob[i, j] = prob[j, i] = float(p)
    return prob


def train_model(X: np.ndarray, y: np.ndarray, model_type: str, seed: int) -> object:
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.5,
                random_state=seed,
            ),
        )
        model.fit(X, y)
        return model
    if model_type == "gbdt":
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.9,
            random_state=seed,
        )
        pos = max(1.0, float(np.sum(y == 1)))
        neg = max(1.0, float(np.sum(y == 0)))
        weights = np.where(y == 1, (pos + neg) / (2.0 * pos), (pos + neg) / (2.0 * neg))
        model.fit(X, y, sample_weight=weights)
        return model
    raise ValueError(f"unknown model_type: {model_type}")


def predict_scores(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    return model.predict(X).astype(np.float32)


def dataset_artifacts(dataset: Path, args: argparse.Namespace) -> dict:
    dataset = dataset.resolve()
    name = dataset.name
    input_csv = dataset / "input.csv"
    gold_csv = osf.gold_path(dataset)
    prob, _features, note, runtime = build_current_best_probability(args, input_csv)
    if prob is None:
        raise RuntimeError(f"failed to build current-best probability for {dataset}: {note}")
    records = osf.build_case_records(name, input_csv, gold_csv)
    cases = osf.read_cases(input_csv)
    pairs = osf.all_pairs(len(records))
    labels = osf.pair_labels(records, pairs)
    anchor_by_window: dict[int, np.ndarray] = {}
    for window in args.window_sizes:
        feats, _debug = ta.build_anchor_trace_case_features([input_csv], window_size=window)
        anchor_by_window[int(window)] = ta.build_anchor_trace_pair_feature_matrix(feats, pairs)
    return {
        "name": name,
        "dataset": dataset,
        "input_csv": input_csv,
        "gold_csv": gold_csv,
        "k": len(set(read_gold(gold_csv))),
        "cases": cases,
        "records": records,
        "pairs": pairs,
        "labels": labels,
        "prob_base": prob,
        "base_note": note,
        "base_runtime": runtime,
        "anchor_by_window": anchor_by_window,
    }


def score_probability(art: dict, prob: np.ndarray, method: str, output_dir: Path, notes: str, runtime: float) -> dict:
    labels = plf.cluster_from_probability(prob.astype(np.float32), int(art["k"]))
    pred_path = output_dir / "preds" / f"{art['name']}_{method}.csv"
    prob_path = output_dir / "probs" / f"{art['name']}_{method}.npy"
    pred = write_pred(pred_path, art["cases"], labels)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prob_path, prob.astype(np.float32))
    ba, tpr, tnr = pairwise_scores(read_gold(art["gold_csv"]), pred)
    return {
        "train_dataset": "",
        "test_dataset": art["name"],
        "method": method,
        "model_type": "",
        "window_size": "",
        "k": art["k"],
        "cases": len(art["cases"]),
        "num_pred_clusters": len(set(pred)),
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "runtime_sec": runtime,
        "pred_path": str(pred_path),
        "prob_path": str(prob_path),
        "notes": notes,
    }


def run_train_eval(
    train_art: dict,
    test_art: dict,
    variant: str,
    model_type: str,
    window_size: int,
    output_dir: Path,
    seed: int,
    blend_alpha: float | None = None,
) -> dict:
    include_graph = variant in {"tags_graph", "tags_graph_anchor"}
    include_anchor = variant == "tags_graph_anchor"
    X_train = osf.build_pair_feature_matrix(
        train_art["records"],
        train_art["pairs"],
        train_art["prob_base"],
        include_graph=include_graph,
        include_anchor=include_anchor,
        anchor_pair_matrix=train_art["anchor_by_window"].get(int(window_size)) if include_anchor else None,
    )
    y_train = train_art["labels"].astype(int)
    X_test = osf.build_pair_feature_matrix(
        test_art["records"],
        test_art["pairs"],
        test_art["prob_base"],
        include_graph=include_graph,
        include_anchor=include_anchor,
        anchor_pair_matrix=test_art["anchor_by_window"].get(int(window_size)) if include_anchor else None,
    )
    t0 = time.perf_counter()
    model = train_model(X_train, y_train, model_type, seed)
    pair_scores = predict_scores(model, X_test)
    prob_model = prob_from_pair_scores(len(test_art["records"]), test_art["pairs"], pair_scores)
    if blend_alpha is not None:
        prob = float(blend_alpha) * prob_model + (1.0 - float(blend_alpha)) * test_art["prob_base"]
        blend_suffix = f"_blend{blend_alpha:.2f}"
    else:
        prob = prob_model
        blend_suffix = ""
    np.fill_diagonal(prob, 1.0)
    runtime = time.perf_counter() - t0
    method = f"official_style_{variant}_{model_type}"
    if include_anchor:
        method += f"_w{window_size}"
    method += blend_suffix
    row = score_probability(
        test_art,
        prob,
        method,
        output_dir,
        notes=(
            f"train={train_art['name']}; variant={variant}; model={model_type}; "
            f"train_pairs={len(y_train)} pos={int(y_train.sum())} neg={int((1-y_train).sum())}"
        ),
        runtime=runtime,
    )
    row.update({
        "train_dataset": train_art["name"],
        "model_type": model_type,
        "window_size": window_size if include_anchor else "",
    })
    return row


def build_error_report(artifacts: Sequence[dict], rows: Sequence[dict], output_dir: Path) -> None:
    art_by_name = {art["name"]: art for art in artifacts}
    pred_rows_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pred_rows_by_dataset[str(row["test_dataset"])].append(row)
    for dataset, ds_rows in pred_rows_by_dataset.items():
        art = art_by_name[dataset]
        gold = read_gold(art["gold_csv"])
        cases = art["cases"]
        lines = [f"# Official-Style Error Analysis: {dataset}", ""]
        for row in ds_rows:
            pred_path = Path(str(row.get("pred_path", "")))
            if not pred_path.is_file():
                continue
            with pred_path.open(newline="", encoding="utf-8-sig") as f:
                pred = [r["bucket"] for r in csv.DictReader(f)]
            ba, tpr, tnr = pairwise_scores(gold, pred)
            lines.extend([
                f"## {row['method']}",
                "",
                f"train={row.get('train_dataset', '')} BA={ba:.6f} TPR={tpr:.6f} TNR={tnr:.6f}",
                "",
                "Cluster composition:",
            ])
            by_bucket: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for case, bug, bucket in zip(cases, gold, pred):
                by_bucket[bucket].append((case, bug))
            for bucket, values in sorted(by_bucket.items()):
                counts = Counter(bug for _case, bug in values)
                case_list = ",".join(case for case, _bug in values)
                lines.append(f"- {bucket}: n={len(values)} bugs={dict(counts)} cases={case_list}")
            fn = Counter()
            fp = Counter()
            for i in range(len(gold)):
                for j in range(i + 1, len(gold)):
                    if gold[i] == gold[j] and pred[i] != pred[j]:
                        fn[gold[i]] += 1
                    elif gold[i] != gold[j] and pred[i] == pred[j]:
                        fp[tuple(sorted((gold[i], gold[j])))] += 1
            lines.append("\nTop FN bugs:")
            lines.extend([f"- {bug}: {cnt}" for bug, cnt in fn.most_common(8)] or ["- none"])
            lines.append("\nTop FP bug pairs:")
            lines.extend([f"- {a} / {b}: {cnt}" for (a, b), cnt in fp.most_common(8)] or ["- none"])
            lines.append("")
        (output_dir / f"error_analysis_{dataset}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official-style root-cause training experiments.")
    p.add_argument("--benchmarks", nargs="+", type=Path, default=[
        Path("test_case/problem/benchmark_set_1"),
        Path("test_case/problem/benchmark_set_2"),
    ])
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/official_style_training_exp"))
    p.add_argument("--variants", nargs="+", default=["tags", "tags_graph", "tags_graph_anchor"])
    p.add_argument("--model-types", nargs="+", default=["logistic", "gbdt"])
    p.add_argument("--window-sizes", nargs="+", type=int, default=[64])
    p.add_argument("--blend-alphas", nargs="+", type=float, default=[0.25, 0.50])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--rich-model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temp", type=float, default=1.15)
    p.add_argument("--ensemble-temp", type=float, default=1.00)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        dataset_artifacts((PROJECT_ROOT / ds).resolve() if not ds.is_absolute() else ds, args)
        for ds in args.benchmarks
    ]
    rows: list[dict] = []
    for art in artifacts:
        rows.append(score_probability(
            art,
            art["prob_base"],
            "B0_no_trace_best",
            args.output_dir,
            notes=art["base_note"],
            runtime=art["base_runtime"],
        ))
    if len(artifacts) >= 2:
        for train_art in artifacts:
            for test_art in artifacts:
                if train_art["name"] == test_art["name"]:
                    continue
                for variant in args.variants:
                    for model_type in args.model_types:
                        windows = args.window_sizes if variant == "tags_graph_anchor" else [0]
                        for window in windows:
                            rows.append(run_train_eval(train_art, test_art, variant, model_type, int(window), args.output_dir, args.seed, blend_alpha=None))
                            for blend_alpha in args.blend_alphas:
                                rows.append(run_train_eval(train_art, test_art, variant, model_type, int(window), args.output_dir, args.seed, blend_alpha=float(blend_alpha)))
    fields = [
        "train_dataset", "test_dataset", "method", "model_type", "window_size",
        "k", "cases", "num_pred_clusters", "BA", "TPR", "TNR", "runtime_sec",
        "pred_path", "prob_path", "notes",
    ]
    write_csv(args.output_dir / "results.csv", rows, fields)
    write_csv(args.output_dir / "summary.csv", rows, fields)
    build_error_report(artifacts, rows, args.output_dir)

    print("\n| train | test | method | BA | TPR | TNR | clusters |")
    print("|---|---|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.get('train_dataset','')} | {row['test_dataset']} | {row['method']} | "
            f"{float(row['BA']):.6f} | {float(row['TPR']):.6f} | {float(row['TNR']):.6f} | "
            f"{row['num_pred_clusters']} |"
        )
    print(f"\nResults: {args.output_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
