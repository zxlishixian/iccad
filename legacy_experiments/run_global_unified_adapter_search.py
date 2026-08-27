#!/usr/bin/env python3
"""Global unified adapter search for regression failure bucketing.

Experimental only. This script may read gold/golden labels for training and
scoring, and may optionally read trace logs for trace feature sets. It never
changes the formal ``regr_fail_bucketing.py`` default predictor.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
import regr_fail_bucketing as rfb
import trace_anchor as ta
from run_experiments import pairwise_scores, read_gold
from run_official_directed_trace_eval import build_current_best_probability
from run_official_style_training_experiments import write_csv, write_pred

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS = {
    "fake_first": (Path("fake_dataset/first_batch_dataset"), "fake"),
    "fake_stage2": (Path("fake_dataset/stage2_dataset_working"), "fake"),
    "fake_stage3": (Path("fake_dataset/stage3_dataset_32bugs_640cases"), "fake"),
    "official_set1": (Path("test_case/problem/benchmark_set_1"), "official"),
    "official_set2": (Path("test_case/problem/benchmark_set_2"), "official"),
    "sanitized": (Path("fake_dataset/official_directed_stage1_sanitized_3bugs_85cases"), "sanitized"),
}


class TorchMLPWrapper:
    def __init__(self, model, scaler, device: str, batch_size: int):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.batch_size = int(batch_size)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        Xs = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(Xs), self.batch_size):
                xb = torch.from_numpy(Xs[start:start + self.batch_size]).to(self.device)
                logits = self.model(xb).squeeze(-1)
                probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        p = np.concatenate(probs).astype(np.float32) if probs else np.zeros(0, dtype=np.float32)
        return np.vstack([1.0 - p, p]).T


def train_model(X: np.ndarray, y: np.ndarray, model_type: str, seed: int, sample_weight: np.ndarray | None = None, args: argparse.Namespace | None = None) -> object:
    if model_type != "mlp":
        from run_official_style_training_experiments import train_model as sklearn_train_model
        return sklearn_train_model(X, y, model_type, seed, sample_weight=sample_weight)
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if args is None:
        raise ValueError("MLP training requires args")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    yv = y.astype(np.float32)
    if sample_weight is None:
        sw = np.ones(len(yv), dtype=np.float32)
    else:
        sw = sample_weight.astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(yv), torch.from_numpy(sw))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, generator=generator, num_workers=0)
    model = nn.Sequential(
        nn.Linear(Xs.shape[1], int(args.hidden_dim)),
        nn.LayerNorm(int(args.hidden_dim)),
        nn.ReLU(),
        nn.Dropout(float(args.dropout)),
        nn.Linear(int(args.hidden_dim), int(args.hidden_dim // 2)),
        nn.LayerNorm(int(args.hidden_dim // 2)),
        nn.ReLU(),
        nn.Dropout(float(args.dropout)),
        nn.Linear(int(args.hidden_dim // 2), int(args.hidden_dim // 4)),
        nn.ReLU(),
        nn.Linear(int(args.hidden_dim // 4), 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_loss = float("inf")
    stale = 0
    best_state = None
    for epoch in range(int(args.epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1)
            loss = (loss_fn(logits, yb) * wb).mean()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(yb)
            count += len(yb)
        avg = total / max(1, count)
        if avg + 1e-5 < best_loss:
            best_loss = avg
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(args.early_stop_patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TorchMLPWrapper(model, scaler, device, int(args.predict_batch_size))


def resolve(path: Path) -> Path:
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def logit(p: float) -> float:
    p = min(1.0 - 1e-5, max(1e-5, float(p)))
    return math.log(p / (1.0 - p))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


@dataclass
class DatasetArtifact:
    key: str
    source_type: str
    dataset: Path
    input_csv: Path
    gold_csv: Path
    k: int
    cases: list[str]
    gold: list[str]
    records: list[osf.OfficialCaseRecord]
    pairs: list[tuple[int, int]]
    labels: np.ndarray
    prob_base: np.ndarray
    base_cluster_labels: list[int]
    base_cluster_sizes: np.ndarray
    base_largest_cluster: int
    llm_features: list[plf.LLMCaseFeature] | None
    anchor_pair_matrix: np.ndarray | None
    base_note: str


def read_cases(input_csv: Path) -> list[str]:
    return osf.read_cases(input_csv)


def load_base_probability(args: argparse.Namespace, key: str, input_csv: Path) -> tuple[np.ndarray, str]:
    names = [f"{key}_B0_no_trace_best.npy", f"{input_csv.parent.name}_B0_no_trace_best.npy"]
    for directory in args.reuse_base_probs_dirs:
        if not directory:
            continue
        for name in names:
            path = Path(directory) / name
            if path.is_file():
                prob = np.load(path).astype(np.float32)
                np.fill_diagonal(prob, 1.0)
                return prob, f"reused_base_probability={path}"
    prob, _features, note, _runtime = build_current_best_probability(args, input_csv)
    if prob is None:
        raise RuntimeError(f"failed to build current-best probability for {input_csv}: {note}")
    np.fill_diagonal(prob, 1.0)
    return prob.astype(np.float32), note


def maybe_build_llm_features(args: argparse.Namespace, input_csv: Path, enabled: bool) -> list[plf.LLMCaseFeature] | None:
    if not enabled:
        return None
    llm_args = plf._make_llm_args(
        llm_mode="embedding",
        llm_doc_style="features",
        llm_cache_dir=args.llm_cache_dir,
        svd_dim=args.svd_dim,
        llm_dual=True,
    )
    features, _bundle = plf.build_llm_case_features(input_csv, svd_dim=args.svd_dim, llm_args=llm_args)
    return features


def maybe_build_anchor_matrix(input_csv: Path, pairs: Sequence[tuple[int, int]], enabled: bool, window_size: int) -> np.ndarray | None:
    if not enabled:
        return None
    feats, _debug = ta.build_anchor_trace_case_features([input_csv], window_size=window_size)
    return ta.build_anchor_trace_pair_feature_matrix(feats, pairs)


def build_artifact(args: argparse.Namespace, key: str, path: Path, source_type: str, need_llm: bool, need_trace: bool) -> DatasetArtifact:
    dataset = resolve(path)
    input_csv = dataset / "input.csv"
    gold_csv = osf.gold_path(dataset)
    prob, note = load_base_probability(args, key, input_csv)
    records = osf.build_case_records(key, input_csv, gold_csv)
    cases = read_cases(input_csv)
    gold = read_gold(gold_csv)
    pairs = all_pairs(len(records))
    labels = osf.pair_labels(records, pairs).astype(np.int8)
    k = len(set(gold))
    base_cluster_labels = plf.cluster_from_probability(prob, k)
    counts = Counter(base_cluster_labels)
    sizes = np.asarray([counts[x] for x in base_cluster_labels], dtype=np.float32)
    largest = max(counts.values()) if counts else 0
    llm_features = maybe_build_llm_features(args, input_csv, need_llm)
    anchor = maybe_build_anchor_matrix(input_csv, pairs, need_trace, args.trace_window_size)
    return DatasetArtifact(
        key=key,
        source_type=source_type,
        dataset=dataset,
        input_csv=input_csv,
        gold_csv=gold_csv,
        k=k,
        cases=cases,
        gold=gold,
        records=records,
        pairs=pairs,
        labels=labels,
        prob_base=prob,
        base_cluster_labels=base_cluster_labels,
        base_cluster_sizes=sizes,
        base_largest_cluster=largest,
        llm_features=llm_features,
        anchor_pair_matrix=anchor,
        base_note=note,
    )


def minimal_tag_matrix(art: DatasetArtifact) -> np.ndarray:
    rows: list[list[float]] = []
    for i, j in art.pairs:
        a = art.records[i]
        b = art.records[j]
        p = float(art.prob_base[i, j])
        tags_a = set(a.root_tags)
        tags_b = set(b.root_tags)
        inter = tags_a & tags_b
        union = tags_a | tags_b
        anchor_a = set(a.anchor.anchor_tags)
        anchor_b = set(b.anchor.anchor_tags)
        values = [
            p,
            logit(p),
            min(p, 1.0 - p),
            float(art.base_cluster_labels[i] == art.base_cluster_labels[j]),
            float(art.base_cluster_sizes[i] / max(1, len(art.records))),
            float(art.base_cluster_sizes[j] / max(1, len(art.records))),
            float(art.base_cluster_sizes[i] == art.base_largest_cluster and art.base_cluster_sizes[j] == art.base_largest_cluster),
            len(inter) / len(union) if union else 1.0,
            float(len(inter)),
            float(bool(a.root_tags and b.root_tags and sorted(a.root_tags)[0] == sorted(b.root_tags)[0])),
            float(len(tags_a ^ tags_b)),
            float(bool((tags_a & tags_b) & {"csr", "mcause_exception", "memory_fault", "illegal_instruction"})),
            float(("xprop_exception_state_failure" in tags_a or "memory_fault" in tags_a) and ("timeout" in tags_b or "core_status_timeout" in tags_b)),
            float(("xprop_exception_state_failure" in tags_b or "memory_fault" in tags_b) and ("timeout" in tags_a or "core_status_timeout" in tags_a)),
            osf.jaccard(anchor_a, anchor_b),
        ]
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def llm_scalar_matrix(art: DatasetArtifact) -> np.ndarray:
    if art.llm_features is None:
        return np.zeros((len(art.pairs), 5), dtype=np.float32)
    rows: list[list[float]] = []
    for i, j in art.pairs:
        a = art.llm_features[i]
        b = art.llm_features[j]
        det = cosine(a.det_vec, b.det_vec)
        feat = cosine(a.effective_llm_vec, b.effective_llm_vec)
        summ = cosine(a.effective_llm_summary_vec, b.effective_llm_summary_vec)
        rows.append([det, feat, summ, abs(feat - summ), abs(det - feat)])
    return np.asarray(rows, dtype=np.float32)


def feature_matrix(art: DatasetArtifact, feature_set: str) -> np.ndarray:
    blocks = [minimal_tag_matrix(art)]
    if feature_set in {"tags_structured", "tags_structured_llm", "tags_structured_llm_trace"}:
        blocks.append(osf.build_pair_feature_matrix(art.records, art.pairs, art.prob_base, include_graph=True, include_anchor=False))
    if feature_set in {"tags_structured_llm", "tags_structured_llm_trace"}:
        blocks.append(llm_scalar_matrix(art))
    if feature_set == "tags_structured_llm_trace":
        if art.anchor_pair_matrix is None:
            blocks.append(np.zeros((len(art.pairs), ta.anchor_trace_pair_feature_dim()), dtype=np.float32))
        else:
            blocks.append(art.anchor_pair_matrix.astype(np.float32, copy=False))
    return np.hstack(blocks).astype(np.float32, copy=False)


def sample_training_indices(art: DatasetArtifact, rng: random.Random, negative_ratio: float) -> np.ndarray:
    y = art.labels
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0:
        return np.asarray([], dtype=np.int64)
    n_neg = min(len(neg), int(math.ceil(len(pos) * negative_ratio)))
    hard = [idx for idx in neg if art.prob_base[art.pairs[idx][0], art.pairs[idx][1]] >= 0.60]
    rng.shuffle(hard)
    hard_take = hard[: min(len(hard), n_neg // 2)]
    remaining = [idx for idx in neg if idx not in set(hard_take)]
    rng.shuffle(remaining)
    chosen_neg = hard_take + remaining[: max(0, n_neg - len(hard_take))]
    idx_list = list(pos) + chosen_neg
    rng.shuffle(idx_list)
    return np.asarray(idx_list, dtype=np.int64)


def train_adapter(train_arts: Sequence[DatasetArtifact], matrices: dict[str, np.ndarray], feature_set: str, model_type: str, official_weight: float, negative_ratio: float, seed: int, args: argparse.Namespace) -> tuple[object, dict]:
    rng = random.Random(seed)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    stats = {"train_pairs": 0, "pos": 0, "neg": 0, "official_pairs": 0, "fake_pairs": 0, "sanitized_pairs": 0}
    for art in train_arts:
        idxs = sample_training_indices(art, rng, negative_ratio)
        if idxs.size == 0:
            continue
        X = matrices[art.key][idxs]
        y = art.labels[idxs].astype(int)
        weight = float(official_weight) if art.source_type == "official" else 1.0
        Xs.append(X)
        ys.append(y)
        ws.append(np.full(len(y), weight, dtype=np.float32))
        stats["train_pairs"] += len(y)
        stats["pos"] += int(np.sum(y == 1))
        stats["neg"] += int(np.sum(y == 0))
        stats[f"{art.source_type}_pairs"] += len(y)
    if not Xs:
        raise RuntimeError("no training pairs sampled")
    X_train = np.vstack(Xs)
    y_train = np.concatenate(ys)
    sample_weight = np.concatenate(ws)
    model = train_model(X_train, y_train, model_type, seed, sample_weight=sample_weight, args=args)
    return model, stats


def probability_from_model(model: object, art: DatasetArtifact, X: np.ndarray) -> np.ndarray:
    scores = model.predict_proba(X)[:, 1].astype(np.float32) if hasattr(model, "predict_proba") else model.predict(X).astype(np.float32)
    prob = np.eye(len(art.records), dtype=np.float32)
    for (i, j), score in zip(art.pairs, scores):
        prob[i, j] = prob[j, i] = float(score)
    return prob


def score_probability(art: DatasetArtifact, prob: np.ndarray, output_dir: Path, method: str) -> dict:
    labels = plf.cluster_from_probability(prob.astype(np.float32), art.k)
    pred_path = output_dir / "preds" / f"{art.key}_{method}.csv"
    prob_path = output_dir / "probs" / f"{art.key}_{method}.npy"
    pred = write_pred(pred_path, art.cases, labels)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prob_path, prob.astype(np.float32))
    ba, tpr, tnr = pairwise_scores(art.gold, pred)
    return {
        "holdout_dataset": art.key,
        "cases": len(art.cases),
        "k": art.k,
        "num_pred_clusters": len(set(pred)),
        "BA": ba,
        "TPR": tpr,
        "TNR": tnr,
        "pred_path": str(pred_path),
        "prob_path": str(prob_path),
    }


def source_means(rows: Sequence[dict], key: str) -> float:
    vals = [float(r["BA"]) for r in rows if r["holdout_dataset"] in key]
    return float(np.mean(vals)) if vals else 0.0


def summarize(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config"])].append(row)
    summaries: list[dict] = []
    fake_keys = {"fake_first", "fake_stage2", "fake_stage3"}
    official_keys = {"official_set1", "official_set2"}
    for config, rs in grouped.items():
        ba = [float(r["BA"]) for r in rs]
        summary = dict(rs[0])
        for noisy in ("holdout_dataset", "cases", "k", "num_pred_clusters", "BA", "TPR", "TNR", "pred_path", "prob_path", "runtime_sec"):
            summary.pop(noisy, None)
        summary.update({
            "config": config,
            "mean_BA": float(np.mean(ba)),
            "min_BA": float(np.min(ba)),
            "fake_mean_BA": float(np.mean([float(r["BA"]) for r in rs if r["holdout_dataset"] in fake_keys])) if any(r["holdout_dataset"] in fake_keys for r in rs) else 0.0,
            "official_mean_BA": float(np.mean([float(r["BA"]) for r in rs if r["holdout_dataset"] in official_keys])) if any(r["holdout_dataset"] in official_keys for r in rs) else 0.0,
            "sanitized_BA": next((float(r["BA"]) for r in rs if r["holdout_dataset"] == "sanitized"), 0.0),
            "set1_BA": next((float(r["BA"]) for r in rs if r["holdout_dataset"] == "official_set1"), 0.0),
            "set2_BA": next((float(r["BA"]) for r in rs if r["holdout_dataset"] == "official_set2"), 0.0),
            "mean_TPR": float(np.mean([float(r["TPR"]) for r in rs])),
            "mean_TNR": float(np.mean([float(r["TNR"]) for r in rs])),
            "num_folds": len(rs),
        })
        summary["rank_score"] = summary["min_BA"] * 1000.0 + summary["mean_BA"] * 10.0 + summary["official_mean_BA"]
        summaries.append(summary)
    summaries.sort(key=lambda r: (float(r["min_BA"]), float(r["mean_BA"]), float(r["official_mean_BA"])), reverse=True)
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank
    return summaries


def error_report(art: DatasetArtifact, rows: Sequence[dict], output_dir: Path) -> None:
    lines = [f"# Error Analysis: {art.key}", ""]
    for row in rows:
        if row["holdout_dataset"] != art.key:
            continue
        pred_path = Path(row["pred_path"])
        if not pred_path.is_file():
            continue
        with pred_path.open(newline="", encoding="utf-8-sig") as f:
            pred = [r["bucket"] for r in csv.DictReader(f)]
        lines.extend([f"## {row['config']}", f"BA={float(row['BA']):.6f} TPR={float(row['TPR']):.6f} TNR={float(row['TNR']):.6f}", "", "Cluster composition:"])
        by_bucket: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for case, gold, bucket in zip(art.cases, art.gold, pred):
            by_bucket[bucket].append((case, gold))
        for bucket, vals in sorted(by_bucket.items()):
            lines.append(f"- {bucket}: n={len(vals)} bugs={dict(Counter(g for _c, g in vals))} cases={','.join(c for c,_g in vals[:30])}")
        fn = Counter(); fp = Counter()
        for i in range(len(art.gold)):
            for j in range(i + 1, len(art.gold)):
                if art.gold[i] == art.gold[j] and pred[i] != pred[j]:
                    fn[art.gold[i]] += 1
                elif art.gold[i] != art.gold[j] and pred[i] == pred[j]:
                    fp[tuple(sorted((art.gold[i], art.gold[j])))] += 1
        lines.append("\nTop FN bugs:")
        lines.extend([f"- {k}: {v}" for k, v in fn.most_common(10)] or ["- none"])
        lines.append("\nTop FP bug pairs:")
        lines.extend([f"- {a}/{b}: {v}" for (a, b), v in fp.most_common(10)] or ["- none"])
        lines.append("")
    path = output_dir / "error_analysis" / f"{art.key}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global unified adapter search.")
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/global_unified_adapter_search"))
    p.add_argument("--feature-sets", nargs="+", default=["tags", "tags_structured", "tags_structured_llm", "tags_structured_llm_trace"])
    p.add_argument("--models", nargs="+", default=["logistic", "gbdt"])
    p.add_argument("--official-weights", nargs="+", type=float, default=[1, 3, 10, 30])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.25, 0.40, 0.50])
    p.add_argument("--random-states", nargs="+", type=int, default=[0])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--folds", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--negative-ratio", type=float, default=2.0)
    p.add_argument("--exclude-sanitized-from-training", action="store_true")
    p.add_argument("--reuse-base-probs-dirs", nargs="+", type=Path, default=[Path("/tmp/official_style_lodo_5datasets_tags/probs"), Path("/tmp/official_gated_adapter_sanitized_auto/probs")])
    p.add_argument("--trace-window-size", type=int, default=64)
    p.add_argument("--rich-model-root", type=Path, default=Path("/tmp/input_signal_5seed_top/models/llm_dual_struct_det_summary_dim64"))
    p.add_argument("--model-tag", default="llm_dual_struct_det_summary_dim64")
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--predict-batch-size", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temp", type=float, default=1.15)
    p.add_argument("--ensemble-temp", type=float, default=1.00)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    need_llm = any("llm" in fs for fs in args.feature_sets)
    need_trace = any("trace" in fs for fs in args.feature_sets)
    artifacts: dict[str, DatasetArtifact] = {}
    for key in args.folds:
        path, source = DATASETS[key]
        artifacts[key] = build_artifact(args, key, path, source, need_llm=need_llm, need_trace=need_trace)

    matrices: dict[str, dict[str, np.ndarray]] = {fs: {} for fs in args.feature_sets}
    for fs in args.feature_sets:
        for key, art in artifacts.items():
            matrices[fs][key] = feature_matrix(art, fs)

    rows: list[dict] = []
    pair_stats: list[dict] = []
    fields = ["holdout_dataset", "config", "feature_set", "model", "official_weight", "alpha", "random_state", "cases", "k", "num_pred_clusters", "BA", "TPR", "TNR", "runtime_sec", "pred_path", "prob_path"]
    partial_path = args.output_dir / "partial_results.csv"
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("w", newline="", encoding="utf-8") as partial_f:
        partial_writer = csv.DictWriter(partial_f, fieldnames=fields, extrasaction="ignore")
        partial_writer.writeheader()
        for fs in args.feature_sets:
            for model_type in args.models:
                for official_weight in args.official_weights:
                    for seed in args.random_states:
                        for holdout_key in args.folds:
                            holdout = artifacts[holdout_key]
                            train_arts = []
                            for key, art in artifacts.items():
                                if key == holdout_key:
                                    continue
                                if args.exclude_sanitized_from_training and art.source_type == "sanitized":
                                    continue
                                train_arts.append(art)
                            if not train_arts:
                                continue
                            t0 = time.perf_counter()
                            model, stats = train_adapter(train_arts, matrices[fs], fs, model_type, official_weight, args.negative_ratio, seed, args)
                            p_adapter = probability_from_model(model, holdout, matrices[fs][holdout_key])
                            train_runtime = time.perf_counter() - t0
                            for alpha in args.alphas:
                                config = f"fs={fs}|model={model_type}|ow={official_weight:g}|alpha={alpha:g}|seed={seed}"
                                p_final = (1.0 - float(alpha)) * holdout.prob_base + float(alpha) * p_adapter
                                np.fill_diagonal(p_final, 1.0)
                                row = score_probability(holdout, p_final, args.output_dir, config)
                                row.update({
                                    "config": config,
                                    "feature_set": fs,
                                    "model": model_type,
                                    "official_weight": official_weight,
                                    "alpha": alpha,
                                    "random_state": seed,
                                    "runtime_sec": train_runtime,
                                })
                                rows.append(row)
                                partial_writer.writerow(row)
                                partial_f.flush()
                                stats_row = dict(stats)
                                stats_row.update({"config": config, "holdout_dataset": holdout_key, "feature_set": fs, "model": model_type, "official_weight": official_weight, "alpha": alpha, "random_state": seed})
                                pair_stats.append(stats_row)
                                print(f"[done] {config} holdout={holdout_key} BA={row['BA']:.4f}", flush=True)

    write_csv(args.output_dir / "results.csv", rows, fields)
    summaries = summarize(rows)
    summary_fields = ["rank", "config", "feature_set", "model", "official_weight", "alpha", "random_state", "mean_BA", "min_BA", "fake_mean_BA", "official_mean_BA", "sanitized_BA", "set1_BA", "set2_BA", "mean_TPR", "mean_TNR", "num_folds", "rank_score"]
    write_csv(args.output_dir / "summary.csv", summaries, summary_fields)
    write_csv(args.output_dir / "ranked_configs.csv", summaries, summary_fields)
    write_csv(args.output_dir / "pair_stats.csv", pair_stats, ["config", "holdout_dataset", "feature_set", "model", "official_weight", "alpha", "random_state", "train_pairs", "pos", "neg", "official_pairs", "fake_pairs", "sanitized_pairs"])
    write_csv(args.output_dir / "feature_debug.csv", [{"dataset": k, "feature_set": fs, "rows": matrices[fs][k].shape[0], "cols": matrices[fs][k].shape[1]} for fs in args.feature_sets for k in artifacts], ["dataset", "feature_set", "rows", "cols"])
    for art in artifacts.values():
        error_report(art, rows[: max(0, min(len(rows), 200))], args.output_dir)

    print("\n| rank | config | mean_BA | min_BA | fake_mean | official_mean | sanitized | set1 | set2 |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries[:20]:
        print(f"| {row['rank']} | {row['config']} | {float(row['mean_BA']):.6f} | {float(row['min_BA']):.6f} | {float(row['fake_mean_BA']):.6f} | {float(row['official_mean_BA']):.6f} | {float(row['sanitized_BA']):.6f} | {float(row['set1_BA']):.6f} | {float(row['set2_BA']):.6f} |")
    print(f"\nResults: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
