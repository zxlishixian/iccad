#!/usr/bin/env python3
"""Experimental pairwise same-bug features with LLM embedding augmentation.

Extends pairwise_features.py with:
- LLM embedding vectors attached to each case
- Pairwise features combining deterministic + LLM cosine similarities
- Three model backends: logistic regression, gradient boosting, small MLP
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import pairwise_features as pf
import regr_fail_bucketing as rfb


@dataclass
class LLMCaseFeature:
    case_id: str
    det_vec: np.ndarray
    llm_vec: np.ndarray
    tokens: list[str]
    token_set: set[str]
    primary_tokens: set[str]
    sim_tokens: set[str]
    regr_tokens: set[str]
    info: dict

    @property
    def has_llm(self) -> bool:
        return self.llm_vec.size > 0


def _make_llm_args(
    llm_mode: str = "embedding",
    llm_fusion: str = "concat",
    llm_weight: float = 4.0,
    llm_alpha: float = 0.75,
    llm_doc_style: str = "features",
    llm_doc_max_features: int = 80,
    llm_cache_dir: Path = Path("/tmp/regr_fail_llm_cache"),
    llm_batch_size: int = 64,
    llm_timeout_sec: float = 20.0,
    svd_dim: int = 64,
) -> argparse.Namespace:
    return argparse.Namespace(
        llm_mode=llm_mode,
        llm_fusion=llm_fusion,
        llm_weight=llm_weight,
        llm_alpha=llm_alpha,
        llm_doc_style=llm_doc_style,
        llm_doc_max_features=llm_doc_max_features,
        llm_cache_dir=Path(llm_cache_dir),
        llm_batch_size=llm_batch_size,
        llm_timeout_sec=llm_timeout_sec,
        svd_dim=svd_dim,
        parser="drain",
    )


def fetch_llm_embeddings_for_counters(
    feature_counters: Sequence[Counter],
    llm_args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    docs = rfb.build_llm_case_documents(
        feature_counters,
        max_features=llm_args.llm_doc_max_features,
        doc_style=llm_args.llm_doc_style,
    )
    embeddings, model_name = rfb.fetch_llm_embeddings(docs, llm_args)
    llm_mat = np.asarray(embeddings, dtype="float32")
    if llm_mat.ndim != 2 or llm_mat.shape[0] != len(feature_counters):
        raise RuntimeError(f"unexpected embedding matrix shape: {llm_mat.shape}")
    from sklearn.preprocessing import Normalizer
    llm_mat = Normalizer(copy=False).fit_transform(llm_mat)
    return llm_mat, model_name


def build_llm_case_features(
    input_csv: str | Path,
    parser: str = "drain",
    svd_dim: int = 64,
    llm_args: argparse.Namespace | None = None,
) -> tuple[list[LLMCaseFeature], pf.VectorizerBundle]:
    """Build deterministic features (via pairwise_features) and attach LLM embeddings."""
    det_features, bundle = pf.build_case_features(
        Path(input_csv).resolve(),
        parser=parser,
        svd_dim=svd_dim,
    )

    llm_vecs: list[np.ndarray] = []
    model_name = ""
    if llm_args is not None and rfb.load_llm_embedding_config() is not None:
        # Re-extract feature counters to build LLM documents
        from sklearn.feature_extraction.text import TfidfVectorizer
        _case_ids, base_features, normalized_lines, _infos = pf._collect_case_inputs(
            Path(input_csv).resolve(), parser
        )
        parser_args = pf._make_parser_args(parser)
        feature_counters, _template_count = rfb.build_feature_counters(
            parser_args,
            base_features,
            normalized_lines,
            token_weights=None,
            token_weight_mode="none",
        )
        try:
            llm_mat, model_name = fetch_llm_embeddings_for_counters(feature_counters, llm_args)
            for i in range(len(det_features)):
                llm_vecs.append(llm_mat[i].astype(np.float32, copy=False))
            print(
                f"[llm_features] model={model_name} embedding_dim={llm_mat.shape[1]} "
                f"docs={len(feature_counters)} doc_style={llm_args.llm_doc_style}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[llm_features] LLM fetch failed ({exc}); using zero llm_vec", file=sys.stderr)
            llm_vecs = [np.zeros(0, dtype=np.float32) for _ in det_features]
    else:
        print("[llm_features] LLM disabled; using zero llm_vec", file=sys.stderr)
        llm_vecs = [np.zeros(0, dtype=np.float32) for _ in det_features]

    result: list[LLMCaseFeature] = []
    for det_feat, llm_vec in zip(det_features, llm_vecs):
        result.append(
            LLMCaseFeature(
                case_id=det_feat.case_id,
                det_vec=det_feat.dense_vec.astype(np.float32, copy=False),
                llm_vec=llm_vec,
                tokens=list(det_feat.tokens),
                token_set=det_feat.token_set,
                primary_tokens=det_feat.primary_tokens,
                sim_tokens=det_feat.sim_tokens,
                regr_tokens=det_feat.regr_tokens,
                info=dict(det_feat.info),
            )
        )
    return result, bundle


# ---------------------------------------------------------------------------
# Pairwise feature vector (fixed-dimension, as specified in the task)
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return dot / max(na * nb, 1e-12)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _same_nonempty(a: str, b: str) -> float:
    return 1.0 if a and b and a == b else 0.0


def _same_bool(di: dict, dj: dict, key: str) -> float:
    return 1.0 if bool(di.get(key)) == bool(dj.get(key)) else 0.0


def llm_pair_feature_dim() -> int:
    """Return fixed dimension of pairwise feature vectors (computed from spec)."""
    return 21


def build_llm_pair_feature_vector(a: LLMCaseFeature, b: LLMCaseFeature) -> np.ndarray:
    det_cosine = _cosine(a.det_vec, b.det_vec)
    det_euclidean = _euclidean(a.det_vec, b.det_vec)

    has_llm = a.has_llm and b.has_llm
    llm_cosine = _cosine(a.llm_vec, b.llm_vec) if has_llm else 0.0
    llm_euclidean = _euclidean(a.llm_vec, b.llm_vec) if has_llm else 0.0
    abs_det_llm_diff = abs(det_cosine - llm_cosine)

    token_jaccard = _jaccard(a.token_set, b.token_set)
    primary_token_jaccard = _jaccard(a.primary_tokens, b.primary_tokens)
    sim_token_jaccard = _jaccard(a.sim_tokens, b.sim_tokens)
    regr_token_jaccard = _jaccard(a.regr_tokens, b.regr_tokens)

    ai = a.info
    bi = b.info

    same_primary_signature = _same_nonempty(
        str(ai.get("primary_signature", "")), str(bi.get("primary_signature", ""))
    )
    same_primary_type = _same_nonempty(
        str(ai.get("primary_type", "")), str(bi.get("primary_type", ""))
    )
    same_mismatch_type = _same_nonempty(
        str(ai.get("mismatch_type", "")), str(bi.get("mismatch_type", ""))
    )
    same_op_pair = _same_nonempty(
        str(ai.get("op_pair", "")), str(bi.get("op_pair", ""))
    )
    same_fatal_file = _same_nonempty(
        str(ai.get("fatal_file", "")), str(bi.get("fatal_file", ""))
    )
    same_failed_reason = _same_nonempty(
        str(ai.get("failed_reason", "")), str(bi.get("failed_reason", ""))
    )
    same_has_uvm_fatal = _same_bool(ai, bi, "has_uvm_fatal")
    same_has_uvm_error = _same_bool(ai, bi, "has_uvm_error")
    same_has_regr_mismatch = _same_bool(ai, bi, "has_regr_mismatch")

    na = int(ai.get("num_tokens", 0))
    nb = int(bi.get("num_tokens", 0))
    abs_num_tokens_diff_log = math.log1p(abs(na - nb))
    min_num_tokens_log = math.log1p(min(na, nb))
    max_num_tokens_log = math.log1p(max(na, nb))

    return np.asarray(
        [
            det_cosine,
            llm_cosine,
            abs_det_llm_diff,
            det_euclidean,
            llm_euclidean,
            token_jaccard,
            primary_token_jaccard,
            sim_token_jaccard,
            regr_token_jaccard,
            same_primary_signature,
            same_primary_type,
            same_mismatch_type,
            same_op_pair,
            same_fatal_file,
            same_failed_reason,
            same_has_uvm_fatal,
            same_has_uvm_error,
            same_has_regr_mismatch,
            abs_num_tokens_diff_log,
            min_num_tokens_log,
            max_num_tokens_log,
        ],
        dtype=np.float32,
    )


def build_llm_pair_feature_matrix(
    features: list[LLMCaseFeature],
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    if not pairs:
        return np.zeros((0, llm_pair_feature_dim()), dtype=np.float32)
    sample = build_llm_pair_feature_vector(features[pairs[0][0]], features[pairs[0][1]])
    dim = len(sample)
    matrix = np.empty((len(pairs), dim), dtype=np.float32)
    for idx, (i, j) in enumerate(pairs):
        matrix[idx] = build_llm_pair_feature_vector(features[i], features[j])
    return matrix


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

def _make_mlp(input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.15):
    import torch
    from torch import nn

    class SmallMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                h = int(h)
                layers.append(nn.Linear(prev, h))
                layers.append(nn.LayerNorm(h))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(float(dropout)))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x).squeeze(-1)

    return SmallMLP()


def train_logistic_model(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_scaled, y)
    return {"model": model, "scaler": scaler, "model_type": "logistic"}


def train_gbdt_model(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        early_stopping=False,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X, y)
    return {"model": model, "scaler": None, "model_type": "gbdt"}


def train_mlp_model(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.15,
    batch_size: int = 4096,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    random_state: int = 0,
) -> Any:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(random_state)
    device = pf.resolve_torch_device(device)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    model = _make_mlp(input_dim, hidden_dims, dropout).to(device)
    pos = float((y == 1.0).sum())
    neg = float((y == 0.0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.from_numpy(X_scaled)
    y_t = torch.from_numpy(y.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {
        "model": model,
        "scaler": scaler,
        "state_dict": state_dict,
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "model_type": "mlp",
        "device": device,
    }


# ---------------------------------------------------------------------------
# Pair probability prediction (batched)
# ---------------------------------------------------------------------------

def predict_probability_matrix_sklearn(
    model_pkg: dict,
    features: list[LLMCaseFeature],
    batch_size: int = 100000,
) -> np.ndarray:
    n = len(features)
    probs = np.eye(n, dtype=np.float32)
    if n <= 1:
        return probs

    pairs: list[tuple[int, int]] = []

    def flush() -> None:
        if not pairs:
            return
        X = build_llm_pair_feature_matrix(features, pairs)
        model = model_pkg["model"]
        scaler = model_pkg.get("scaler")
        if scaler is not None:
            X = scaler.transform(X)
        model_type = model_pkg.get("model_type", "")
        if model_type == "mlp":
            import torch
            device = model_pkg.get("device", "cpu")
            model.to(device)
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(X.astype(np.float32)).to(device)).detach().cpu()
                batch_probs = torch.sigmoid(logits).numpy().astype(np.float32)
        elif hasattr(model, "predict_proba"):
            batch_probs = model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            batch_probs = np.clip(model.predict(X).astype(np.float32), 1e-6, 1.0 - 1e-6)
        for (i, j), prob in zip(pairs, batch_probs):
            probs[i, j] = float(prob)
            probs[j, i] = float(prob)
        pairs.clear()

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            if len(pairs) >= batch_size:
                flush()
    flush()
    return probs


def predict_probability_matrix_ensemble(
    model_pkgs: list[dict],
    weights: list[float],
    features: list[LLMCaseFeature],
    ensemble_mode: str = "prob_average",
    batch_size: int = 100000,
) -> np.ndarray:
    """Soft voting ensemble of multiple pairwise models.

    ensemble_mode:
      - prob_average: P_ens = sum(w_i * P_i)
      - logit_average: P_ens = sigmoid(sum(w_i * logit(P_i)))
    """
    n = len(features)
    if n <= 1:
        return np.eye(n, dtype=np.float32)
    if len(model_pkgs) != len(weights):
        raise ValueError(
            f"model_pkgs ({len(model_pkgs)}) and weights ({len(weights)}) length mismatch"
        )
    if abs(sum(weights) - 1.0) > 1e-4:
        raise ValueError(f"weights must sum to 1, got {sum(weights)}")

    prob_matrices = []
    for model_pkg in model_pkgs:
        prob_matrices.append(
            predict_probability_matrix_sklearn(model_pkg, features, batch_size=batch_size)
        )

    if ensemble_mode == "prob_average":
        fused = np.zeros((n, n), dtype=np.float64)
        for w, P in zip(weights, prob_matrices):
            fused += w * P.astype(np.float64)
    elif ensemble_mode == "logit_average":
        fused = np.zeros((n, n), dtype=np.float64)
        eps = 1e-9
        for w, P in zip(weights, prob_matrices):
            P_clipped = np.clip(P.astype(np.float64), eps, 1.0 - eps)
            fused += w * np.log(P_clipped / (1.0 - P_clipped))
        fused = 1.0 / (1.0 + np.exp(-fused))
    else:
        raise ValueError(f"unknown ensemble_mode: {ensemble_mode}")

    return fused.astype(np.float32)


def cluster_from_probability(prob: np.ndarray, k: int) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    n = prob.shape[0]
    if n == 0:
        return []
    k = max(1, min(k, n))
    if k == n:
        return list(range(n))
    distance = 1.0 - prob
    np.fill_diagonal(distance, 0.0)
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return model.fit_predict(distance).tolist()


def save_model_pkg(model_pkg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_type = model_pkg.get("model_type", "")
    if model_type == "mlp":
        import torch
        torch.save(
            {
                "state_dict": model_pkg["state_dict"],
                "input_dim": model_pkg["input_dim"],
                "hidden_dims": model_pkg["hidden_dims"],
                "dropout": model_pkg["dropout"],
                "model_type": "mlp",
            },
            path,
        )
        scaler = model_pkg.get("scaler")
        if scaler is not None:
            import pickle
            scaler_path = path.with_suffix(".scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
    else:
        import pickle
        save_obj = {
            "model": model_pkg["model"],
            "scaler": model_pkg.get("scaler"),
            "model_type": model_type,
        }
        with open(path, "wb") as f:
            pickle.dump(save_obj, f)


def load_model_pkg(path: Path) -> dict:
    if path.suffix in (".pt", ".pth"):
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_type") == "mlp":
            model = _make_mlp(
                int(checkpoint["input_dim"]),
                hidden_dims=checkpoint.get("hidden_dims", (128, 64)),
                dropout=float(checkpoint.get("dropout", 0.15)),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            scaler = None
            scaler_path = path.with_suffix(".scaler.pkl")
            if scaler_path.exists():
                import pickle
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            return {
                "model": model,
                "scaler": scaler,
                "state_dict": checkpoint["state_dict"],
                "input_dim": int(checkpoint["input_dim"]),
                "hidden_dims": checkpoint.get("hidden_dims", (128, 64)),
                "dropout": float(checkpoint.get("dropout", 0.15)),
                "model_type": "mlp",
                "device": "cpu",
            }
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
