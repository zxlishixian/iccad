#!/usr/bin/env python3
"""Neural two-tower pair model for experimental Theta TriLog routes."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn


def supcon_loss(features, targets, temperature: float = 0.1):
    """Supervised contrastive loss on batch representations.

    Pulls same-label rows together and pushes different-label rows apart. Targets
    are binary same/different labels; every same-label pair is a positive.
    """
    import torch
    from torch.nn import functional as F

    normalized = F.normalize(features, dim=1)
    similarity = torch.matmul(normalized, normalized.T) / max(float(temperature), 1e-6)
    exp_sim = torch.exp(similarity - similarity.max(dim=1, keepdim=True).values.detach())
    eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
    same = targets[:, None] == targets[None, :]
    positive = same & ~eye
    denominator = exp_sim.sum(dim=1) - exp_sim.diagonal()
    numerator = (exp_sim * positive.float()).sum(dim=1)
    log_prob = torch.log(numerator / torch.clamp(denominator, min=1e-9))
    mask = positive.any(dim=1)
    if not bool(mask.any()):
        return features.new_zeros(())
    return -log_prob[mask].mean()


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class _Tower(nn.Module):
    def __init__(self, source_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(source_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), _ResidualBlock(hidden, dropout),
        )

    def forward(self, value):
        return self.net(value)


class TriLogPairNet(nn.Module):
    def __init__(self, base_dim: int, trace_dim: int, dropout: float, fusion: str) -> None:
        super().__init__()
        self.base_dim = int(base_dim)
        self.trace_dim = int(trace_dim)
        self.fusion = str(fusion)
        self.base_tower = _Tower(self.base_dim, 256, dropout) if self.base_dim else None
        self.trace_tower = _Tower(self.trace_dim, 256, dropout) if self.trace_dim else None
        both = self.base_tower is not None and self.trace_tower is not None
        fused_dim = 256 * 4 if both and self.fusion == "concat" else 256
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), _ResidualBlock(512, dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), _ResidualBlock(256, dropout), nn.Linear(256, 1),
        )

    def forward_features(self, value):
        base = self.base_tower(value[:, : self.base_dim]) if self.base_tower is not None else None
        trace = self.trace_tower(value[:, self.base_dim :]) if self.trace_tower is not None else None
        if base is not None and trace is not None:
            if self.fusion == "sum":
                return base + trace
            return torch.cat([base, trace, torch.abs(base - trace), base * trace], dim=1)
        return base if base is not None else trace

    def forward(self, value):
        return self.head(self.forward_features(value)).squeeze(-1)


def _build_network(input_dim: int, base_dim: int, dropout: float, fusion: str = "concat"):
    trace_dim = int(input_dim) - int(base_dim)
    if base_dim < 0 or trace_dim < 0 or input_dim <= 0:
        raise ValueError(f"invalid TriLog dimensions input={input_dim} base={base_dim}")
    if fusion not in ("concat", "sum"):
        raise ValueError(f"unknown fusion mode: {fusion}")
    return TriLogPairNet(int(base_dim), int(trace_dim), float(dropout), str(fusion))


def train_trilog_pair_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    base_dim: int,
    args: Any,
) -> dict:
    """Train a joint sim/regr and trace tower using weighted focal loss."""
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from torch.nn import functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    from pairwise_features import resolve_torch_device

    matrix = np.asarray(matrix, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    sample_weight = np.asarray(sample_weight, dtype=np.float32)
    seed = int(args.random_state)
    supcon_weight = float(getattr(args, "supcon_weight", 0.0))
    supcon_temperature = float(getattr(args, "supcon_temperature", 0.1))
    torch.manual_seed(seed)
    np.random.seed(seed)
    indices = np.arange(len(labels))
    if len(labels) >= 20 and len(np.unique(labels)) == 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_indices, validation_indices = next(splitter.split(matrix, labels))
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        split = max(1, min(len(indices) - 1, int(round(0.85 * len(indices)))))
        train_indices, validation_indices = indices[:split], indices[split:]

    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(matrix[train_indices]).astype(np.float32)
    validation_matrix = scaler.transform(matrix[validation_indices]).astype(np.float32)
    device = resolve_torch_device(str(args.device))
    fusion = str(getattr(args, "fusion", "concat"))
    model = _build_network(matrix.shape[1], base_dim, float(args.dropout), fusion=fusion).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_matrix),
            torch.from_numpy(labels[train_indices]),
            torch.from_numpy(sample_weight[train_indices]),
        ),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    def focal_loss(logits, targets, weights):  # type: ignore[no-untyped-def]
        raw = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probability, 1.0 - probability)
        raw = raw * torch.pow(1.0 - pt, float(args.focal_gamma))
        return torch.sum(raw * weights) / torch.clamp(weights.sum(), min=1e-6)

    validation_x = torch.from_numpy(validation_matrix).to(device)
    validation_y = torch.from_numpy(labels[validation_indices]).to(device)
    validation_w = torch.from_numpy(sample_weight[validation_indices]).to(device)
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    for epoch in range(max(1, int(args.epochs))):
        model.train()
        running = 0.0
        batches = 0
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)
            features = model.forward_features(batch_x)
            logits = model.head(features).squeeze(-1)
            loss = focal_loss(logits, batch_y, batch_w)
            if supcon_weight > 0.0:
                loss = loss + supcon_weight * supcon_loss(features, batch_y, supcon_temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(focal_loss(model(validation_x), validation_y, validation_w).cpu())
        history.append({
            "epoch": epoch,
            "train_loss": running / max(1, batches),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= max(1, int(args.early_stop_patience)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model.cpu().eval(),
        "scaler": scaler,
        "model_type": "theta_trilog_mlp",
        "input_dim": int(matrix.shape[1]),
        "base_dim": int(base_dim),
        "trace_dim": int(matrix.shape[1] - base_dim),
        "dropout": float(args.dropout),
        "fusion": fusion,
        "supcon_weight": supcon_weight,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "device_used": str(device),
    }


def _predict_logits(
    model_package: dict,
    matrix: np.ndarray,
    device: str = "cpu",
    batch_size: int = 65536,
) -> np.ndarray:
    import torch

    from pairwise_features import resolve_torch_device

    scaler = model_package["scaler"]
    effective = scaler.transform(np.asarray(matrix, dtype=np.float32)).astype(np.float32)
    torch_device = resolve_torch_device(str(device))
    model = model_package["model"].to(torch_device).eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(effective), max(1, int(batch_size))):
            batch = torch.from_numpy(effective[start : start + batch_size]).to(torch_device)
            chunks.append(model(batch).cpu().numpy().astype(np.float32))
    model.cpu()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def _class_balanced_weights(labels: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    output = np.ones(len(labels), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32).copy()
    for target in (0.0, 1.0):
        mask = labels == target
        count = int(np.sum(mask))
        if count:
            output[mask] *= len(labels) / (2.0 * count)
    output /= max(float(np.mean(output)), 1e-12)
    return output.astype(np.float32, copy=False)


def fit_official_affine_calibration(
    model_package: dict,
    official_matrix: np.ndarray,
    official_labels: np.ndarray,
    official_weight: np.ndarray,
    args: Any,
) -> dict:
    """Fit two calibration scalars on one source episode's pair labels."""
    import torch
    from torch.nn import functional as functional

    package = dict(model_package)
    logits = torch.from_numpy(_predict_logits(model_package, official_matrix)).float()
    labels = torch.from_numpy(np.asarray(official_labels, dtype=np.float32))
    weights = torch.from_numpy(_class_balanced_weights(official_labels, official_weight))
    raw_scale = torch.nn.Parameter(torch.zeros(()))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([raw_scale, bias], lr=float(args.finetune_lr))
    smoothing = float(getattr(args, "label_smoothing", 0.02))
    targets = labels * (1.0 - 2.0 * smoothing) + smoothing
    history: list[dict] = []
    best = (float("inf"), 1.0, 0.0)
    for epoch in range(max(1, int(args.finetune_epochs))):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(raw_scale)
        adjusted = logits * scale + bias
        raw = functional.binary_cross_entropy_with_logits(adjusted, targets, reduction="none")
        pair_loss = torch.sum(raw * weights) / torch.clamp(weights.sum(), min=1e-6)
        regularization = float(args.affine_reg) * ((scale - 1.0) ** 2 + bias ** 2)
        loss = pair_loss + regularization
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        history.append({"epoch": epoch, "loss": value, "pair_loss": float(pair_loss.detach())})
        if value < best[0]:
            best = (value, float(scale.detach()), float(bias.detach()))
    package.update({
        "calibration_scale": best[1],
        "calibration_bias": best[2],
        "adaptation": "official_affine",
        "adaptation_history": history,
    })
    return package


def _positive_components(
    pairs: list[tuple[int, int]], labels: np.ndarray,
) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {}
    for (left, right), label in zip(pairs, labels):
        if label <= 0.5:
            continue
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))
    components: list[list[int]] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, ()))
        remaining -= component
        components.append(sorted(component))
    return components


def fine_tune_official_pair_model(
    model_package: dict,
    official_matrix: np.ndarray,
    official_labels: np.ndarray,
    official_pairs: list[tuple[int, int]],
    official_weight: np.ndarray,
    replay_matrix: np.ndarray,
    args: Any,
) -> dict:
    """Constrained official-pair adaptation with ranking/connectivity/replay."""
    import torch
    from torch.nn import functional as functional

    from pairwise_features import resolve_torch_device

    package = dict(model_package)
    model = copy.deepcopy(model_package["model"])
    scope = str(args.finetune_scope)
    for parameter in model.parameters():
        parameter.requires_grad = scope == "full"
    if scope == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad = True
    elif scope == "last":
        for parameter in model.head[-1].parameters():
            parameter.requires_grad = True
    elif scope != "full":
        raise ValueError(f"unknown fine-tune scope: {scope}")

    device = resolve_torch_device(str(args.device))
    model = model.to(device)
    scaler = model_package["scaler"]
    official_x = torch.from_numpy(
        scaler.transform(np.asarray(official_matrix, dtype=np.float32)).astype(np.float32)
    ).to(device)
    official_y_np = np.asarray(official_labels, dtype=np.float32)
    official_y = torch.from_numpy(official_y_np).to(device)
    official_w = torch.from_numpy(
        _class_balanced_weights(official_y_np, official_weight)
    ).to(device)
    replay_np = np.asarray(replay_matrix, dtype=np.float32)
    replay_x = torch.from_numpy(
        scaler.transform(replay_np).astype(np.float32)
    ).to(device) if len(replay_np) else None
    replay_teacher = None
    if replay_x is not None:
        teacher = model_package["model"].to(device).eval()
        with torch.no_grad():
            replay_teacher = torch.sigmoid(teacher(replay_x)).detach()
        teacher.cpu()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(args.finetune_lr), weight_decay=float(args.finetune_weight_decay)
    )
    smoothing = float(args.label_smoothing)
    targets = official_y * (1.0 - 2.0 * smoothing) + smoothing
    positive_rows = torch.nonzero(official_y > 0.5, as_tuple=False).flatten()
    negative_rows = torch.nonzero(official_y < 0.5, as_tuple=False).flatten()
    pair_row = {tuple(sorted(map(int, pair))): row for row, pair in enumerate(official_pairs)}
    incident_positive: dict[int, list[int]] = {}
    for row, ((left, right), label) in enumerate(zip(official_pairs, official_y_np)):
        if label <= 0.5:
            continue
        incident_positive.setdefault(int(left), []).append(row)
        incident_positive.setdefault(int(right), []).append(row)
    components = _positive_components(official_pairs, official_y_np)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    history: list[dict] = []
    for epoch in range(max(1, int(args.finetune_epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(official_x)
        raw = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pair_loss = torch.sum(raw * official_w) / torch.clamp(official_w.sum(), min=1e-6)

        ranking_loss = logits.new_zeros(())
        if len(positive_rows) and len(negative_rows) and float(args.ranking_weight) > 0.0:
            positive_logits = logits[positive_rows]
            negative_logits = logits[negative_rows]
            ranking_loss = torch.relu(
                float(args.ranking_margin)
                - positive_logits[:, None] + negative_logits[None, :]
            ).mean()

        connectivity_loss = logits.new_zeros(())
        if incident_positive and float(args.connectivity_weight) > 0.0:
            probability = torch.sigmoid(logits)
            terms = []
            for rows in incident_positive.values():
                values = probability[torch.as_tensor(rows, device=device)]
                count = min(max(1, int(args.connectivity_top_m)), len(rows))
                top = torch.topk(values, k=count).values
                terms.append(-torch.log(torch.clamp(top.mean(), min=1e-6)))
            connectivity_loss = torch.stack(terms).mean()

        transitivity_loss = logits.new_zeros(())
        if float(args.transitivity_weight) > 0.0:
            probability = torch.sigmoid(logits)
            terms = []
            for component in components:
                for first_pos in range(len(component)):
                    for second_pos in range(first_pos + 1, len(component)):
                        for third_pos in range(second_pos + 1, len(component)):
                            nodes = (component[first_pos], component[second_pos], component[third_pos])
                            rows = [pair_row.get(tuple(sorted(pair))) for pair in (
                                (nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[1], nodes[2])
                            )]
                            if any(row is None for row in rows):
                                continue
                            values = probability[torch.as_tensor(rows, device=device)]
                            terms.append(torch.relu(values.max() - values.min() - 0.20))
            if terms:
                transitivity_loss = torch.stack(terms).mean()

        replay_loss = logits.new_zeros(())
        if replay_x is not None and replay_teacher is not None and float(args.replay_weight) > 0.0:
            replay_logits = model(replay_x)
            replay_loss = functional.binary_cross_entropy_with_logits(
                replay_logits, replay_teacher
            )
        loss = (
            pair_loss
            + float(args.ranking_weight) * ranking_loss
            + float(args.connectivity_weight) * connectivity_loss
            + float(args.transitivity_weight) * transitivity_loss
            + float(args.replay_weight) * replay_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=3.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        history.append({
            "epoch": epoch,
            "loss": value,
            "pair_loss": float(pair_loss.detach().cpu()),
            "ranking_loss": float(ranking_loss.detach().cpu()),
            "connectivity_loss": float(connectivity_loss.detach().cpu()),
            "transitivity_loss": float(transitivity_loss.detach().cpu()),
            "replay_loss": float(replay_loss.detach().cpu()),
        })
        if value < best_loss:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    package.update({
        "model": model.cpu().eval(),
        "adaptation": f"official_{scope}",
        "adaptation_history": history,
        "adaptation_best_epoch": best_epoch,
        "adaptation_best_loss": best_loss,
    })
    return package


def predict_trilog_pair_model(model_package: dict, matrix: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    logits = _predict_logits(model_package, matrix, batch_size=batch_size)
    logits = (
        logits * float(model_package.get("calibration_scale", 1.0))
        + float(model_package.get("calibration_bias", 0.0))
    )
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).astype(np.float32)
