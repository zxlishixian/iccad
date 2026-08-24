#!/usr/bin/env python3
"""Siamese per-case encoder + prototype learning for regression-failure bucketing.

Replaces the O(N^2) pair model with an O(N) per-case encoder:
  case features -> encoder -> normalized embedding z
  train  = SupCon (same-bug together) + prototype/center loss (per-bug centers)
  infer  = encode all cases, then k-means into k buckets

Gold labels are used only to derive the k prototypes and the contrastive targets
during training; inference never reads labels.
"""
from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


def supcon_loss(features: torch.Tensor, targets: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss: pull same-label rows together, push others apart.

    ``targets`` are integer labels (bug ids); every pair of rows sharing a label
    is a positive pair.
    """
    from torch.nn import functional as F
    if features.shape[0] < 2:
        return features.new_zeros(())
    normalized = F.normalize(features, dim=1)
    similarity = torch.matmul(normalized, normalized.T) / max(float(temperature), 1e-6)
    exp_sim = torch.exp(similarity - similarity.max(dim=1, keepdim=True).values.detach())
    eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
    same = targets[:, None] == targets[None, :]
    positive = same & ~eye
    denominator = exp_sim.sum(dim=1) - exp_sim.diagonal()
    numerator = (exp_sim * positive.float()).sum(dim=1) + 1e-12  # eps avoids log(0) -> NaN grad
    log_prob = torch.log(numerator / torch.clamp(denominator, min=1e-9))
    mask = positive.any(dim=1)
    if not bool(mask.any()):
        return features.new_zeros(())
    return -log_prob[mask].mean()


def positive_aggregation_loss(features: torch.Tensor, targets: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Explicitly pull same-bug embeddings together (raise TPR / reduce fragmentation).

    SupCon's denominator is dominated by the many negatives, so the positive
    signal can be drowned (high TNR, low TPR).  This term maximizes the mean
    similarity of same-bug pairs directly, complementing the negative push.
    """
    from torch.nn import functional as F
    n = features.shape[0]
    if n < 2:
        return features.new_zeros(())
    normalized = F.normalize(features, dim=1)
    sim = torch.matmul(normalized, normalized.T) / max(float(temperature), 1e-6)
    same = targets[:, None] == targets[None, :]
    eye = torch.eye(n, device=features.device, dtype=torch.bool)
    pos_mask = same & ~eye
    if not bool(pos_mask.any()):
        return features.new_zeros(())
    return -sim[pos_mask].mean()


def _simple_kmeans(x: torch.Tensor, k: int, iters: int = 5) -> torch.Tensor:
    """Tiny non-differentiable k-means (NumPy, detached) returning k centroids."""
    import numpy as np
    xn = x.detach().cpu().numpy()
    n = len(xn)
    k = max(1, min(int(k), n))
    if k == 1:
        return x.mean(dim=0, keepdim=True).detach()
    rng = np.random.default_rng(0)
    centers = xn[rng.choice(n, size=k, replace=False)]
    for _ in range(iters):
        d = ((xn[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(-1)
        for j in range(k):
            if (assign == j).any():
                centers[j] = xn[assign == j].mean(0)
    return torch.from_numpy(centers.astype(np.float32)).to(x.device)


def prototype_loss(features: torch.Tensor, targets: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Prototype/center contrastive loss (DigPro-style).

    Computes one prototype per distinct label as the mean of that label's
    normalized embeddings, then pulls each embedding toward its own prototype and
    pushes it away from other prototypes.
    """
    from torch.nn import functional as F
    if features.shape[0] < 2:
        return features.new_zeros(())
    normalized = F.normalize(features, dim=1)
    labels = torch.unique(targets)
    if len(labels) < 2:
        return features.new_zeros(())
    prototypes = torch.stack([normalized[targets == lab].mean(dim=0) for lab in labels])
    prototypes = F.normalize(prototypes, dim=1)
    sim = torch.matmul(normalized, prototypes.T) / max(float(temperature), 1e-6)
    # target index for each row
    lab_to_idx = {int(lab): i for i, lab in enumerate(labels)}
    idx = torch.tensor([lab_to_idx[int(t)] for t in targets], device=features.device)
    log_prob = torch.log_softmax(sim, dim=1)
    loss = -log_prob[torch.arange(features.shape[0], device=features.device), idx].mean()
    return loss


def subcenter_loss(features: torch.Tensor, targets: torch.Tensor, temperature: float = 0.1,
                   n_subcenters: int = 3) -> torch.Tensor:
    """Multi-prototype (sub-center) contrastive loss.

    Instead of one centroid per label, cluster each label's embeddings into
    ``n_subcenters`` centroids (detached k-means) and pull each embedding toward
    ALL its own label's sub-centers, pushing away from other labels' sub-centers.
    This handles multi-modal (same-bug-manifests-differently) classes.
    """
    from torch.nn import functional as F
    if features.shape[0] < 2:
        return features.new_zeros(())
    normalized = F.normalize(features, dim=1)
    labels = torch.unique(targets)
    if len(labels) < 2:
        return features.new_zeros(())
    subcenters: list[torch.Tensor] = []
    sub_labels: list[int] = []
    for li, lab in enumerate(labels):
        lab_feats = normalized[targets == lab]
        sub = _simple_kmeans(lab_feats, n_subcenters)
        subcenters.append(sub)
        sub_labels.extend([li] * sub.shape[0])
    subcenters = F.normalize(torch.cat(subcenters, dim=0), dim=1)
    sim = torch.matmul(normalized, subcenters.T) / max(float(temperature), 1e-6)
    lab_to_idx = {int(lab): i for i, lab in enumerate(labels)}
    case_lab = torch.tensor([lab_to_idx[int(t)] for t in targets], device=features.device)
    sub_lab = torch.tensor(sub_labels, device=features.device)
    pos_mask = (case_lab[:, None] == sub_lab[None, :])
    log_sum_pos = torch.logsumexp(sim.masked_fill(~pos_mask, -float("inf")), dim=1)
    log_sum_all = torch.logsumexp(sim, dim=1)
    return -(log_sum_pos - log_sum_all).mean()


class _GradReverse(torch.autograd.Function):
    """Gradient reversal layer (DANN): reverses the gradient flowing back to the encoder.

    The domain discriminator's gradient is NOT reversed, so it learns to predict the
    domain, while the encoder's gradient IS reversed, so it learns domain-invariant
    (fake-vs-official indistinguishable) embeddings.
    """

    @staticmethod
    def forward(ctx, x):  # noqa: D401
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # noqa: D401
        return -grad_output


def grad_reverse(x: torch.Tensor) -> torch.Tensor:
    return _GradReverse.apply(x)


class DomainDiscriminator(nn.Module):
    """Small MLP predicting fake(0)/official(1) from a normalized embedding."""

    def __init__(self, in_dim: int = 256, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TraceSeqEncoder(nn.Module):
    """Order-sensitive GRU over the instruction-family sequence (DeepLog-style).

    The divergence path is a SEQUENCE; a GRU (unlike a max-pool CNN) preserves the
    order of the retired instructions, which is the key signal for separating
    control-flow bugs from each other.
    """

    def __init__(self, vocab: int = 74, embed_dim: int = 32, hidden: int = 48, out_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=vocab - 1)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (batch, seq_len) long
        x = self.embed(seq)                # (batch, seq_len, embed_dim)
        out, _ = self.gru(x)               # (batch, seq_len, hidden)
        pooled = out.mean(dim=1)           # mean-pool over time (order-aware via GRU)
        return self.proj(pooled)           # (batch, out_dim)


class SiameseEncoder(nn.Module):
    """Per-case MLP encoder mapping a case feature vector to a normalized embedding.

    Optionally fuses a per-case instruction-family SEQUENCE via a 1D-CNN (``use_seq``)
    so the embedding sees the ordered divergence path, not just the hash-count residual.
    """

    def __init__(self, input_dim: int, hidden: int = 256, out_dim: int = 256, dropout: float = 0.2,
                 use_seq: bool = False, seq_vocab: int = 74, seq_out: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.use_seq = use_seq
        if use_seq:
            self.seq_enc = TraceSeqEncoder(vocab=seq_vocab, out_dim=seq_out)
            self.proj = nn.Linear(out_dim + seq_out, out_dim)

    def forward(self, value: torch.Tensor, seq: torch.Tensor | None = None) -> torch.Tensor:
        from torch.nn import functional as F
        flat = self.net(value)
        if self.use_seq and seq is not None:
            flat = self.proj(torch.cat([flat, self.seq_enc(seq)], dim=1))
        return F.normalize(flat, dim=1)


def train_siamese_model(
    case_matrix: np.ndarray,
    case_labels: np.ndarray,
    args: Any,
    case_domains: np.ndarray | None = None,
    case_seq: np.ndarray | None = None,
) -> dict:
    """Train a per-case siamese encoder with SupCon + prototype loss.

    ``case_matrix`` is (n_cases, feature_dim); ``case_labels`` is integer bug ids.
    ``case_seq`` (optional) is (n_cases, seq_len) instruction-family sequence; when
    given, a 1D-CNN fuses it into the embedding (replacing the hash-count residual).
    ``case_domains`` (optional) is 0/1 (fake/official) per case; when given and
    ``args.domain_adv_weight > 0``, a gradient-reversed domain discriminator is
    added so the encoder learns domain-invariant embeddings (DANN).
    """
    from torch.utils.data import DataLoader, TensorDataset
    from pairwise_features import resolve_torch_device

    case_matrix = np.asarray(case_matrix, dtype=np.float32)
    case_labels = np.asarray(case_labels)
    use_seq = case_seq is not None
    if use_seq:
        case_seq = np.asarray(case_seq, dtype=np.int64)
    seed = int(args.random_state)
    supcon_w = float(getattr(args, "supcon_weight", 0.1))
    proto_w = float(getattr(args, "prototype_weight", 0.1))
    proto_sub = int(getattr(args, "prototype_subcenters", 1))
    pos_agg_w = float(getattr(args, "pos_agg_weight", 0.0))
    domain_adv_w = float(getattr(args, "domain_adv_weight", 0.0))
    temperature = float(getattr(args, "supcon_temperature", 0.1))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_torch_device(str(args.device))
    model = SiameseEncoder(int(case_matrix.shape[1]), dropout=float(args.dropout), use_seq=use_seq).to(device)
    discriminator = None
    if domain_adv_w > 0.0 and case_domains is not None:
        discriminator = DomainDiscriminator().to(device)
    params = list(model.parameters()) + (list(discriminator.parameters()) if discriminator is not None else [])
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    # Stratified train/val split for validation-based early stopping.
    n = len(case_labels)
    indices = np.arange(n)
    split_ok = False
    if n >= 20 and len(np.unique(case_labels)) >= 2:
        try:
            from sklearn.model_selection import train_test_split
            tr_idx, va_idx = train_test_split(indices, test_size=0.15, stratify=case_labels, random_state=seed)
            split_ok = True
        except ValueError:
            # sparse classes (e.g. benchmark5 has bugs with 1 case) break stratification
            split_ok = False
    if not split_ok:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        split = max(1, int(round(0.85 * n)))
        tr_idx, va_idx = indices[:split], indices[split:]
    tensors = [torch.from_numpy(case_matrix[tr_idx]), torch.from_numpy(case_labels[tr_idx])]
    if case_domains is not None and domain_adv_w > 0.0:
        tensors.append(torch.from_numpy(np.asarray(case_domains, dtype=np.float32)[tr_idx]))
    if use_seq:
        tensors.append(torch.from_numpy(case_seq[tr_idx]))
    loader = DataLoader(
        TensorDataset(*tensors), batch_size=max(1, int(args.batch_size)), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_x = torch.from_numpy(case_matrix[va_idx]).to(device)
    val_y = torch.from_numpy(case_labels[va_idx]).to(device)
    val_seq = torch.from_numpy(case_seq[va_idx]).to(device) if use_seq else None

    best_state = None
    best_val = float("inf")
    stale = 0
    for epoch in range(max(1, int(args.epochs))):
        model.train()
        for batch in loader:
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
            batch_domain = batch[2].to(device) if (case_domains is not None and domain_adv_w > 0.0 and len(batch) > 2) else None
            batch_seq = batch[-1].to(device) if use_seq else None
            optimizer.zero_grad(set_to_none=True)
            z = model(batch_x, batch_seq)
            loss = torch.zeros((), device=device)
            if supcon_w > 0.0:
                loss = loss + supcon_w * supcon_loss(z, batch_y, temperature)
            if proto_w > 0.0:
                if proto_sub > 1:
                    loss = loss + proto_w * subcenter_loss(z, batch_y, temperature, proto_sub)
                else:
                    loss = loss + proto_w * prototype_loss(z, batch_y, temperature)
            if pos_agg_w > 0.0:
                loss = loss + pos_agg_w * positive_aggregation_loss(z, batch_y, temperature)
            if discriminator is not None and batch_domain is not None:
                from torch.nn import functional as F
                domain_logits = discriminator(grad_reverse(z))
                domain_loss = F.binary_cross_entropy_with_logits(domain_logits, batch_domain)
                loss = loss + domain_adv_w * domain_loss
            if not torch.isfinite(loss):
                # Numerical blowup (e.g. a zero prototype vector); skip this batch.
                continue
            if float(loss.item()) == 0.0:
                loss = torch.mean(z * z) * 1e-4  # degenerate fallback to keep gradients alive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        # Validation loss (early stopping target — prevents overfitting).
        model.eval()
        with torch.no_grad():
            zv = model(val_x, val_seq)
            val_loss = torch.zeros((), device=device)
            if supcon_w > 0.0:
                val_loss = val_loss + supcon_w * supcon_loss(zv, val_y, temperature)
            if proto_w > 0.0:
                if proto_sub > 1:
                    val_loss = val_loss + proto_w * subcenter_loss(zv, val_y, temperature, proto_sub)
                else:
                    val_loss = val_loss + proto_w * prototype_loss(zv, val_y, temperature)
        val_loss = float(val_loss.cpu())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
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
        "model_type": "siamese_encoder",
        "input_dim": int(case_matrix.shape[1]),
        "supcon_weight": supcon_w,
        "prototype_weight": proto_w,
        "best_val_loss": best_val,
        "best_epoch": epoch if best_state is not None else -1,
    }


def encode_cases(model: nn.Module, case_matrix: np.ndarray, batch_size: int = 4096,
                 case_seq: np.ndarray | None = None) -> np.ndarray:
    """Encode all cases to embeddings (O(N))."""
    import torch
    model = model.cpu().eval()
    out = []
    x = np.asarray(case_matrix, dtype=np.float32)
    s = np.asarray(case_seq, dtype=np.int64) if case_seq is not None else None
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = torch.from_numpy(x[start:start + batch_size])
            seq_chunk = torch.from_numpy(s[start:start + batch_size]) if s is not None else None
            out.append(model(chunk, seq_chunk).numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 256), dtype=np.float32)


def cluster_embeddings(embeddings: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """k-means over embeddings -> cluster labels (O(N*k*iter))."""
    from sklearn.cluster import KMeans
    embeddings = np.nan_to_num(np.asarray(embeddings, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = embeddings.shape[0]
    k = max(1, min(int(k), n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64)
    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(embeddings).astype(np.int64)
