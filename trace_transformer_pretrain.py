#!/usr/bin/env python3
"""Self-supervised Trace Transformer with Masked Opcode Modeling pretraining.

Provides:
  - TraceTransformerEncoder  – lightweight 2-layer Transformer with CLS pooling
  - pretrain_trace_encoder   – MLM training loop on tokenized trace sequences
  - TraceEncoderWrapper      – inference-only encoder returning 128d embeddings
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from trace_sequence import (
    PAD_TOKEN,
    UNK_TOKEN,
    MASK_TOKEN,
    CLS_TOKEN,
    TRACE_MISSING_TOKEN,
    EMPTY_TRACE_TOKEN,
    build_vocabulary,
)


# ---------------------------------------------------------------------------
# Positional encoding (learned, not sinusoidal — small seq lens)
# ---------------------------------------------------------------------------
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return x + self.embedding(positions)


# ---------------------------------------------------------------------------
# Trace Transformer Encoder
# ---------------------------------------------------------------------------
class TraceTransformerEncoder(nn.Module):
    """Lightweight Transformer with CLS-pooled trace embedding."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embed = LearnedPositionalEmbedding(max_seq_len + 2, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)

        # MLM head
        self.mlm_head = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.cls_token, std=0.02)

    def _create_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens == self.pad_token_id)  # (B, S), True where pad

    def forward(
        self,
        token_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            token_ids: (B, S) int64 tensor of token ids.
            return_hidden: if True, also return full hidden states (for MLM).

        Returns:
            cls_embed: (B, embed_dim)
            (cls_embed, hidden): if return_hidden=True, hidden is (B, S+1, embed_dim)
        """
        B, S = token_ids.shape
        x = self.token_embed(token_ids)  # (B, S, D)
        x = self.pos_embed(x)
        x = self.dropout(x)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, S+1, D)

        src_key_padding_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=token_ids.device),
             self._create_padding_mask(token_ids)],
            dim=1,
        )
        hidden = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        hidden = self.ln(hidden)

        cls_embed = hidden[:, 0, :]  # (B, D)
        if return_hidden:
            return cls_embed, hidden
        return cls_embed

    def predict_mlm(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict logits over vocabulary from hidden states (skip CLS position)."""
        return self.mlm_head(hidden[:, 1:, :])  # (B, S, vocab_size)


# ---------------------------------------------------------------------------
# Masked Opcode Modeling helpers
# ---------------------------------------------------------------------------
def _is_opcode_token(token_id: int, vocab: dict[str, int]) -> bool:
    """Check whether a token id corresponds to an OP_* token."""
    # Cache the set of OP token ids for efficiency
    if not hasattr(_is_opcode_token, "_cache"):
        _is_opcode_token._cache = {}
    if vocab is not _is_opcode_token._cache.get("_vocab"):
        _is_opcode_token._cache = {
            "_vocab": vocab,
            "_op_ids": {i for t, i in vocab.items() if t.startswith("OP_")},
        }
    return token_id in _is_opcode_token._cache["_op_ids"]


def create_mlm_inputs(
    token_ids: np.ndarray,
    vocab: dict[str, int],
    mlm_prob: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply BERT-style masked-opcode modeling to a batch of token sequences.

    Only tokens that are OP_* category are candidates for masking.  Non-OP
    tokens (CLASS, PCREG, RD, RS, flags) are never masked.

    Returns:
        masked_ids: (B, S) input with some tokens masked/replaced
        labels:     (B, S) with -100 at non-masked positions, true token at masked
        mask_pos:   (B, S) bool where True = a prediction should be made
    """
    if rng is None:
        rng = np.random.default_rng()
    B, S = token_ids.shape
    masked_ids = token_ids.copy()
    labels = np.full((B, S), -100, dtype=np.int64)

    mask_id = vocab[MASK_TOKEN]
    unk_id = vocab[UNK_TOKEN]

    for b in range(B):
        op_positions = [p for p in range(S) if _is_opcode_token(int(token_ids[b, p]), vocab)]
        if not op_positions:
            continue
        n_mask = max(1, int(len(op_positions) * mlm_prob))
        chosen = rng.choice(op_positions, size=n_mask, replace=False)

        for pos in chosen:
            true_id = int(token_ids[b, pos])
            labels[b, pos] = true_id
            r = rng.random()
            if r < 0.8:
                masked_ids[b, pos] = mask_id
            elif r < 0.9:
                masked_ids[b, pos] = rng.integers(6, len(vocab))  # random token (skip specials)
            # else: 10% keep unchanged

    mask_pos = labels != -100
    return masked_ids, labels, mask_pos


# ---------------------------------------------------------------------------
# Pretraining loop
# ---------------------------------------------------------------------------
def pretrain_trace_encoder(
    token_sequences: Sequence[Sequence[str]],
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    max_seq_len: int = 1024,
    mlm_prob: float = 0.15,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device: str = "cuda",
    val_split: float = 0.05,
    patience: int = 10,
    random_state: int = 0,
    min_vocab_freq: int = 1,
    max_vocab_size: int = 16384,
) -> tuple[TraceTransformerEncoder, dict[str, int], dict]:
    """Pretrain a TraceTransformerEncoder via Masked Opcode Modeling.

    Returns (encoder, vocab, history_dict).
    """
    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)

    # Build vocabulary from training set
    vocab = build_vocabulary(token_sequences, min_freq=min_vocab_freq, max_vocab_size=max_vocab_size)

    # Convert token sequences to numpy arrays of ids
    def _encode(seq: Sequence[str]) -> np.ndarray:
        ids = np.array([vocab.get(t, vocab[UNK_TOKEN]) for t in seq[-max_seq_len:]], dtype=np.int64)
        if len(ids) == 0:
            ids = np.array([vocab[EMPTY_TRACE_TOKEN]], dtype=np.int64)
        return ids

    encoded = [_encode(seq) for seq in token_sequences]

    # Pad to max length
    max_len = min(max_seq_len, max(len(e) for e in encoded))
    pad_id = vocab[PAD_TOKEN]
    padded = np.full((len(encoded), max_len), pad_id, dtype=np.int64)
    for i, e in enumerate(encoded):
        n = min(len(e), max_len)
        padded[i, :n] = e[-n:]  # right-aligned (keep tail)

    # Train/val split
    n_val = max(1, int(len(padded) * val_split))
    indices = rng.permutation(len(padded))
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    X_train = padded[train_idx]
    X_val = padded[val_idx]

    print(f"[pretrain] vocab_size={len(vocab)} train_seqs={len(X_train)} val_seqs={len(X_val)} "
          f"max_seq_len={max_len}")

    # Model
    model = TraceTransformerEncoder(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=max_len,
        pad_token_id=pad_id,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        perm = rng.permutation(len(X_train))
        for start in range(0, len(X_train), batch_size):
            batch_idx = perm[start:start + batch_size]
            batch = X_train[batch_idx]
            masked, labels, _ = create_mlm_inputs(batch, vocab, mlm_prob, rng)

            t_ids = torch.from_numpy(masked).to(device)
            t_labels = torch.from_numpy(labels).to(device)

            _, hidden = model(t_ids, return_hidden=True)
            logits = model.predict_mlm(hidden)  # (B, S, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), t_labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_idx)

        train_loss /= len(X_train)
        history["train_loss"].append(train_loss)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(X_val), batch_size):
                batch = X_val[start:start + batch_size]
                masked, labels, _ = create_mlm_inputs(batch, vocab, mlm_prob, rng)

                t_ids = torch.from_numpy(masked).to(device)
                t_labels = torch.from_numpy(labels).to(device)

                _, hidden = model(t_ids, return_hidden=True)
                logits = model.predict_mlm(hidden)
                loss = criterion(logits.reshape(-1, logits.size(-1)), t_labels.reshape(-1))
                val_loss += loss.item() * len(batch)

        val_loss /= len(X_val)
        history["val_loss"].append(val_loss)

        print(f"[pretrain] epoch {epoch + 1:3d}/{epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[pretrain] early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), vocab, history


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------
class TraceEncoderWrapper:
    """Load a pretrained TraceTransformerEncoder and produce trace embeddings."""

    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        device: str = "cuda",
        embed_dim: int = 128,
        max_seq_len: int = 1024,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_layers: int = 2,
    ):
        self.device = device
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Load vocab
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        # Build model with same architecture
        self.model = TraceTransformerEncoder(
            vocab_size=len(self.vocab),
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=0.1,
            max_seq_len=max_seq_len,
            pad_token_id=self.vocab.get(PAD_TOKEN, 0),
        )
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

    def encode_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        """Encode a token sequence to a 128d embedding vector.

        Returns zero vector for empty/missing traces.
        """
        if not tokens or tokens == [TRACE_MISSING_TOKEN] or tokens == [EMPTY_TRACE_TOKEN]:
            return np.zeros(self.embed_dim, dtype=np.float32)

        ids = np.array(
            [self.vocab.get(t, self.vocab.get(UNK_TOKEN, 1)) for t in tokens[-self.max_seq_len:]],
            dtype=np.int64,
        )
        if len(ids) == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)

        # Pad to max_seq_len
        pad_id = self.vocab.get(PAD_TOKEN, 0)
        padded = np.full(self.max_seq_len, pad_id, dtype=np.int64)
        n = min(len(ids), self.max_seq_len)
        padded[:n] = ids[-n:] if n == self.max_seq_len else ids  # left-pad

        t_ids = torch.from_numpy(padded).unsqueeze(0).to(self.device)  # (1, S)
        with torch.no_grad():
            vec = self.model(t_ids)  # (1, D)
        return vec.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_trace_tail(
        self,
        trace_path: str | Path,
        tail_lines: int = 500,
    ) -> np.ndarray:
        """Read tail of a trace file and return its embedding."""
        from trace_sequence import read_trace_file, parse_trace_to_tokens

        try:
            lines = read_trace_file(trace_path)
            tokens = parse_trace_to_tokens(
                lines, window_mode="tail", window_size=tail_lines,
                max_seq_len=self.max_seq_len,
            )
            return self.encode_tokens(tokens)
        except (OSError, ValueError):
            return np.zeros(self.embed_dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------
def save_pretrained(
    model: TraceTransformerEncoder,
    vocab: dict[str, int],
    out_dir: str | Path,
) -> None:
    """Save encoder state_dict, vocab, and architecture config to a directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "encoder.pt")
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    config = {
        "embed_dim": model.embed_dim,
        "num_heads": model.encoder.layers[0].self_attn.num_heads,
        "ff_dim": model.encoder.layers[0].linear1.out_features,
        "num_layers": len(model.encoder.layers),
        "max_seq_len": model.max_seq_len,
        "pad_token_id": model.pad_token_id,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_pretrained(
    model_dir: str | Path,
    device: str = "cuda",
) -> TraceEncoderWrapper:
    """Load a pretrained encoder from a directory (reads config.json for arch params)."""
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", "r") as f:
        config = json.load(f)
    return TraceEncoderWrapper(
        model_path=model_dir / "encoder.pt",
        vocab_path=model_dir / "vocab.json",
        device=device,
        embed_dim=config["embed_dim"],
        max_seq_len=config["max_seq_len"],
        num_heads=config.get("num_heads", 4),
        ff_dim=config.get("ff_dim", 512),
        num_layers=config.get("num_layers", 2),
    )
