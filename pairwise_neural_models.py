#!/usr/bin/env python3
"""Experimental neural classifiers for dense pairwise tabular features."""
from __future__ import annotations

from typing import Sequence


def default_hidden_dims(model_arch: str) -> list[int]:
    if model_arch in {"res_mlp", "gated_mlp"}:
        return [512, 512, 512, 256, 256, 128]
    return [128, 64]


def build_pairwise_neural_model(
    input_dim: int,
    model_arch: str = "res_mlp",
    hidden_dims: Sequence[int] | None = None,
    dropout: float = 0.2,
    layernorm: bool = True,
    batchnorm: bool = False,
    gate_reg: float = 1e-4,
    ft_d_token: int = 64,
    ft_layers: int = 2,
    ft_heads: int = 4,
    ft_dropout: float = 0.1,
    ft_attention_dropout: float = 0.1,
    ft_ffn_mult: int = 2,
    ft_max_tokens: int = 0,
):
    import torch
    from torch import nn
    from torch.nn import functional as F

    if model_arch == "residual":
        model_arch = "res_mlp"
    dims = [int(x) for x in (hidden_dims or default_hidden_dims(model_arch))]

    def norm(dim: int):
        if batchnorm:
            return nn.BatchNorm1d(dim)
        if layernorm:
            return nn.LayerNorm(dim)
        return nn.Identity()

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim), norm(dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim, dim), norm(dim),
            )
            self.activation = nn.GELU()

        def forward(self, x):
            return self.activation(x + self.net(x))

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            first = dims[0]
            layers: list[nn.Module] = [nn.Linear(input_dim, first), norm(first), nn.GELU(), nn.Dropout(dropout)]
            prev = first
            for hidden in dims[1:]:
                if hidden == prev:
                    layers.append(ResidualBlock(hidden))
                else:
                    layers.extend([nn.Linear(prev, hidden), norm(hidden), nn.GELU(), nn.Dropout(dropout)])
                prev = hidden
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

        def regularization_loss(self):
            return next(self.parameters()).new_zeros(())

    class GatedResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            gate_hidden = min(256, input_dim)
            self.gate = nn.Sequential(
                nn.Linear(input_dim, gate_hidden), nn.GELU(),
                nn.Linear(gate_hidden, input_dim), nn.Sigmoid(),
            )
            self.backbone = ResidualMLP()
            self.last_gate = None

        def forward(self, x):
            gate = self.gate(x)
            self.last_gate = gate
            return self.backbone(x * gate)

        def regularization_loss(self):
            if self.last_gate is None:
                return next(self.parameters()).new_zeros(())
            return float(gate_reg) * self.last_gate.mean()

    class FTTransformerPairClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if ft_d_token % ft_heads != 0:
                raise ValueError("ft_d_token must be divisible by ft_heads")
            self.weight = nn.Parameter(torch.empty(input_dim, ft_d_token))
            self.bias = nn.Parameter(torch.zeros(input_dim, ft_d_token))
            self.cls = nn.Parameter(torch.zeros(1, 1, ft_d_token))
            nn.init.normal_(self.weight, std=0.02)
            nn.init.normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=ft_d_token,
                nhead=ft_heads,
                dim_feedforward=ft_d_token * ft_ffn_mult,
                dropout=ft_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # PyTorch exposes one dropout value for attention and residual paths.
            # Keep the requested attention value in config; use the larger value
            # so neither regularizer is silently weakened.
            if ft_attention_dropout > ft_dropout:
                layer.self_attn.dropout = float(ft_attention_dropout)
            self.encoder = nn.TransformerEncoder(layer, num_layers=ft_layers, norm=nn.LayerNorm(ft_d_token))
            self.head = nn.Sequential(
                nn.LayerNorm(ft_d_token), nn.Linear(ft_d_token, 128), nn.ReLU(),
                nn.Dropout(ft_dropout), nn.Linear(128, 1),
            )

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
            if ft_max_tokens > 0 and tokens.shape[1] > ft_max_tokens:
                tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), ft_max_tokens).transpose(1, 2)
            cls = self.cls.expand(x.shape[0], -1, -1)
            encoded = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.head(encoded[:, 0]).squeeze(-1)

        def regularization_loss(self):
            return next(self.parameters()).new_zeros(())

    if model_arch == "res_mlp":
        return ResidualMLP()
    if model_arch == "gated_mlp":
        return GatedResidualMLP()
    if model_arch == "ft_transformer":
        return FTTransformerPairClassifier()
    raise ValueError(f"unknown model_arch: {model_arch}")


def model_config_from_args(args) -> dict:
    return {
        "gate_reg": float(getattr(args, "gate_reg", 1e-4)),
        "ft_d_token": int(getattr(args, "ft_d_token", 64)),
        "ft_layers": int(getattr(args, "ft_layers", 2)),
        "ft_heads": int(getattr(args, "ft_heads", 4)),
        "ft_dropout": float(getattr(args, "ft_dropout", 0.1)),
        "ft_attention_dropout": float(getattr(args, "ft_attention_dropout", 0.1)),
        "ft_ffn_mult": int(getattr(args, "ft_ffn_mult", 2)),
        "ft_max_tokens": int(getattr(args, "ft_max_tokens", 96)),
    }
