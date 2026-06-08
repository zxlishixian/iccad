from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pairwise_llm_features as plf
from pairwise_neural_models import build_pairwise_neural_model

torch = pytest.importorskip("torch")

@pytest.mark.parametrize("arch", ["res_mlp", "gated_mlp", "ft_transformer"])
def test_architecture_forward_shape(arch: str) -> None:
    model = build_pairwise_neural_model(
        input_dim=37, model_arch=arch,
        hidden_dims=[32, 32, 16] if arch != "ft_transformer" else None,
        ft_d_token=16, ft_heads=4, ft_layers=1, ft_max_tokens=12,
    )
    output = model(torch.randn(5, 37))
    assert output.shape == (5,)
    assert torch.isfinite(output).all()

def test_gate_regularization_is_active() -> None:
    model = build_pairwise_neural_model(
        input_dim=11, model_arch="gated_mlp", hidden_dims=[16, 16], gate_reg=1e-4,
    )
    model(torch.randn(4, 11))
    assert float(model.regularization_loss()) > 0.0

@pytest.mark.parametrize("arch", ["res_mlp", "gated_mlp", "ft_transformer"])
def test_new_checkpoint_round_trip(tmp_path: Path, arch: str) -> None:
    config = {
        "ft_d_token": 16, "ft_layers": 1, "ft_heads": 4,
        "ft_dropout": 0.1, "ft_attention_dropout": 0.1,
        "ft_ffn_mult": 2, "ft_max_tokens": 12, "gate_reg": 1e-4,
    }
    hidden = [24, 24, 12]
    model = build_pairwise_neural_model(19, arch, hidden_dims=hidden, dropout=0.1, **config)
    model.eval()
    sample = torch.randn(3, 19)
    with torch.no_grad(): expected = model(sample).numpy()
    path = tmp_path / f"{arch}.pt"
    plf.save_model_pkg({
        "model_type": "mlp", "model": model,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": 19, "hidden_dims": hidden, "dropout": 0.1,
        "mlp_arch": "residual", "model_arch": arch, "model_config": config,
        "layernorm": True, "batchnorm": False,
        "feature_mode": "llm_dual_struct_det_summary", "feature_schema_version": 1,
    }, path)
    loaded = plf.load_model_pkg(path)
    with torch.no_grad(): actual = loaded["model"](sample).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

def test_legacy_auto_checkpoint_round_trip(tmp_path: Path) -> None:
    model = plf._make_mlp(13, hidden_dims=[16, 8], dropout=0.0, arch="deep", layernorm=True, batchnorm=False)
    path = tmp_path / "legacy.pt"
    plf.save_model_pkg({
        "model_type": "mlp", "model": model,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": 13, "hidden_dims": [16, 8], "dropout": 0.0,
        "mlp_arch": "deep", "model_arch": "legacy_mlp", "model_config": {},
        "layernorm": True, "batchnorm": False,
    }, path)
    loaded = plf.load_model_pkg(path)
    assert loaded["model_arch"] == "legacy_mlp"
    assert loaded["model"](torch.randn(2, 13)).shape == (2,)
