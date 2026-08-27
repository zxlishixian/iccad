#!/usr/bin/env python3
"""Train one experimental rich pairwise neural architecture.

This compatibility front-end fixes the current-best feature, sampling, loss,
validation, and clustering protocol. Only ``--model-arch`` varies.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an experimental rich pairwise classifier")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config-output", type=Path, required=True)
    p.add_argument("--model-arch", choices=("res_mlp", "gated_mlp", "ft_transformer"), default="res_mlp")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-train-pairs", type=int, default=300000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--gate-reg", type=float, default=1e-4)
    p.add_argument("--ft-d-token", type=int, default=64)
    p.add_argument("--ft-layers", type=int, default=2)
    p.add_argument("--ft-heads", type=int, default=4)
    p.add_argument("--ft-dropout", type=float, default=0.1)
    p.add_argument("--ft-attention-dropout", type=float, default=0.1)
    p.add_argument("--ft-ffn-mult", type=int, default=2)
    p.add_argument("--ft-max-tokens", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop-patience", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work = args.output.parent / f".{args.output.stem}_artifacts"
    tag = args.output.stem
    datasets = [path if path.is_absolute() else (ROOT / path).resolve() for path in args.datasets]
    cmd = [
        sys.executable, str(ROOT / "train_pairwise_llm.py"),
        "--datasets", *map(str, datasets),
        "--output-dir", str(work),
        "--model-type", "mlp", "--model-tag", tag,
        "--feature-mode", "llm_dual_struct_det_summary", "--llm-reduce-dim", "64",
        "--mlp-arch", "residual", "--model-arch", args.model_arch,
        "--loss", "focal", "--focal-gamma", "2.0", "--focal-alpha", "auto",
        "--svd-dim", "64", "--seed", str(args.seed), "--combo", str(args.combo),
        "--random-state", str(args.seed), "--device", args.device,
        "--epochs", str(args.epochs), "--max-train-pairs", str(args.max_train_pairs),
        "--batch-size", str(args.batch_size), "--dropout", str(args.dropout),
        "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
        "--negative-ratio", "2.0", "--hard-negative-ratio", "0.5",
        "--hard-positive-ratio", "0.5", "--early-stop-patience", str(args.early_stop_patience),
        "--llm-cache-dir", str(args.llm_cache_dir), "--gate-reg", str(args.gate_reg),
        "--ft-d-token", str(args.ft_d_token), "--ft-layers", str(args.ft_layers),
        "--ft-heads", str(args.ft_heads), "--ft-dropout", str(args.ft_dropout),
        "--ft-attention-dropout", str(args.ft_attention_dropout),
        "--ft-ffn-mult", str(args.ft_ffn_mult), "--ft-max-tokens", str(args.ft_max_tokens),
    ]
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode:
        return proc.returncode

    source = work / f"model_seed{args.seed}_combo{args.combo:03b}_{tag}.pt"
    config_source = work / f"config_seed{args.seed}_combo{args.combo:03b}_{tag}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.output)
    for suffix in (".preproc.pkl", ".scaler.pkl"):
        side = source.with_suffix(suffix)
        if side.exists():
            shutil.copy2(side, args.output.with_suffix(suffix))
    config = json.loads(config_source.read_text(encoding="utf-8"))
    config["model_path"] = str(args.output)
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
