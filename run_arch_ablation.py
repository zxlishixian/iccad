#!/usr/bin/env python3
"""Strict classifier-architecture ablation for the current rich pair features."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import official_style_features as osf
import pairwise_llm_features as plf
from run_experiments import pairwise_scores, read_gold
from run_half_split_experiments import opposite_part, part_for_bit, stratified_half_split
from run_input_signal_experiments import ENSEMBLE_MODEL_TYPES, ENSEMBLE_WEIGHTS, _model_ext

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    Path("old_fake_dataset/first_batch_dataset"),
    Path("old_fake_dataset/stage2_dataset_working"),
    Path("old_fake_dataset/stage3_dataset_32bugs_640cases"),
]


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def temperature(prob: np.ndarray, value: float) -> np.ndarray:
    clipped = np.clip(prob.astype(np.float64), 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped)) / max(float(value), 1e-6)
    out = 1.0 / (1.0 + np.exp(-logits))
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def model_path(root: Path, seed: int, combo: int, arch: str) -> Path:
    return root / arch / f"model_seed{seed}_combo{combo:03b}_{arch}.pt"


def config_path(root: Path, seed: int, combo: int, arch: str) -> Path:
    return root / arch / f"config_seed{seed}_combo{combo:03b}_{arch}.json"


def ensemble_paths(root: Path, seed: int, combo: int) -> list[Path]:
    paths = [root / f"model_seed{seed}_combo{combo:03b}_{kind}.{_model_ext(kind)}" for kind in ENSEMBLE_MODEL_TYPES]
    return paths if all(path.exists() for path in paths) else []


def build_val_parts(datasets: Sequence[Path], seed: int, combo: int, split_root: Path) -> list[dict]:
    output = []
    for idx, dataset in enumerate(datasets):
        split = stratified_half_split(dataset, seed, split_root)
        train_part = part_for_bit((combo >> idx) & 1)
        info = dict(split[opposite_part(train_part)])
        info["dataset"] = dataset.name
        output.append(info)
    return output


def write_prediction(path: Path, input_csv: Path, labels: Sequence[int]) -> None:
    cases = osf.read_cases(input_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["Case", "bucket"])
        for case, label in zip(cases, labels):
            writer.writerow([case, f"bucket_{int(label):03d}"])


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_arch"]), str(row["eval_mode"]), str(row["dataset"]))].append(row)
    output = []
    for (arch, mode, dataset), values in sorted(groups.items()):
        bas = [float(x["BA"]) for x in values]
        output.append({
            "model_arch": arch, "eval_mode": mode, "dataset": dataset,
            "mean_BA": statistics.mean(bas),
            "std_BA": statistics.stdev(bas) if len(bas) > 1 else 0.0,
            "mean_TPR": statistics.mean(float(x["TPR"]) for x in values),
            "mean_TNR": statistics.mean(float(x["TNR"]) for x in values),
            "mean_runtime_sec": statistics.mean(float(x["runtime_sec"]) for x in values),
            "num_runs": len(values),
        })
    return output


def print_summary(summary: Sequence[dict]) -> None:
    print("| architecture | eval mode | dataset | BA | std | TPR | TNR | runtime | runs |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        print(f"| {row['model_arch']} | {row['eval_mode']} | {row['dataset']} | {float(row['mean_BA']):.4f} | {float(row['std_BA']):.4f} | {float(row['mean_TPR']):.4f} | {float(row['mean_TNR']):.4f} | {float(row['mean_runtime_sec']):.2f} | {row['num_runs']} |")

    by_config: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in summary:
        by_config[(str(row["model_arch"]), str(row["eval_mode"]))][str(row["dataset"])] = row
    ranked = []
    for key, datasets in by_config.items():
        mean_ba = statistics.mean(float(x["mean_BA"]) for x in datasets.values())
        ranked.append((mean_ba, key, datasets))
    ranked.sort(reverse=True)
    if not ranked:
        return
    best_mean, best_key, best_ds = ranked[0]
    print(f"\nBest mean over datasets: {best_key[0]} / {best_key[1]} = {best_mean:.4f}")
    stage3 = best_ds.get("stage3_dataset_32bugs_640cases", {})
    if stage3:
        print(f"Best-config stage3 BA: {float(stage3['mean_BA']):.4f}")
    baseline = by_config.get(("res_mlp", best_key[1]), {})
    first_name = "first_batch_dataset"
    if first_name in best_ds and first_name in baseline:
        print(f"Worst first_batch drop vs res_mlp: {float(best_ds[first_name]['mean_BA'])-float(baseline[first_name]['mean_BA']):+.4f}")
    deltas_tnr=[]; deltas_tpr=[]
    for dataset, row in best_ds.items():
        if dataset in baseline:
            deltas_tnr.append(float(row["mean_TNR"])-float(baseline[dataset]["mean_TNR"]))
            deltas_tpr.append(float(row["mean_TPR"])-float(baseline[dataset]["mean_TPR"]))
    if deltas_tnr:
        print(f"Mean TNR improvement vs res_mlp: {statistics.mean(deltas_tnr):+.4f}")
        print(f"Mean TPR change vs res_mlp: {statistics.mean(deltas_tpr):+.4f}")
    base_rank = next((x for x in ranked if x[1] == ("res_mlp", best_key[1])), None)
    if base_rank:
        gain = best_mean - base_rank[0]
        print(f"Mean BA gain vs res_mlp: {gain:+.4f} ({'significant' if gain >= 0.003 else 'not significant'})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise classifier architecture ablation")
    p.add_argument("--python", default="/home/lishixian/miniforge3/envs/collab-overcooked/bin/python")
    p.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--model-arches", nargs="+", choices=("res_mlp", "gated_mlp", "ft_transformer"), default=["res_mlp", "gated_mlp", "ft_transformer"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--combo", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-train-pairs", type=int, default=300000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--ensemble-model-dir", type=Path, default=Path("/tmp/pairwise_llm_exp_full/models"))
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rich-temperature", type=float, default=1.15)
    p.add_argument("--ensemble-temperature", type=float, default=1.0)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--gate-reg", type=float, default=1e-4)
    p.add_argument("--ft-d-token", type=int, default=64)
    p.add_argument("--ft-layers", type=int, default=2)
    p.add_argument("--ft-heads", type=int, default=4)
    p.add_argument("--ft-dropout", type=float, default=0.1)
    p.add_argument("--ft-max-tokens", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [path if path.is_absolute() else (ROOT/path).resolve() for path in args.datasets]
    results=[]
    for seed in args.seeds:
        split_root = args.output_dir / "splits" / f"seed_{seed}"
        val_parts = build_val_parts(datasets, seed, args.combo, split_root)
        for arch in args.model_arches:
            out_dir = args.output_dir / "models" / arch
            path = model_path(args.output_dir / "models", seed, args.combo, arch)
            cfg_path = config_path(args.output_dir / "models", seed, args.combo, arch)
            if not args.skip_train:
                cmd = [
                    args.python, str(ROOT/"train_pairwise_llm.py"),
                    "--datasets", *map(str,datasets), "--output-dir", str(out_dir),
                    "--model-type", "mlp", "--model-tag", arch,
                    "--feature-mode", "llm_dual_struct_det_summary", "--llm-reduce-dim", "64",
                    "--mlp-arch", "residual", "--model-arch", arch,
                    "--loss", "focal", "--focal-gamma", "2.0", "--focal-alpha", "auto",
                    "--svd-dim", "64", "--seed", str(seed), "--combo", str(args.combo),
                    "--random-state", str(seed), "--device", args.device,
                    "--epochs", str(args.epochs), "--max-train-pairs", str(args.max_train_pairs),
                    "--batch-size", str(args.batch_size), "--dropout", "0.2",
                    "--negative-ratio", "2.0", "--hard-negative-ratio", "0.5",
                    "--hard-positive-ratio", "0.5", "--early-stop-patience", "8",
                    "--llm-cache-dir", str(args.llm_cache_dir), "--gate-reg", str(args.gate_reg),
                    "--ft-d-token", str(args.ft_d_token), "--ft-layers", str(args.ft_layers),
                    "--ft-heads", str(args.ft_heads), "--ft-dropout", str(args.ft_dropout),
                    "--ft-max-tokens", str(args.ft_max_tokens),
                ]
                started=time.perf_counter(); proc=subprocess.run(cmd,cwd=ROOT,text=True)
                if proc.returncode: raise RuntimeError(f"training failed: arch={arch} seed={seed}")
                train_runtime=time.perf_counter()-started
            else:
                train_runtime=0.0
            config=json.loads(cfg_path.read_text(encoding="utf-8"))
            pkg=plf.load_model_pkg(path); pkg["device"]="cpu"
            ens_paths=ensemble_paths(args.ensemble_model_dir,seed,args.combo)
            ens_pkgs=[plf.load_model_pkg(x) for x in ens_paths]
            rich_args=plf._make_llm_args(llm_mode="embedding",llm_doc_style="features",llm_cache_dir=args.llm_cache_dir,svd_dim=64,llm_dual=True)
            ens_args=plf._make_llm_args(llm_mode="embedding",llm_doc_style="features",llm_cache_dir=args.llm_cache_dir,svd_dim=64)
            for part in val_parts:
                started=time.perf_counter()
                features,_=plf.build_llm_case_features(part["input"],svd_dim=64,llm_args=rich_args)
                p_model=plf.predict_probability_matrix_sklearn(pkg,features)
                modes=[("model_only",p_model)]
                if ens_pkgs:
                    ens_features,_=plf.build_llm_case_features(part["input"],svd_dim=64,llm_args=ens_args)
                    p_ens=plf.predict_probability_matrix_ensemble(ens_pkgs,list(ENSEMBLE_WEIGHTS),ens_features)
                    fused=args.alpha*temperature(p_model,args.rich_temperature)+(1.0-args.alpha)*temperature(p_ens,args.ensemble_temperature)
                    modes.append(("calibrated_blend",fused.astype(np.float32)))
                for mode,prob in modes:
                    labels=plf.cluster_from_probability(prob,part["k"])
                    gold=read_gold(part["gold"]); pred=[f"bucket_{x:03d}" for x in labels]
                    ba,tpr,tnr=pairwise_scores(gold,pred)
                    pred_path=args.output_dir/"preds"/f"seed{seed}_{arch}_{mode}_{part['dataset']}.csv"
                    write_prediction(pred_path,part["input"],labels)
                    results.append({
                        "seed":seed,"dataset":part["dataset"],"model_arch":arch,"eval_mode":mode,
                        "BA":ba,"TPR":tpr,"TNR":tnr,"runtime_sec":time.perf_counter()-started,
                        "train_runtime_sec":train_runtime,"best_epoch":config.get("best_epoch",0),
                        "best_val_BA":config.get("best_val_BA",-1),"model_path":str(path),"pred_path":str(pred_path),
                    })
            write_csv(args.output_dir/"results.partial.csv",results,list(results[0]))
    fields=["seed","dataset","model_arch","eval_mode","BA","TPR","TNR","runtime_sec","train_runtime_sec","best_epoch","best_val_BA","model_path","pred_path"]
    write_csv(args.output_dir/"results.csv",results,fields)
    summary=summarize(results)
    summary_fields=["model_arch","eval_mode","dataset","mean_BA","std_BA","mean_TPR","mean_TNR","mean_runtime_sec","num_runs"]
    write_csv(args.output_dir/"summary.csv",summary,summary_fields)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
