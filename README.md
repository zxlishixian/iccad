# EDA Regression Failure Bucketing

这个项目用于处理 EDA/ICCAD 风格的 regression failure bucketing 任务：给定一批 Ibex RISC-V CPU 回归仿真的失败日志，把由同一根因 bug 导致的 case 尽量分到同一个 bucket。

正式预测程序是 `regr_fail_bucketing.py`，接口兼容赛题要求：

```bash
regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

## 当前推荐方法

当前推荐默认配置是：

```text
primary_signature + FixedDepthDrain + AgglomerativeClustering
```

默认主线明确不做这些事：

- 不加载 supervised token weights
- 不使用 `trace.log`
- 不调用 LLM
- 不读取 `gold.csv` 或 `meta.csv`

`primary_signature` 是从 `sim.log` / `regr.log` 中确定性抽取的失败摘要 token，是默认 baseline 的一部分，不需要 gold label。

默认参数已经固定为：

```text
--parser drain
--cluster agglomerative
--cluster-factor 1.0
--token-weight-mode none
```

推荐直接运行：

```bash
.venv/bin/python regr_fail_bucketing.py \
  --input dataset/stage3_dataset_32bugs_640cases/input.csv \
  --output /tmp/stage3_out.csv \
  --k 32
```

等价的完整显式命令：

```bash
.venv/bin/python regr_fail_bucketing.py \
  --input dataset/stage3_dataset_32bugs_640cases/input.csv \
  --output /tmp/stage3_out.csv \
  --k 32 \
  --parser drain \
  --cluster agglomerative \
  --cluster-factor 1.0 \
  --token-weight-mode none
```

## 项目结构

- `B_20260212.pdf`：赛题说明 PDF，定义输入输出、日志类型、benchmark 组织、运行时间限制和提交接口。
- `dataset/`：本地样例数据集，包含三个规模：
  - `first_batch_dataset/`：80 cases，8 bugs
  - `stage2_dataset_working/`：240 cases，16 bugs
  - `stage3_dataset_32bugs_640cases/`：640 cases，32 bugs
- `regr_fail_bucketing.py`：正式预测脚本。只读取 `input.csv` 指向的 `sim.log` / `regr.log`。
- `regr_fail_bucketing`：轻量 wrapper，优先使用 `.venv/bin/python`。
- `run_experiments.py`：主要 baseline 实验脚本，默认只跑当前推荐主线。
- `error_analysis.py`：对比 `gold.csv` 和预测输出，分析 fragmentation、purity、FN/FP。
- `train_token_weights.py`：experimental/research utility，用 `gold.csv` 训练 token weights。
- `run_supervised_experiments.py`：experimental leave-one-dataset-out token weighting 实验。
- `run_half_split_experiments.py`：experimental half-split cross-scale validation 实验。

每个 bundled benchmark 通常包含：

```text
input.csv
gold.csv
meta.csv
cases/case_xxxxxx/sim.log
cases/case_xxxxxx/regr.log
cases/case_xxxxxx/trace.log
```

`input.csv` 实际列名为：

```csv
trace_log,sim_log,regr_log
```

预测输出 CSV 只需要一列：

```csv
bucket
bucket_000
bucket_001
...
```

## 运行主线实验

默认只跑当前主线配置，也就是三个 dataset 各跑一次：

```bash
.venv/bin/python run_experiments.py \
  --python .venv/bin/python \
  --output-dir /private/tmp/base_exp
```

输出字段：

```text
dataset, parser, cluster, cluster_factor, token_weight_mode, token_weights,
cases, k, num_pred_clusters, BA, TPR, TNR, runtime_sec
```

如果要显式做 ablation，可以传入更多组合：

```bash
.venv/bin/python run_experiments.py \
  --python .venv/bin/python \
  --output-dir /private/tmp/ablation_exp \
  --parsers simple drain \
  --clusters kmeans agglomerative hdbscan \
  --cluster-factors 1.0 1.25 1.5
```

## 当前验证结果

以下结果用于记录当前工程整理时的主线状态。三类结果含义不同，需要分开看。

### Full Local Dataset Direct Evaluation

在完整本地 dataset 上直接聚类和评估：

| dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| first | 0.765119 | 0.741667 | 0.788571 |
| stage2 | 0.740608 | 0.715476 | 0.765741 |
| stage3 | 0.736150 | 0.849342 | 0.622959 |

### Half-Split Sanity Validation

`weight_mode=none, cluster_factor=1.0` 的 sanity summary：

| dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| first | 0.744196 | 0.681250 | 0.807143 |
| stage2 | 0.737854 | 0.681176 | 0.794533 |
| stage3 | 0.745656 | 0.733681 | 0.757631 |

### Partial Full Half-Split Trend

`weight_mode=none, cluster_factor=1.0` 的 partial full experiment trend：

| dataset | mean_BA |
|---|---:|
| first | 0.738973 |
| stage2 | 0.734631 |
| stage3 | 0.747363 |

## Supervised Token Weighting

Supervised token weighting 已经实现并评估过。`repeat` / `conservative` / `blacklist` modes 在 half-split cross-scale validation 中没有稳定超过 no-weight baseline。因此 token weights 默认关闭，并作为 experimental/research 功能保留。

训练 token weights 时才允许读取 `gold.csv`：

```bash
.venv/bin/python train_token_weights.py \
  --datasets dataset/first_batch_dataset dataset/stage2_dataset_working dataset/stage3_dataset_32bugs_640cases \
  --output /tmp/token_weights.json
```

正式预测时如果要使用 token weights，必须显式启用：

```bash
.venv/bin/python regr_fail_bucketing.py \
  --input dataset/stage3_dataset_32bugs_640cases/input.csv \
  --output /tmp/stage3_weighted_out.csv \
  --k 32 \
  --token-weights /tmp/token_weights.json \
  --token-weight-mode repeat
```

如果传了 `--token-weights` 但仍使用 `--token-weight-mode none`，正式预测会忽略权重并打印 warning。

## Experimental Utilities

这些脚本保留用于后续研究，不是默认主线：

```bash
.venv/bin/python run_supervised_experiments.py \
  --python .venv/bin/python \
  --output-dir /tmp/supervised_exp \
  --cluster-factors 1.0 1.25 1.5
```

```bash
.venv/bin/python run_half_split_experiments.py \
  --python .venv/bin/python \
  --datasets \
    dataset/first_batch_dataset \
    dataset/stage2_dataset_working \
    dataset/stage3_dataset_32bugs_640cases \
  --seeds 0 1 2 3 4 \
  --cluster-factors 1.0 1.25 1.5 \
  --weight-modes none repeat conservative blacklist \
  --output-dir /private/tmp/half_split_exp
```

half-split 实验会把每个规模的数据集按 `bug_id` 分层拆成 `part1` / `part2`，枚举三个规模的 train/validation 组合。validation 仍然按每个 benchmark part 独立聚类和评估，更接近正式赛题。

## Experimental: Pairwise Same-Bug MLP

新增的 pairwise MLP 后端直接学习：

```text
P(case_i 和 case_j 是否属于同一个 bug bucket)
```

这个方向和官方 pairwise balanced accuracy 更对齐，但当前仍是 experimental backend：

- 只使用 `sim.log` 和 `regr.log`
- 训练阶段读取本地 `gold.csv`
- 推理阶段不读取 `gold.csv` / `meta.csv`
- 默认 baseline 不依赖 PyTorch
- 必须显式传 `--cluster pairwise_mlp` 才会 import/use torch
- 训练可用 GPU，推理支持 CPU

训练模型：

```bash
.venv/bin/python train_pairwise_mlp.py \
  --datasets dataset/first_batch_dataset dataset/stage2_dataset_working dataset/stage3_dataset_32bugs_640cases \
  --output models/pairwise_mlp.pt \
  --config-output models/pairwise_config.json \
  --device auto \
  --epochs 50
```

用 pairwise MLP 推理：

```bash
.venv/bin/python regr_fail_bucketing.py \
  --input dataset/stage3_dataset_32bugs_640cases/input.csv \
  --output /tmp/stage3_pairwise.csv \
  --k 32 \
  --cluster pairwise_mlp \
  --pairwise-model models/pairwise_mlp.pt \
  --pairwise-config models/pairwise_config.json \
  --pairwise-device cpu
```

half-split 对照实验：

```bash
.venv/bin/python run_pairwise_mlp_half_split.py \
  --python .venv/bin/python \
  --seeds 0 \
  --output-dir /tmp/pairwise_mlp_exp \
  --device auto \
  --epochs 10 \
  --max-train-pairs 100000
```

This backend is experimental. The default submitted baseline remains `drain + agglomerative` unless validation shows stable gains.

## Error Analysis

当 TNR 高但 TPR 低时，通常说明不同 bug 容易分开，但同一 bug 的多种表现被拆碎。可以用：

```bash
.venv/bin/python error_analysis.py \
  --gold dataset/stage3_dataset_32bugs_640cases/gold.csv \
  --pred /tmp/stage3_out.csv \
  --top 12 \
  --out /tmp/stage3_error_analysis.md
```

## 依赖

推荐使用本地虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

如果 `scikit-learn` 不存在，正式预测会打印 warning，并回退到标准库 hashing TF-IDF + k-means fallback。

## Next Steps

1. Add postprocess `split_mixed` for large mixed clusters using `primary_signature`.
2. Add lightweight trace tail features instead of full trace processing.
3. Keep supervised token weighting experimental unless a validation setup shows stable gains.
