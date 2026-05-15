# EDA Regression Failure Bucketing

这个项目用于处理 EDA/ICCAD 风格的 regression failure bucketing 任务：给定一批 Ibex RISC-V CPU 回归仿真的失败日志，把由同一根因 bug 导致的 case 尽量分到同一个 bucket。

正式预测程序是 `regr_fail_bucketing.py`，接口兼容赛题要求：

```bash
regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

## 当前推荐方法

当前推荐默认配置是：

```text
primary_signature + FixedDepthDrain + template quality weighting
+ TF-IDF/SVD64 + AgglomerativeClustering
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
--feature-level baseline
--normalizer v1
--line-mode default
--template-weighting quality
--svd-dim 64
--cluster-factor 0.875
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
  --feature-level baseline \
  --normalizer v1 \
  --line-mode default \
  --template-weighting quality \
  --svd-dim 64 \
  --cluster-factor 0.875 \
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
dataset, parser, cluster, feature_level, svd_dim, cluster_factor, token_weight_mode, token_weights,
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

## Non-DL Vector-Space Tuning

近期实验表明，直接堆更多结构化日志 token 容易提高 TNR 但伤害 TPR，导致同一 bug 被拆碎。当前更稳的非深度学习改进是：

```text
primary_signature + FixedDepthDrain
+ template quality weighting
+ TF-IDF -> SVD 64
+ AgglomerativeClustering with cluster_factor 0.875
```

关键点是保留 v1 normalizer 中的日志细节，但在 Drain 模板进入向量空间时进行质量加权：高权重保留 `UVM_FATAL`、`cosim mismatch`、`PC mismatch`、`register write data mismatch`、`synchronous trap` 等模板，跳过或弱化 `TEST FAILED`、`error seen in rtl_sim.log`、timeout setup、summary counter 等公共噪声模板。

```bash
.venv/bin/python regr_fail_bucketing.py \
  --input dataset/stage3_dataset_32bugs_640cases/input.csv \
  --output /tmp/stage3_quality_svd64_cf0875.csv \
  --k 32 \
  --svd-dim 64 \
  --cluster-factor 0.875 \
  --template-weighting quality
```

对应实验命令：

```bash
.venv/bin/python run_experiments.py \
  --python .venv/bin/python \
  --output-dir /private/tmp/non_dl_quality_svd64_cf0875 \
  --feature-level baseline \
  --svd-dim 64 \
  --cluster-factors 0.875 \
  --template-weighting quality
```

`--feature-level structured` 和 `--normalizer semantic` 作为 ablation 保留，包含 mismatch/event/register-class/op-pair 等确定性结构化特征和语义归一化；目前它们不作为推荐配置，因为本地验证中容易降低 TPR 或在 stage2 上丢失区分信号。

## 当前验证结果

以下结果用于记录当前工程整理时的主线状态。三类结果含义不同，需要分开看。

### Full Local Dataset Direct Evaluation

在完整本地 dataset 上直接聚类和评估：

| dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| first | 0.765119 | 0.741667 | 0.788571 |
| stage2 | 0.740608 | 0.715476 | 0.765741 |
| stage3 | 0.736150 | 0.849342 | 0.622959 |

`svd_dim=64, cluster_factor=0.75, feature_level=baseline` 的非 DL tuning 结果：

| dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| first | 0.774107 | 0.775000 | 0.773214 |
| stage2 | 0.756187 | 0.767262 | 0.745111 |
| stage3 | 0.756242 | 0.771053 | 0.741431 |

`svd_dim=64, cluster_factor=0.875, template_weighting=quality` 的 Drain-cleaning v2 结果：

| dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| first | 0.782083 | 0.791667 | 0.772500 |
| stage2 | 0.765667 | 0.816667 | 0.714667 |
| stage3 | 0.778823 | 0.847862 | 0.709783 |

### Half-Split Sanity Validation

`weight_mode=none, cluster_factor=1.0` 的 sanity summary：

| dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| first | 0.744196 | 0.681250 | 0.807143 |
| stage2 | 0.737854 | 0.681176 | 0.794533 |
| stage3 | 0.745656 | 0.733681 | 0.757631 |

`svd_dim=64, cluster_factor=0.75, feature_level=baseline` 的 seed=0 half-split sanity summary：

| dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| first | 0.762232 | 0.743750 | 0.780714 |
| stage2 | 0.743751 | 0.707217 | 0.780285 |
| stage3 | 0.762673 | 0.794792 | 0.730554 |

`svd_dim=64, cluster_factor=0.875, template_weighting=quality` 的 seed=0 half-split sanity summary：

| dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| first | 0.768661 | 0.743750 | 0.793571 |
| stage2 | 0.752593 | 0.757440 | 0.747745 |
| stage3 | 0.768617 | 0.814583 | 0.722651 |

同一配置的 5-seed half-split full validation summary：

| dataset | mean_BA | std_BA | mean_TPR | mean_TNR | runs |
|---|---:|---:|---:|---:|---:|
| first | 0.766589 | 0.015168 | 0.733750 | 0.799429 | 40 |
| stage2 | 0.747185 | 0.010046 | 0.750223 | 0.744147 | 40 |
| stage3 | 0.761379 | 0.019138 | 0.823333 | 0.699425 | 40 |

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
  --cluster-factors 0.875 1.0 1.25
```

```bash
.venv/bin/python run_half_split_experiments.py \
  --python .venv/bin/python \
  --datasets \
    dataset/first_batch_dataset \
    dataset/stage2_dataset_working \
    dataset/stage3_dataset_32bugs_640cases \
  --seeds 0 1 2 3 4 \
  --cluster-factors 0.875 1.0 1.25 \
  --weight-modes none repeat conservative blacklist \
  --output-dir /private/tmp/half_split_exp
```

half-split 实验会把每个规模的数据集按 `bug_id` 分层拆成 `part1` / `part2`，枚举三个规模的 train/validation 组合。validation 仍然按每个 benchmark part 独立聚类和评估，更接近正式赛题。

## Experimental: LLM Embedding Augmentation

赛题允许使用 LLM APIs 做 log understanding 或 embedding，但网络延迟计入 runtime，且评测只运行一次，所以 LLM 默认关闭。当前实现只把 LLM 作为可选 embedding 增强，不让它直接决定 bucket：

```text
compressed case document
-> official embedding endpoint
-> concat with deterministic TF-IDF/SVD64 vector
-> normalize
-> AgglomerativeClustering
```

压缩文档只包含 `primary_signature`、高信号 Drain templates、flag/count/structured failure hints，不会把完整 `sim.log` / `regr.log` 发给 LLM。

启用方式：

```bash
export LLM_MODEL_CONFIG="$(cat openai.yaml)"

.venv/bin/python regr_fail_bucketing.py \
  --input dataset/stage3_dataset_32bugs_640cases/input.csv \
  --output /tmp/stage3_llm_embedding.csv \
  --k 32 \
  --llm-mode embedding \
  --llm-weight 0.25 \
  --llm-cache-dir /tmp/regr_fail_llm_cache
```

如果 `LLM_MODEL_CONFIG` 缺失、`openai` 未安装、API 超时或调用失败，程序会 warning 并自动 fallback 到 deterministic baseline。调试时可以加 `--strict-llm` 让失败直接返回非零。

可用实验命令：

```bash
.venv/bin/python run_experiments.py \
  --python .venv/bin/python \
  --output-dir /tmp/llm_embedding_exp \
  --llm-mode embedding \
  --llm-weight 0.25
```

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
  --epochs 50 \
  --batch-size 8192 \
  --max-train-pairs 300000 \
  --architecture residual \
  --hidden-dims 512 512 256 256 128 \
  --dropout 0.25 \
  --negative-ratio 1.5 \
  --hard-positive-ratio 0.5 \
  --hard-negative-ratio 0.5 \
  --pos-weight-scale 1.2
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

当前 pairwise 特征 schema 为 v2，新增了 token Dice/containment、hard-positive 友好的结构化冲突特征，以及轻量 probability calibration。校准默认只影响 `pairwise_mlp` 后端：

```text
--pairwise-primary-floor 0.70
--pairwise-op-pair-floor 0.65
--pairwise-mismatch-floor 0.55
--pairwise-conflict-penalty 0.05
```

如果验证发现 TNR 下降，可以把这些 floor 调低，甚至设为 `0.0` 关闭。

Pairwise MLP 架构也是可切换的：

```text
--architecture plain      # 兼容最初的 2-layer MLP checkpoint
--architecture layernorm  # plain MLP + LayerNorm
--architecture residual   # 默认实验主线，更深的 residual MLP
```

`residual` 默认使用 `512 512 256 256 128`，重复维度会插入 residual block。旧模型 checkpoint 没有 `architecture` 字段时，推理会自动按 `plain` 加载，保持向后兼容。

如果 residual 过拟合，可以尝试：

```text
--architecture layernorm --hidden-dims 512 256 128 --dropout 0.30
```

### Current Pairwise Findings

截至当前实验，pairwise MLP 仍不能替换默认 baseline：

- `v1/v2` 在 `first_batch` 上能超过 baseline，说明 pairwise learning 有信号。
- `stage2/stage3` 上主要输在 TPR，同一 bug 的不同表现仍容易被拆开。
- calibration search 不能稳定解决问题：无校准时 TPR 高但 TNR 崩，floor-based calibration 又会压低 TPR。
- 下一轮重点改模型架构和训练目标，而不是继续手工调 calibration。

当前建议实验顺序：

1. `plain` 旧架构复现，作为对照。
2. `layernorm` 中等深度模型，检查 normalization 是否改善跨规模泛化。
3. `residual` 深模型，检查更大容量是否提升 stage3 TPR。
4. 对最好的架构再做 calibration search，而不是对所有模型都先调 calibration。

half-split 对照实验：

```bash
.venv/bin/python run_pairwise_mlp_half_split.py \
  --python .venv/bin/python \
  --seeds 0 \
  --output-dir /tmp/pairwise_mlp_exp \
  --device auto \
  --epochs 10 \
  --max-train-pairs 100000 \
  --architecture residual \
  --hidden-dims 512 512 256 256 128 \
  --hard-positive-ratio 0.5 \
  --pos-weight-scale 1.2
```

推荐下一轮架构 ablation：

```bash
# A. plain baseline, comparable with older checkpoints
.venv/bin/python run_pairwise_mlp_half_split.py \
  --python .venv/bin/python \
  --seeds 0 \
  --output-dir /tmp/pairwise_arch_plain \
  --device auto \
  --epochs 20 \
  --max-train-pairs 300000 \
  --batch-size 8192 \
  --architecture plain \
  --hidden-dims 256 128 \
  --dropout 0.20 \
  --pos-weight-scale 1.0

# B. layernorm medium-depth model
.venv/bin/python run_pairwise_mlp_half_split.py \
  --python .venv/bin/python \
  --seeds 0 \
  --output-dir /tmp/pairwise_arch_layernorm \
  --device auto \
  --epochs 20 \
  --max-train-pairs 300000 \
  --batch-size 8192 \
  --architecture layernorm \
  --hidden-dims 512 256 128 \
  --dropout 0.30 \
  --pos-weight-scale 1.0

# C. residual deep model
.venv/bin/python run_pairwise_mlp_half_split.py \
  --python .venv/bin/python \
  --seeds 0 \
  --output-dir /tmp/pairwise_arch_residual \
  --device auto \
  --epochs 20 \
  --max-train-pairs 300000 \
  --batch-size 8192 \
  --architecture residual \
  --hidden-dims 512 512 256 256 128 \
  --dropout 0.25 \
  --pos-weight-scale 1.0
```

搜索 pairwise probability calibration：

```bash
.venv/bin/python run_pairwise_calibration_search.py \
  --model /tmp/pairwise_mlp_v2.pt \
  --config /tmp/pairwise_config_v2.json \
  --seed 0 \
  --combo 0 \
  --output-dir /tmp/pairwise_calib_search \
  --device auto
```

脚本会输出：

```text
/tmp/pairwise_calib_search/calibration_results.csv
/tmp/pairwise_calib_search/calibration_summary.csv
/tmp/pairwise_calib_search/best_calibration.json
```

然后可以用搜索到的参数复跑 half-split：

```bash
.venv/bin/python run_pairwise_mlp_half_split.py \
  --python .venv/bin/python \
  --seeds 0 \
  --output-dir /tmp/pairwise_mlp_calibrated_exp \
  --device auto \
  --epochs 10 \
  --max-train-pairs 100000 \
  --calibration-json /tmp/pairwise_calib_search/best_calibration.json
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

1. Run architecture ablation: `plain` vs `layernorm` vs `residual`, prioritizing stage3 TPR and mean BA.
2. Add focal loss or class-balanced loss if residual still over-merges or under-merges.
3. Add stronger hard negatives: same mismatch type / similar primary type but different bug.
4. Add postprocess `split_mixed` for large mixed clusters using `primary_signature`.
5. Keep pairwise MLP experimental unless multi-seed validation shows stable gains over `drain + agglomerative`.
