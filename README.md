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
-> concat with deterministic TF-IDF/SVD64 vector, or similarity-level fusion
-> AgglomerativeClustering
```

压缩文档只包含 `primary_signature`、高信号 Drain templates、flag/count/structured failure hints，不会把完整 `sim.log` / `regr.log` 发给 LLM。

### 环境变量配置

`LLM_MODEL_CONFIG` 必须包含 YAML 内容，**不是文件路径**：

```bash
# 正确：把 YAML 内容赋给环境变量
export LLM_MODEL_CONFIG="$(cat /path/to/config.yaml)"

# 错误：这样会把文件路径当 YAML 内容，导致 LLM 静默不生效
export LLM_MODEL_CONFIG=/path/to/config.yaml
```

如果程序检测到环境变量值看起来像文件路径（以 `/`、`./`、`~` 开头，或以 `.yaml`/`.yml` 结尾），会打印 warning 提示修正。

config.yaml 示例（兼容 OpenAI-compatible 接口，包括本地 nomic-embed-text-v1.5 服务）：

```yaml
embedding:
  model_name: "nomic-embed-text-v1.5"
  config:
    api_key: "dummy"
    base_url: "http://127.0.0.1:8001/v1"
    client_library: "openai.AsyncOpenAI"
    model: "nomic-embed-text-v1.5"
```

### llm-mode 选项

| mode | 行为 |
|---|---|
| `none` (默认) | 永不使用 LLM |
| `embedding` | 强制尝试 LLM，失败则 warning + fallback 到 deterministic baseline |
| `auto` | 检测到有效 `LLM_MODEL_CONFIG` 时启用 embedding，否则走 deterministic |

`--strict-llm` 可以让 LLM 失败时直接返回非零退出码（调试用）。

### Fusion / Document Style Ablations

当前稳定配置仍是 `--llm-fusion concat --llm-weight 4.0 --llm-doc-style features`。新增实验开关用于验证 LLM embedding 的使用方式：

```text
--llm-fusion concat|similarity
--llm-alpha <float>
--llm-doc-style features|summary
```

`concat` 会把 deterministic SVD64 vector 和 LLM embedding 拼接后归一化。`similarity` 会分别计算 deterministic cosine similarity 和 LLM cosine similarity，再融合：

```text
S = alpha * S_deterministic + (1 - alpha) * S_llm
distance = 1 - S
```

`features` 使用当前 feature dump 风格文档；`summary` 使用更自然的 failure summary 文档，方便测试 embedding 模型是否更容易捕捉语义。

### 推荐配置与验证结果

经过权重扫描（w=0.1 到 5.0），推荐 `--llm-weight 4.0`。5-seed half-split 验证结果（weight_mode=none, cluster_factor=0.875）：

| dataset | BA | TPR | TNR | vs Baseline ΔBA |
|---|---:|---:|---:|---:|
| first_batch | 0.7771 | 0.7488 | 0.8054 | +1.0pp |
| stage2 | 0.7579 | 0.7276 | 0.7881 | +1.1pp |
| stage3 | 0.7683 | 0.8139 | 0.7228 | +0.7pp |

确定性 baseline 对照：

| dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| first_batch | 0.7666 | 0.7338 | 0.7994 |
| stage2 | 0.7472 | 0.7502 | 0.7441 |
| stage3 | 0.7614 | 0.8233 | 0.6994 |

关键发现：
- **三个数据集 BA 全面提升**，stage2 提升最大 (+1.1pp)
- **stage2 TNR 从 0.744 提升到 0.788 (+4.4pp)**，假阳性显著减少 — 这是最关键的改善
- stage3 TNR 也提升 2.3pp，从 0.699 到 0.723
- TPR 在 stage2/stage3 上有小幅 trade-off（-2.3pp / -0.9pp），在可接受范围内

### 使用方式

启用 LLM embedding 增强（推荐 w=4.0）：

```bash
export LLM_MODEL_CONFIG="$(cat /path/to/config.yaml)"

.venv/bin/python regr_fail_bucketing.py \
  --input dataset/stage3_dataset_32bugs_640cases/input.csv \
  --output /tmp/stage3_llm_embedding.csv \
  --k 32 \
  --llm-mode embedding \
  --llm-weight 4.0 \
  --llm-cache-dir /tmp/regr_fail_llm_cache
```

如果 `LLM_MODEL_CONFIG` 缺失、`openai` 未安装、API 超时或调用失败，程序会 warning 并自动 fallback 到 deterministic baseline。调试时可以加 `--strict-llm` 让失败直接返回非零。

可用实验命令：

```bash
.venv/bin/python run_experiments.py \
  --python .venv/bin/python \
  --output-dir /tmp/llm_embedding_exp \
  --llm-mode embedding \
  --llm-weight 4.0 \
  --llm-fusion concat
```

half-split 交叉验证：

```bash
.venv/bin/python run_half_split_experiments.py \
  --python .venv/bin/python \
  --seeds 0 1 2 3 4 \
  --output-dir /tmp/llm_half_split_exp \
  --llm-mode embedding \
  --llm-weight 4.0 \
  --llm-fusion concat
```

相似度融合实验示例：

```bash
.venv/bin/python run_half_split_experiments.py \
  --python .venv/bin/python \
  --seeds 0 1 2 \
  --output-dir /tmp/llm_similarity_alpha075 \
  --llm-mode embedding \
  --llm-fusion similarity \
  --llm-alpha 0.75 \
  --llm-doc-style features
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

## Experimental: Pairwise LLM Learning

新增 lightweight pairwise learning 路线，用固定维度的 pairwise 特征 + 三种 backend 做 ablation。
与旧的 Pairwise Same-Bug MLP 不同，这条路线：

- 特征维度固定（21 维），不拼接高维 dense vector
- 同时使用 deterministic TF-IDF/SVD64 和 LLM embedding 计算 pairwise 特征
- 支持三种 backend：logistic regression、gradient boosting (GBDT)、small MLP
- 训练读取 gold.csv，推理不读取 gold/meta
- 必须显式运行实验脚本，默认 baseline 不变

### Pairwise Feature Schema

每个 case pair 提取 21 维固定特征：

| 特征 | 说明 |
|------|------|
| `det_cosine` | deterministic SVD64 向量的 cosine similarity |
| `llm_cosine` | LLM embedding 向量的 cosine similarity |
| `abs_det_llm_diff` | abs(det_cosine - llm_cosine) |
| `det_euclidean` | deterministic 向量欧氏距离 |
| `llm_euclidean` | LLM 向量欧氏距离 |
| `token_jaccard` | token set Jaccard |
| `primary_token_jaccard` | PRIMARY token Jaccard |
| `sim_token_jaccard` | sim log token Jaccard |
| `regr_token_jaccard` | regr log token Jaccard |
| `same_primary_signature` | 是否相同 primary signature |
| `same_primary_type` | 是否相同 primary type |
| `same_mismatch_type` | 是否相同 mismatch type |
| `same_op_pair` | 是否相同 op_pair |
| `same_fatal_file` | 是否相同 fatal file |
| `same_failed_reason` | 是否相同 failed reason |
| `same_has_uvm_fatal` | 是否都有 UVM_FATAL |
| `same_has_uvm_error` | 是否都有 UVM_ERROR |
| `same_has_regr_mismatch` | 是否都有 mismatch |
| `abs_num_tokens_diff_log` | log(1+|num_tokens_i - num_tokens_j|) |
| `min_num_tokens_log` | log(1+min(num_tokens_i, num_tokens_j)) |
| `max_num_tokens_log` | log(1+max(num_tokens_i, num_tokens_j)) |

### Three Backends

| Backend | 实现 | 优点 |
|---------|------|------|
| **logistic** | sklearn LogisticRegression + StandardScaler | 可解释，稳定 |
| **gbdt** | sklearn HistGradientBoostingClassifier | 非线性，tabular 友好 |
| **mlp** | PyTorch MLP (21→64→32→1, GELU, LayerNorm, Dropout 0.1) | 深度学习 |

### 5-Seed Half-Split Results

With `doc_style=features`, `llm_weight=4.0`, `cluster_factor=0.875`:

| Method | first_batch BA | stage2 BA | stage3 BA | Mean BA |
|---|---:|---:|---:|---:|
| deterministic | 0.7671 | 0.7445 | 0.7603 | 0.7573 |
| llm_concat_features | 0.7849 | 0.7689 | 0.7678 | 0.7739 |
| pairwise_logistic | 0.7592 | 0.8085 | 0.7918 | 0.7865 |
| **pairwise_gbdt** | 0.7522 | **0.8310** | **0.8039** | **0.7957** |
| pairwise_mlp | 0.7572 | 0.8278 | 0.7969 | 0.7940 |

### TPR/TNR Detail

| Method | first_batch | | stage2 | | stage3 | |
|---|---|---|---|---|---|---|
| | TPR | TNR | TPR | TNR | TPR | TNR |
| deterministic | 0.720 | 0.814 | 0.738 | 0.751 | 0.822 | 0.699 |
| llm_concat_features | 0.748 | 0.822 | 0.728 | 0.810 | 0.814 | 0.722 |
| pairwise_gbdt | 0.708 | 0.797 | 0.785 | 0.877 | 0.793 | 0.815 |
| pairwise_logistic | 0.725 | 0.793 | 0.744 | 0.873 | 0.775 | 0.809 |
| pairwise_mlp | 0.718 | 0.797 | 0.772 | 0.884 | 0.784 | 0.810 |

### Key Findings

1. **pairwise GBDT 综合最优**：mean BA=0.7957，比 concat baseline (+2.18pp) 显著提升。stage2/stage3 上 BA 大幅领先。
2. **所有 pairwise 方法都在 first_batch 上低于 concat baseline**：小数据集 (80 cases, 8 bugs, ~40 train) 上 pairwise 训练的正样本对太少，泛化不足。
3. **pairwise 方法普遍提高 TNR 但牺牲 TPR**：stage2 上 TNR 提升约 12pp (0.81→0.88)，但 TPR 从 0.81 降为 0.79。
4. **MLP 不是必须的**：logistic 和 GBDT 已经达到或超过 MLP 的水平。在小特征 (21 dims) 下，tree-based 模型和线性模型的泛化更好。
5. **建议保留为 experimental**：pairwise learning 在大数据集上有明确优势，但小数据集退化和 TPR 损失需要进一步工作（如 few-shot transfer、cross-dataset training、更丰富的 pairwise 特征）。

### Reproduce

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

# Full 5-seed experiment
python run_pairwise_llm_half_split.py \
  --python /path/to/venv/bin/python \
  --seeds 0 1 2 3 4 \
  --methods logistic gbdt mlp \
  --baselines deterministic llm_concat_features \
  --output-dir /tmp/pairwise_llm_exp \
  --llm-doc-style features \
  --device auto \
  --epochs 60

# Train a single model
python train_pairwise_llm.py \
  --output-dir /tmp/pairwise_llm_model \
  --model-type gbdt \
  --llm-doc-style features \
  --seed 0
```

This backend is experimental. The default submitted baseline remains `drain + agglomerative`.

### Ensemble: Soft Voting

Fuses logistic + gbdt + mlp probability matrices via weighted averaging, then clusters with `AgglomerativeClustering(precomputed, average, k)`.

**Fusion modes:**
- `prob_average`: P_ens = w_L * P_L + w_G * P_G + w_M * P_M
- `logit_average`: P_ens = sigmoid(w_L * logit(P_L) + w_G * logit(P_G) + w_M * logit(P_M))

**Seed=0 Weight Search** (10 weight configs x 2 modes = 20 evals):

| Rank | Mode | w_logistic | w_gbdt | w_mlp | Mean BA | first_batch | stage2 | stage3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | logit_average | 0.30 | 0.35 | 0.35 | 0.8015 | 0.7559 | 0.8446 | 0.8041 |
| 2 | prob_average | 0.20 | 0.40 | 0.40 | 0.8005 | 0.7513 | 0.8395 | 0.8108 |
| 3 | prob_average | 0.10 | 0.80 | 0.10 | 0.7996 | 0.7513 | 0.8443 | 0.8032 |
| - | gbdt (single) | - | 1.00 | - | 0.8093 | 0.7770 | 0.8443 | 0.8065 |

**5-Seed Validation** (top 3 configs):

| Method | first_batch BA | stage2 BA | stage3 BA | Mean BA |
|---|---:|---:|---:|---:|
| pairwise_gbdt (baseline) | 0.7522 | 0.8310 | 0.8039 | 0.7957 |
| **ensemble prob_avg (0.20/0.40/0.40)** | **0.7649** | 0.8279 | 0.8043 | **0.7990** |
| ensemble logit_avg (0.30/0.35/0.35) | 0.7591 | 0.8266 | 0.8029 | 0.7962 |
| ensemble prob_avg (0.10/0.80/0.10) | 0.7455 | 0.8178 | 0.8037 | 0.7890 |

**Key Findings:**
1. Best ensemble marginally beats GBDT (+0.0033 mean BA), driven by first_batch improvement (+0.0127)
2. first_batch degradation reduced: ensemble 0.7649 vs GBDT 0.7522, though still below concat 0.7849
3. stage2/stage3 essentially match GBDT (within ±0.003)
4. Ensemble mode (prob vs logit) matters less than weight distribution; both top configs give GBDT+MLP > 70% combined weight
5. Very GBDT-heavy weights (0.10/0.80/0.10) underperform the balanced config, confirming that MLP adds complementary signal

**Reproduce:**
```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

# Seed=0 weight search
python run_ensemble.py --search \
  --model-dir /tmp/pairwise_llm_exp_full/models \
  --split-root /tmp/pairwise_llm_exp_full/splits \
  --output-dir /tmp/pairwise_llm_exp_ensemble

# 5-seed on top config
python run_ensemble.py \
  --seeds 0 1 2 3 4 \
  --weights 0.20 0.40 0.40 \
  --ensemble-mode prob_average \
  --output-dir /tmp/pairwise_llm_exp_ensemble
```

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
