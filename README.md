# EDA Regression Failure Bucketing

这个项目用于处理 EDA/ICCAD 风格的 regression failure bucketing 任务：给定一批 Ibex RISC-V CPU 回归仿真的失败日志，把由同一根因 bug 导致的 case 尽量分到同一个 bucket。

正式预测程序是 `regr_fail_bucketing.py`，接口兼容赛题要求：

```bash
regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```


## Latest Work Summary

Recent experiments focused on whether a single generalized model can work well across the old fake datasets, fixed official benchmarks, and the official-directed sanitized diagnostic dataset without routing by dataset origin. The formal default predictor remains unchanged and still does not read `gold.csv`, `meta.csv`, or `trace.log`.

Key findings:

- The previous source-gated official adapter is strong on the fixed official benchmarks, but it uses an input-format gate, so it is not the final generalization target.
- A stricter global unified adapter with one model and one alpha improves fake stage2/stage3 and official set2, especially with LLM scalar features and anchor-trace features, but official set1 remains the bottleneck.
- Full retraining on official set1+set2 overfits official data: official scores improve, but fake datasets and sanitized collapse through over-merging.
- Mixed fake+official full retraining with higher official pair weight preserves fake performance and makes set2 strong, but still does not solve set1 and hurts sanitized recall.
- Official set1 and set2 do transfer to each other when trained alone, which means the official labels contain useful shared signal; the main unresolved problem is how to keep that boundary signal when mixed with the larger fake distribution.

Current experimental recommendation: keep the no-trace calibrated blend as the stable experimental baseline, keep the source-gated adapter only as an analysis/reference route, and move next to domain-balanced error-aware sampling/objectives that emphasize high-confidence false merges without introducing dataset-source routing at inference time.

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
| **mlp** | PyTorch MLP; legacy 21-dim shallow and experimental deep/residual rich modes | 深度学习 |

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



### Rich Pairwise MLP (Experimental)

Rich Pairwise MLP is a research-only deep learning route. It does not change the official predictor interface and is not the default. Training/evaluation scripts read half-split `gold.csv` labels; formal prediction with `regr_fail_bucketing.py --input ... --output ... --k ...` still reads only `input.csv` plus referenced `sim.log` / `regr.log`, never `gold.csv` or `meta.csv`, and never `trace.log`.

新增 feature modes:

| feature_mode | Pair input |
|---|---|
| `summary21` | existing 21 scalar pair features |
| `rich` | `abs(det_i-det_j), det_i*det_j, abs(llm_i-llm_j), llm_i*llm_j, summary21` |
| `rich_no_llm` | deterministic vector relation plus summary scalars with LLM scalar terms zeroed |
| `rich_no_det` | LLM vector relation plus summary scalars with deterministic scalar terms zeroed |

For rich modes, Nomic/OpenAI-compatible embedding vectors are reduced with a train-fitted `TruncatedSVD + Normalizer` sidecar before pair construction (`--llm-reduce-dim`, default 128). Torch checkpoints only store `state_dict` and architecture/config fields; scaler/reducer preprocessing is saved separately as `.preproc.pkl` to avoid torch deserialization issues.

New MLP options include:

```text
--feature-mode summary21|rich|rich_no_llm|rich_no_det|llm_dual|llm_dual_struct|llm_dual_struct_det_summary
--llm-reduce-dim 128
--mlp-arch shallow|deep|residual
--loss bce|focal
--dropout 0.2
--layernorm / --no-layernorm
--batchnorm
```

Seed=0 search with real Nomic embeddings (`embedding_dim=768`, no fallback warnings):

| method | feature_mode | arch | loss | first_BA | stage2_BA | stage3_BA | mean_BA |
|---|---|---|---|---:|---:|---:|---:|
| rich_mlp_summary21_shallow_bce | summary21 | shallow | bce | 0.785714 | 0.830357 | 0.805856 | 0.807309 |
| rich_mlp_summary21_deep_focal | summary21 | deep | focal | 0.802679 | 0.835587 | 0.794351 | 0.810872 |
| rich_mlp_rich_deep_bce | rich | deep | bce | 0.720179 | 0.709949 | 0.751991 | 0.727373 |
| rich_mlp_rich_deep_focal | rich | deep | focal | 0.772143 | 0.732440 | 0.788889 | 0.764491 |
| rich_mlp_rich_residual_focal | rich | residual | focal | 0.831964 | 0.797109 | 0.792417 | 0.807163 |

Seed=0 ablation on residual/focal:

| method | feature_mode | first_BA | stage2_BA | stage3_BA | mean_BA |
|---|---|---:|---:|---:|---:|
| summary21_deep_focal | summary21 | 0.802679 | 0.835587 | 0.794351 | 0.810872 |
| rich_residual_focal | rich | 0.831964 | 0.797109 | 0.792417 | 0.807163 |
| rich_no_llm_residual_focal | rich_no_llm | 0.750000 | 0.690986 | 0.734130 | 0.725039 |
| rich_no_det_residual_focal | rich_no_det | 0.708036 | 0.804847 | 0.812350 | 0.775078 |

5-seed validation of the best candidates:

| method | feature_mode | arch | loss | first_BA | stage2_BA | stage3_BA | mean_BA |
|---|---|---|---|---:|---:|---:|---:|
| rich_no_det_residual_focal | rich_no_det | residual | focal | 0.790071 | 0.815425 | 0.811272 | 0.805590 |
| rich_residual_focal | rich | residual | focal | 0.811750 | 0.775901 | 0.793327 | 0.793659 |
| summary21_deep_focal | summary21 | deep | focal | 0.750036 | 0.830689 | 0.795655 | 0.792127 |
| summary21_shallow_bce | summary21 | shallow | bce | 0.752071 | 0.827551 | 0.797428 | 0.792350 |

Conclusion: `rich_no_det_residual_focal` exceeds the current soft-voting ensemble on mean BA (0.8056 vs 0.7990) and improves stage3 (0.8113 vs 0.8043), but it trails ensemble on stage2 (0.8154 vs 0.8279) and has a high-TNR/low-TPR under-merge profile. It is worth keeping as an experimental candidate, especially for larger stage3-like sets, but the soft-voting ensemble remains the safer experimental best for balanced stage2/stage3 behavior.

Reproduce:

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_rich_mlp_experiments.py \
  --output-dir /tmp/rich_pairwise_mlp_5seed_best \
  --seeds 0 1 2 3 4 \
  --configs summary21_deep_focal summary21_shallow_bce \
            rich_residual_focal rich_no_det_residual_focal \
  --epochs 30 \
  --batch-size 8192 \
  --max-train-pairs 300000 \
  --negative-ratio 2.0 \
  --hard-negative-ratio 0.5 \
  --hard-positive-ratio 0.5 \
  --early-stop-patience 8 \
  --llm-reduce-dim 128 \
  --dropout 0.2 \
  --lr 1e-3 \
  --weight-decay 1e-4
```


### Input-Signal Upgrade: Dual LLM + Structured Signals

The current best deep-learning route improves the input representation rather than making the network deeper. It keeps the residual MLP/focal-loss setup, but gives each pair two independently reduced LLM embedding channels:

```text
features-doc embedding relation: abs(f_i - f_j), f_i * f_j
summary-doc embedding relation:  abs(s_i - s_j), s_i * s_j
structured sim/regr signals and deterministic scalar summaries
```

This route is experimental and only runs through `train_pairwise_llm.py` / `run_input_signal_experiments.py`. The default `regr_fail_bucketing.py --input ... --output ... --k ...` baseline remains unchanged and does not read `gold.csv`, `meta.csv`, or `trace.log`.

New input modes:

| feature_mode | Pair input |
|---|---|
| `llm_dual` | two LLM relation blocks plus LLM cosine/distance scalars |
| `llm_dual_struct` | `llm_dual` plus structured pair features from `sim.log` / `regr.log` |
| `llm_dual_struct_det_summary` | `llm_dual_struct` plus deterministic scalar summaries; no deterministic vector relation |

Structured signals include same/error-conflict checks for source file, UVM component, test name, mismatch type, op pair, Ibex/Spike opcode, register, PC region, primary signature/type, and UVM/regr failure flags. These are extracted from selected `sim.log` / `regr.log` lines only.

Seed=0 input-signal search with real Nomic embeddings showed both channels active (`embedding_dim=768` for `doc_style=features` and `doc_style=summary`):

| method | feature_mode | reduce_dim | blend_alpha | first_BA | stage2_BA | stage3_BA | mean_BA |
|---|---|---:|---:|---:|---:|---:|---:|
| rich_no_det_residual_focal | rich_no_det | 128 | - | 0.7080 | 0.8139 | 0.8060 | 0.7760 |
| llm_dual_residual_focal | llm_dual | 128 | - | 0.8296 | 0.8778 | 0.8292 | 0.8455 |
| llm_dual_struct_residual_focal | llm_dual_struct | 128 | - | 0.8764 | 0.8751 | 0.8050 | 0.8522 |
| llm_dual_struct_det_summary | llm_dual_struct_det_summary | 128 | - | 0.8500 | 0.8499 | 0.8468 | 0.8489 |
| llm_dual_struct_det_summary_dim64 | llm_dual_struct_det_summary | 64 | - | 0.9348 | 0.8523 | 0.8145 | 0.8672 |
| llm_dual_struct_det_summary_dim256 | llm_dual_struct_det_summary | 256 | - | 0.7013 | 0.8213 | 0.7529 | 0.7585 |
| llm_dual_struct_det_summary_dim64_blend_a0.50 | llm_dual_struct_det_summary | 64 | 0.50 | 0.7988 | 0.8863 | 0.8407 | 0.8419 |
| llm_dual_struct_det_summary_dim64_blend_a0.75 | llm_dual_struct_det_summary | 64 | 0.75 | 0.8921 | 0.8673 | 0.8403 | 0.8666 |

5-seed validation of the top configs:

| method | feature_mode | reduce_dim | blend_alpha | first_BA | stage2_BA | stage3_BA | mean_BA | first TPR/TNR | stage2 TPR/TNR | stage3 TPR/TNR |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| llm_dual_struct_det_summary_dim64 | llm_dual_struct_det_summary | 64 | - | 0.8915 | 0.8491 | 0.8276 | 0.8561 | 0.8425/0.9406 | 0.7327/0.9654 | 0.6831/0.9722 |
| llm_dual_struct_det_summary_dim64_blend_a0.50 | llm_dual_struct_det_summary | 64 | 0.50 | 0.7770 | 0.8672 | 0.8417 | 0.8287 | 0.7300/0.8240 | 0.8250/0.9095 | 0.7711/0.9123 |
| **llm_dual_struct_det_summary_dim64_blend_a0.75** | `llm_dual_struct_det_summary` | 64 | 0.75 | **0.8469** | **0.8741** | **0.8510** | **0.8573** | 0.7900/0.9037 | 0.8101/0.9381 | 0.7446/0.9574 |
| llm_dual_struct_det_summary_residual_focal | llm_dual_struct_det_summary | 128 | - | 0.8914 | 0.8314 | 0.8227 | 0.8485 | 0.8500/0.9329 | 0.7042/0.9586 | 0.6765/0.9690 |

Compared with prior candidates:

| Method | first_BA | stage2_BA | stage3_BA | Mean BA |
|---|---:|---:|---:|---:|
| soft voting ensemble (0.20/0.40/0.40) | 0.7649 | 0.8279 | 0.8043 | 0.7990 |
| rich_no_det_residual_focal | 0.7901 | 0.8154 | 0.8113 | 0.8056 |
| **dual input + ensemble blend alpha=0.75** | **0.8469** | **0.8741** | **0.8510** | **0.8573** |

Conclusion: dual LLM input signals are a clear improvement over the single-embedding `rich_no_det` route and over the previous soft-voting ensemble. The initial 5-seed best was `llm_dual_struct_det_summary_dim64` blended with the existing pairwise ensemble at `alpha=0.75`; follow-up calibration below improves this further. Keep it experimental because it depends on LLM embeddings and trained sidecar reducers/models.

Reproduce:

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_input_signal_experiments.py \
  --output-dir /tmp/input_signal_5seed_top \
  --seeds 0 1 2 3 4 \
  --configs llm_dual_struct_det_summary_dim64 llm_dual_struct_det_summary_residual_focal \
  --blend-with-ensemble \
  --blend-configs llm_dual_struct_det_summary_dim64 \
  --blend-alphas 0.5 0.75 \
  --epochs 30 \
  --batch-size 8192 \
  --max-train-pairs 300000 \
  --negative-ratio 2.0 \
  --hard-negative-ratio 0.5 \
  --hard-positive-ratio 0.5 \
  --early-stop-patience 8 \
  --llm-reduce-dim 128 \
  --dropout 0.2 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --ensemble-model-dir /tmp/pairwise_llm_exp_full/models \
  --ensemble-split-root /tmp/pairwise_llm_exp_full/splits
```


### Calibration Follow-Up

After the dual-input run, `run_input_signal_calibration.py` reuses trained models and evaluates blend calibration without retraining. It supports alpha grids and probability temperature on the rich MLP and ensemble probability matrices.

Global alpha sweep, no temperature:

| config | first_BA | stage2_BA | stage3_BA | mean_BA | stage2 TPR/TNR | stage3 TPR/TNR |
|---|---:|---:|---:|---:|---|---|
| alpha=0.75 | 0.8469 | 0.8741 | 0.8510 | 0.8573 | 0.8101/0.9381 | 0.7446/0.9574 |
| **alpha=0.80** | **0.8669** | 0.8704 | 0.8545 | **0.8639** | 0.7940/0.9468 | 0.7442/0.9649 |
| alpha=0.85 | 0.8638 | 0.8692 | 0.8546 | 0.8625 | 0.7911/0.9473 | 0.7410/0.9683 |

Small temperature grid around the best alpha region:

| config | first_BA | stage2_BA | stage3_BA | mean_BA | stage2 TPR/TNR | stage3 TPR/TNR |
|---|---:|---:|---:|---:|---|---|
| alpha=0.80, rich_temp=1.00, ens_temp=0.75 | 0.8698 | 0.8727 | 0.8564 | 0.8663 | 0.8042/0.9412 | 0.7482/0.9646 |
| alpha=0.85, rich_temp=1.25, ens_temp=1.00 | 0.8726 | 0.8728 | 0.8601 | 0.8685 | 0.7982/0.9474 | 0.7504/0.9698 |
| **alpha=0.85, rich_temp=0.75, ens_temp=1.25** | **0.8798** | 0.8710 | 0.8564 | **0.8691** | 0.7881/0.9539 | 0.7383/0.9744 |

Dataset-size policy search gives an upper-bound style result on the current three validation sets:

| dataset | setting | BA | TPR/TNR |
|---|---|---:|---|
| first_batch | alpha=0.85, rich_temp=0.75, ens_temp=1.25 | 0.8798 | 0.8425/0.9171 |
| stage2 | alpha=0.80, rich_temp=1.25, ens_temp=0.75 | 0.8748 | 0.8113/0.9383 |
| stage3 | alpha=0.80, rich_temp=1.25, ens_temp=1.00 | 0.8606 | 0.7540/0.9671 |
| policy mean | - | 0.8717 | - |

Historical note: this initial 0-4 calibration selected `alpha=0.85, rich_temp=0.75, ensemble_temp=1.25`, but it is superseded by the 0-9 retune below. Treat the dataset-size policy as a promising validation result to retest on new data before relying on it.

A later 0-9 calibration grid supersedes the initial 0-4 setting. Reusing the same trained rich MLP and ensemble artifacts, the most stable global no-trace setting by primary mean BA is:

| setting | first_BA | stage2_BA | stage3_BA | mean_BA | stage2 TPR/TNR | stage3 TPR/TNR |
|---|---:|---:|---:|---:|---|---|
| old 0-9 reference, alpha=0.85, rt=0.75, et=1.25 | 0.8761 | 0.8556 | 0.8541 | 0.8620 | 0.7560/0.9553 | 0.7349/0.9733 |
| **new 0-9 best, alpha=0.88, rt=1.15, et=1.00** | **0.8930** | **0.8595** | **0.8552** | **0.8692** | 0.7655/0.9535 | 0.7369/0.9735 |
| best minimum-dataset BA, alpha=0.82, rt=1.15, et=0.90 | 0.8723 | 0.8632 | 0.8600 | 0.8651 | 0.7762/0.9502 | 0.7497/0.9702 |

Recommendation after the 0-9 retune: keep the no-trace calibrated dual-input blend as the recommended experimental best, now with global `alpha=0.88, rich_temp=1.15, ensemble_temp=1.00`. The `alpha=0.82, rich_temp=1.15, ensemble_temp=0.90` setting is more balanced by minimum dataset BA, but lower on the primary mean-BA metric.

Reproduce calibration:

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_input_signal_calibration.py \
  --output-dir /tmp/input_signal_temp_grid \
  --seeds 0 1 2 3 4 \
  --alphas 0.75 0.80 0.85 \
  --rich-temperatures 0.75 1.0 1.25 \
  --ensemble-temperatures 0.75 1.0 1.25
```


### Stability and Postprocess Follow-Up

A follow-up validation extended the calibrated global blend to seeds 5-9 and then summarized seeds 0-9. The model and ensemble artifacts for seeds 5-9 were trained with the same experimental settings; LLM logs confirmed `embedding_dim=768` for both features and summary embedding paths, with no fallback warning.

| seeds | first_BA | stage2_BA | stage3_BA | mean_BA | stage2 TPR/TNR | stage3 TPR/TNR |
|---|---:|---:|---:|---:|---|---|
| 0-4 | 0.8798 | 0.8710 | 0.8564 | 0.8691 | 0.7881/0.9539 | 0.7383/0.9744 |
| 5-9 | 0.8724 | 0.8403 | 0.8519 | 0.8548 | 0.7238/0.9567 | 0.7315/0.9722 |
| 0-9 | 0.8761 | 0.8556 | 0.8541 | 0.8620 | 0.7560/0.9553 | 0.7349/0.9733 |

The 5-9 split is lower than 0-4, mostly due to stage2 TPR, so the calibrated alpha/temperature is not perfectly stable. However, the 0-9 mean BA still remains above the previous uncalibrated dual blend (0.8573), rich_no_det (0.8056), and soft-voting ensemble (0.7990).

Pairwise error analysis shows the remaining error profile is still mostly recall-limited:

| dataset | BA | TPR | TNR | FN p50 | FP p50 | top fragmented bugs |
|---|---:|---:|---:|---:|---:|---|
| first_batch | 0.8761 | 0.8263 | 0.9260 | 0.7692 | 0.8412 | bug_001, bug_003, bug_002 |
| stage2 | 0.8556 | 0.7560 | 0.9553 | 0.8168 | 0.8554 | bug_003, bug_002, bug_001 |
| stage3 | 0.8541 | 0.7349 | 0.9733 | 0.8027 | 0.8254 | bug_002, bug_030, bug_003 |

FN pairs often have high pair probability already, but naive merging is dangerous because many FP pairs also have high probability and share broad signatures such as `unknown_source`, `register_write_data_mismatch`, or `pc_mismatch`. The largest FP groups are adjacent-looking bug pairs such as stage2 `bug_018/bug_019`, `bug_012/bug_013`, and stage3 `bug_012/bug_013`, `bug_016/bug_031`, `bug_023/bug_024`.

Experimental postprocess helpers were added:

- `merge_close_buckets`: merge predicted buckets using top-k inter-bucket probability plus sim/regr-derived consistency checks.
- `split_mixed_buckets`: split buckets by primary signature, op pair, mismatch type, or fatal file when there is clear internal structure.

Seed=0 search showed aggressive merge raises TPR but destroys TNR; split is mostly no-op or hurts TPR. A representative 0-9 validation confirmed no postprocess beats the calibrated blend:

| postprocess | first_BA | stage2_BA | stage3_BA | mean_BA | stage2 TPR/TNR | stage3 TPR/TNR |
|---|---:|---:|---:|---:|---|---|
| none | 0.8761 | 0.8556 | 0.8541 | 0.8620 | 0.7560/0.9553 | 0.7349/0.9733 |
| split_mixed auto, min_bucket=6, min_group=3 | 0.8761 | 0.8535 | 0.8536 | 0.8611 | 0.7515/0.9556 | 0.7337/0.9735 |
| merge_close p>=0.92, consistency>=0.95 | 0.8761 | 0.8536 | 0.8517 | 0.8605 | 0.7693/0.9379 | 0.7476/0.9558 |
| merge_close p>=0.90, consistency>=0.95 | 0.8711 | 0.8485 | 0.8442 | 0.8546 | 0.8057/0.8914 | 0.7762/0.9121 |

Recommendation: keep postprocess disabled for now. The current experimental best remains the calibrated global dual-input blend without postprocess, using `alpha=0.88, rich_temp=1.15, ensemble_temp=1.00`. Future gains should come from better training/sampling or more discriminative input signals rather than a simple structural merge/split pass.

Reproduce stability and postprocess checks:

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_input_signal_calibration.py \
  --output-dir /tmp/calibrated_blend_stability_raw \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --alphas 0.85 \
  --rich-temperatures 0.75 \
  --ensemble-temperatures 1.25

python error_analysis_pairwise.py \
  --output-dir /tmp/calibrated_blend_error_analysis \
  --seeds 0 1 2 3 4 5 6 7 8 9

python run_postprocess_experiments.py \
  --output-dir /tmp/postprocess_top_0_9 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --postprocess merge_close split_mixed \
  --merge-prob-thresholds 0.90 0.92 \
  --merge-consistency-thresholds 0.95 \
  --merge-conflict-maxes 0.05 \
  --split-min-bucket-sizes 6 \
  --split-min-group-sizes 3 \
  --split-keys auto
```


### Sampling and Cross-Signal Follow-Up

A new experimental round tested whether the remaining recall gap could be reduced by changing training sampling or adding lightweight LLM agreement signals, without increasing the residual MLP depth.

Implementation additions:

- `--positive-sampling det_low|diverse`: hard positive mining can rank same-bug pairs by combined deterministic/features/summary embedding dissimilarity plus different failure hints.
- `--negative-sampling det_high|confusable`: hard negative mining can rank different-bug pairs by semantic similarity plus shared primary signature, mismatch type, fatal file, register, or PC region.
- `llm_dual_struct_det_summary_cross`: adds 14 scalar agreement features between features-style and summary-style LLM embeddings, including cross cosine, max/min agreement, product agreement, and L1/Linf diff summaries.

All of these remain experimental-only. They do not affect `regr_fail_bucketing.py` default prediction and still use only `sim.log` and `regr.log`; no `gold.csv` is read outside training/evaluation scripts.

Seed=0 quick search did not produce a stable replacement for the current calibrated global blend:

| experiment | best setting | first_BA | stage2_BA | stage3_BA | mean_BA | conclusion |
|---|---|---:|---:|---:|---:|---|
| baseline sampling grid | `alpha=0.75, rt=0.75, et=1.25` | 0.8329 | 0.8601 | 0.8576 | 0.8502 | useful calibration reference, below current best |
| hard positive, ratio=0.50 | `alpha=0.75, rt=0.75, et=1.00` | 0.8329 | 0.8743 | 0.8585 | 0.8552 | promising on seed=0 only |
| hard positive, ratio=0.50, seeds 0-4 | same setting | 0.8041 | 0.8537 | 0.8365 | 0.8314 | not stable enough |
| hard negative / hard pos+neg | calibration grid | <=0.8473 | <=0.8604 | <=0.8486 | <=0.8473 | no improvement |
| cross-scalar input | calibration grid | 0.8236 | 0.8741 | 0.8334 | 0.8437 | no improvement |
| hard positive, ratio=0.25 | `alpha=0.75, rt=0.75, et=1.25` | 0.8329 | 0.8743 | 0.8519 | 0.8530 | better than ratio=0.25 baseline, still below ratio=0.50 seed=0 and not enough for 0-9 |

The main lesson is that naive hard-positive oversampling can improve seed=0 stage2/stage3 recall, but it has high seed variance and hurts first_batch. Cross-embedding scalar agreement did not add a useful signal beyond the existing dual relation blocks and structured features.

Recommendation: keep the current experimental best as `llm_dual_struct_det_summary_dim64 + residual focal MLP + pairwise ensemble calibrated blend`. After the 0-9 calibration retune above, the recommended global no-trace setting is `alpha=0.88, rich_temp=1.15, ensemble_temp=1.00`. Do not enable the new sampling or cross-scalar mode as the recommended best. Future work should focus on less noisy supervised objectives, such as pair weighting by cluster fragmentation, per-dataset calibration learned from held-out splits, or ranking/metric losses that directly optimize close same-bug pairs without over-repeating them.


### Selective Trace-Assisted Pair Refinement

A selective trace experiment was added to test whether `trace.log` can help only on uncertain no-trace pairs, without making trace part of the default model. The official deterministic baseline and the current no-trace experimental best remain unchanged; trace is parsed only by experimental scripts and is disabled by default everywhere else.

Method:

- Build the current no-trace calibrated blend probability matrix `P_base` from `llm_dual_struct_det_summary_dim64 + residual focal MLP + pairwise ensemble`.
- Select only uncertain pairs where `P_base` falls inside a probability band.
- Parse only the tail of each `trace.log` and extract compact structural signals: opcode/register/PC sets and counts, branch/load/store/CSR ratios, exception markers, and tail opcode n-gram/LCS overlap.
- Train a lightweight `HistGradientBoostingClassifier` trace refiner on uncertain training pairs only, then replace `P_base[i,j]` only for uncertain validation pairs.
- Missing trace files fall back gracefully; current public datasets had 0 missing trace files in this experiment.

Seed=0 showed a promising but narrow signal. The best seed=0 setting was `tail_lines=500`, band `0.40-0.60`, GBDT refiner:

| setting | first_BA | stage2_BA | stage3_BA | mean_BA | note |
|---|---:|---:|---:|---:|---|
| no-trace base, same seed/split | 0.9348 | 0.8956 | 0.8423 | 0.8909 | calibrated blend |
| trace refine tail=500 band=0.40-0.60 | 0.9348 | 0.8956 | 0.8549 | 0.8951 | stage3 improved on seed=0 |

However, the improvement did not hold strongly across seeds:

| seeds | first_BA | stage2_BA | stage3_BA | mean_BA | mean delta vs no-trace | stage2 TPR/TNR | stage3 TPR/TNR |
|---|---:|---:|---:|---:|---:|---|---|
| 0-4 no-trace base | 0.8798 | 0.8710 | 0.8564 | 0.8691 | - | 0.7881/0.9539 | 0.7383/0.9744 |
| 0-4 trace refine | 0.8837 | 0.8742 | 0.8560 | 0.8713 | +0.0022 | 0.7893/0.9591 | 0.7361/0.9759 |
| 5-9 no-trace base | 0.8724 | 0.8403 | 0.8519 | 0.8548 | - | 0.7238/0.9567 | 0.7315/0.9722 |
| 5-9 trace refine | 0.8695 | 0.8403 | 0.8509 | 0.8536 | -0.0013 | 0.7232/0.9574 | 0.7271/0.9747 |
| 0-9 no-trace base | 0.8761 | 0.8556 | 0.8541 | 0.8620 | - | 0.7560/0.9553 | 0.7349/0.9733 |
| 0-9 trace refine | 0.8766 | 0.8573 | 0.8535 | 0.8624 | +0.0005 | 0.7562/0.9583 | 0.7316/0.9753 |

The 0-9 refiner touched a selective slice of pairs: about 47 uncertain pairs on first_batch, 364 on stage2, and 3111 on stage3 per validation split, with no missing trace pairs. Pair-level error deltas show why the method is not yet robust: it fixes FP pairs, but also creates enough new FP/FN or loses same-bug recall to offset most gains. On 0-9, stage3 averaged about 290 fixed FP pairs but also about 192 new FP pairs and 30 new FN pairs.

Recommendation: keep trace refinement experimental and disabled. The current recommended experimental best remains the no-trace calibrated blend. Trace features may still be useful, but the next iteration should use a safer correction policy, for example only allowing high-confidence vetoes/boosts with monotonic constraints or calibrating separate FP-veto and FN-merge refiners instead of replacing probabilities directly.

Reproduce:

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_trace_refinement_experiments.py \
  --output-dir /tmp/trace_refine_exp_seed0 \
  --seeds 0 \
  --tail-lines 200 500 1000 \
  --uncertain-bands 0.25,0.75 0.35,0.65 0.40,0.60 \
  --refiner-model gbdt

python run_trace_refinement_experiments.py \
  --output-dir /tmp/trace_refine_exp_0_4 \
  --seeds 0 1 2 3 4 \
  --tail-lines 200 500 1000 \
  --uncertain-bands 0.40,0.60 \
  --refiner-model gbdt
```

### Conservative Trace Veto/Boost

A follow-up trace experiment replaced the learned GBDT refiner with a rule-based, monotonic policy. It still treats trace as experimental-only postprocess: the official predictor and no-trace experimental blend do not read `trace.log`, `gold.csv`, or `meta.csv`; trace parsing is invoked only by `run_trace_policy_experiments.py`.

Policy:

- `none`: keep the calibrated no-trace probability matrix.
- `veto`: cap high `P_base` pairs only when trace agreement is very low and sim/regr structured signals do not strongly agree.
- `boost`: floor moderate `P_base` pairs only when trace agreement is high and sim/regr structured signals do not conflict.
- `veto_boost`: apply both conservative edits.

Trace agreement is a compact structural score from opcode Jaccard, tail opcode 2-gram Jaccard, tail opcode LCS ratio, and PC-region Jaccard. Trace text is not sent to the embedding endpoint or to an LLM completion API. Missing traces fall back gracefully; the current public datasets had 0 missing trace pairs.

0-9 no-trace calibration grid:

| setting | first_BA | stage2_BA | stage3_BA | mean_BA | stage2 TPR/TNR | stage3 TPR/TNR |
|---|---:|---:|---:|---:|---|---|
| old reference, alpha=0.85, rt=0.75, et=1.25 | 0.8761 | 0.8556 | 0.8541 | 0.8620 | 0.7560/0.9553 | 0.7349/0.9733 |
| **new no-trace best, alpha=0.88, rt=1.15, et=1.00** | **0.8930** | **0.8595** | **0.8552** | **0.8692** | 0.7655/0.9535 | 0.7369/0.9735 |

Seed=0 trace policy search using the new calibration showed possible local gains:

| policy | first_BA | stage2_BA | stage3_BA | mean_BA | note |
|---|---:|---:|---:|---:|---|
| none | 0.9348 | 0.8781 | 0.8395 | 0.8841 | no-trace calibrated base |
| boost, base 0.25-0.60, trace>=0.55, floor=0.65 | 0.9348 | 0.8922 | 0.8409 | 0.8893 | improves stage2 on seed=0 |
| veto, base>=0.55, trace<=0.20, cap=0.35 | 0.9348 | 0.8781 | 0.8436 | 0.8855 | small stage3 gain |
| veto_boost, same veto plus trace>=0.85/floor=0.75 | 0.9348 | 0.8931 | 0.8473 | 0.8918 | best seed=0 only |

The gains did not hold across 0-9:

| policy | first_BA | stage2_BA | stage3_BA | mean_BA | stage2 TPR/TNR | stage3 TPR/TNR |
|---|---:|---:|---:|---:|---|---|
| **none** | **0.8930** | 0.8595 | 0.8552 | **0.8692** | 0.7655/0.9535 | 0.7369/0.9735 |
| boost, base 0.25-0.60, trace>=0.55, floor=0.65 | 0.8716 | **0.8655** | 0.8554 | 0.8642 | 0.7795/0.9516 | 0.7374/0.9734 |
| veto_boost, base>=0.55 trace<=0.20 cap=0.35 plus same boost | 0.8716 | 0.8651 | **0.8557** | 0.8641 | 0.7783/0.9519 | 0.7380/0.9735 |

Pair-level deltas explain the instability. The best boost reduces FN on stage2/stage3, but introduces too many FP pairs and hurts first_batch strongly on some seeds:

| policy | fixed_FN | new_FN | fixed_FP | new_FP | net_FN_delta | net_FP_delta |
|---|---:|---:|---:|---:|---:|---:|
| boost | 252 | 219 | 1343 | 1625 | -33 | +282 |
| veto_boost | 261 | 224 | 1405 | 1640 | -37 | +235 |

Recommendation: do not enable trace policy by default and do not treat it as the experimental best. The current recommended experimental best remains the no-trace calibrated blend with `alpha=0.88, rich_temp=1.15, ensemble_temp=1.00`. Trace features have a real but noisy recall signal, especially on stage2, but the current boost rule is not conservative enough for small datasets and creates a net FP cost.

Reproduce:

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_input_signal_calibration.py \
  --output-dir /tmp/notrace_calibration_0_9 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --alphas 0.75 0.80 0.82 0.85 0.88 0.90 \
  --rich-temperatures 0.65 0.75 0.85 1.00 1.15 \
  --ensemble-temperatures 0.90 1.00 1.10 1.25 1.40

python run_trace_policy_experiments.py \
  --output-dir /tmp/trace_policy_exp_0_9 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --alpha 0.88 \
  --rich-temperature 1.15 \
  --ensemble-temperature 1.00 \
  --trace-policy none boost veto_boost \
  --tail-lines 500 \
  --veto-base-min 0.55 \
  --veto-trace-max 0.20 \
  --veto-cap 0.35 \
  --boost-base-ranges 0.25,0.60 \
  --boost-trace-min 0.55 \
  --boost-floor 0.65
```


### Current Best Strict Validation

The current recommended experimental best is the no-trace calibrated blend:

```text
llm_dual_struct_det_summary_dim64
+ residual focal MLP
+ pairwise soft-voting ensemble blend
+ alpha=0.88, rich_temp=1.15, ensemble_temp=1.00
```

This remains experimental and does not change the official default predictor. Training/evaluation scripts may read `gold.csv`; official prediction still reads only `input.csv` plus `sim.log` / `regr.log`, does not read `gold.csv` or `meta.csv`, and does not use `trace.log` by default. LLM usage is embedding-only.

Leave-one-dataset-out validation trains on two bundled datasets and evaluates on the held-out dataset:

| method | first_BA | stage2_BA | stage3_BA | mean_BA | note |
|---|---:|---:|---:|---:|---|
| deterministic | 0.7821 | 0.7657 | 0.7788 | 0.7755 | default no-LLM baseline |
| llm_concat_features | 0.7805 | 0.7612 | 0.7773 | 0.7730 | embedding concat baseline |
| pairwise_soft_voting_ensemble | 0.8061 | 0.7757 | 0.7153 | 0.7657 | summary21 supervised ensemble |
| **current_no_trace_calibrated_blend** | **0.8656** | **0.8551** | **0.8582** | **0.8596** | alpha=0.88, rt=1.15, et=1.00 |

The LODO result shows the dual-input calibrated blend still generalizes better than the older baselines when the validation dataset is entirely held out. The held-out stage2/stage3 runs keep high TNR but lower TPR, so the model remains conservative and recall-limited.

Additional seeds 10-19 were run with the same current-best parameters. They are materially lower than seeds 0-9, so the more conservative stability estimate is the 0-19 aggregate rather than the earlier 0-9 headline:

| dataset | seeds | mean_BA | std_BA | mean_TPR | mean_TNR | min_BA | max_BA |
|---|---|---:|---:|---:|---:|---:|---:|
| first_batch | 0-9 | 0.8930 | 0.0424 | 0.8525 | 0.9334 | 0.8030 | 0.9607 |
| stage2 | 0-9 | 0.8595 | 0.0169 | 0.7655 | 0.9535 | 0.8235 | 0.8840 |
| stage3 | 0-9 | 0.8552 | 0.0173 | 0.7369 | 0.9735 | 0.8185 | 0.8766 |
| first_batch | 10-19 | 0.8548 | 0.0219 | 0.8113 | 0.8983 | 0.8180 | 0.8841 |
| stage2 | 10-19 | 0.8310 | 0.0333 | 0.7354 | 0.9266 | 0.7942 | 0.8995 |
| stage3 | 10-19 | 0.8213 | 0.0116 | 0.7409 | 0.9017 | 0.8048 | 0.8447 |
| **ALL mean** | **0-19** | **0.8525** | **0.0304** | **0.7737** | **0.9312** | **0.7942** | **0.9607** |

Current-best pairwise error analysis over seeds 0-19:

| dataset | BA | TPR | TNR | FN pairs | FP pairs | top FN bug | top FP bug pair |
|---|---:|---:|---:|---:|---:|---|---|
| first_batch | 0.8739 | 0.8319 | 0.9159 | 269 | 1178 | bug_001 | bug_001/bug_002 |
| stage2 | 0.8452 | 0.7504 | 0.9401 | 1677 | 7050 | bug_003 | bug_012/bug_013 |
| stage3 | 0.8383 | 0.7389 | 0.9376 | 7519 | 61883 | bug_030 | bug_012/bug_013 |

The dominant metric imbalance is low TPR relative to TNR: same-bug pairs are still fragmented. However, the absolute pair-error count is dominated by FP because negative pairs are much more numerous, and many FP pairs sit in the same high-probability band as FN pairs. The repeated FP pairs are adjacent-looking or structurally similar bugs such as `bug_012/bug_013`, `bug_018/bug_019`, `bug_023/bug_024`, and `bug_025/bug_026`. This explains why naive merge/boost postprocess and trace boost improve recall locally but do not reliably improve overall BA.

Recommendation after strict validation: keep the current best as the no-trace calibrated blend with `alpha=0.88, rich_temp=1.15, ensemble_temp=1.00`, but report both the 0-9 headline mean BA (0.8692) and the stricter 0-19 stability mean BA (0.8525). Do not enable trace policy or structural postprocess by default.

Reproduce strict validation:

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_lodo_experiments.py \
  --output-dir /tmp/lodo_exp

python run_input_signal_calibration.py \
  --output-dir /tmp/current_best_seeds_10_19 \
  --seeds 10 11 12 13 14 15 16 17 18 19 \
  --alphas 0.88 \
  --rich-temperatures 1.15 \
  --ensemble-temperatures 1.00

python error_analysis_pairwise.py \
  --output-dir /tmp/current_best_error_analysis \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --alpha 0.88 \
  --rich-temperature 1.15 \
  --ensemble-temperature 1.00
```


### Anchor-Guided Trace Window Experiments

Full-trace transformer experiments showed that trace has some signal, but the original trace runner's no-trace path did not reproduce the current best. Its 5-seed runner baseline was about mean BA 0.8374, while `trace_embedding_blend_a0.88` reached about 0.8427. Because that no-trace baseline was misaligned with the current calibrated blend, these numbers are not considered a new best.

The trace runner was fixed so `no_trace` reuses the current-best artifacts and applies the same calibration (`alpha=0.88, rich_temp=1.15, ensemble_temp=1.00`). Seed 0 now matches the current-best calibration runner exactly:

| runner | seed | first_BA | stage2_BA | stage3_BA | mean_BA |
|---|---:|---:|---:|---:|---:|
| current calibration runner | 0 | 0.9348 | 0.8781 | 0.8395 | 0.8841 |
| trace runner no_trace fixed | 0 | 0.9348 | 0.8781 | 0.8395 | 0.8841 |

Anchor-guided trace windows use failure information from `sim.log` / `regr.log` to choose a local `trace.log` window, then add only lightweight structural pair features. This remains experimental and is not used by the default predictor. On the fake datasets, anchors were found for all train/validation cases via simulation failure time, with zero tail fallback in seed 0 (`located_by=time` for 960/960 anchor-debug rows across window sizes 32/64/128).

Seed 0 results after fixing failure-time anchoring:

| method | window | first_BA | stage2_BA | stage3_BA | mean_BA | first_TPR/TNR | stage2_TPR/TNR | stage3_TPR/TNR |
|---|---:|---:|---:|---:|---:|---|---|---|
| no_trace_current_best | - | 0.9348 | 0.8781 | 0.8395 | 0.8841 | 0.9125/0.9571 | 0.7917/0.9646 | 0.7042/0.9747 |
| tail_trace_struct | 500 | 0.7796 | 0.8298 | 0.8464 | 0.8186 | 0.7250/0.8343 | 0.7232/0.9364 | 0.7528/0.9400 |
| anchor_trace_struct | 32 | 0.8236 | 0.8625 | 0.8604 | 0.8488 | 0.8000/0.8471 | 0.7798/0.9452 | 0.7799/0.9409 |
| anchor_trace_struct | 64 | 0.8236 | 0.8642 | 0.8487 | 0.8455 | 0.8000/0.8471 | 0.7798/0.9486 | 0.7576/0.9397 |
| anchor_trace_struct | 128 | 0.8236 | 0.8745 | 0.8434 | 0.8471 | 0.8000/0.8471 | 0.7857/0.9633 | 0.7250/0.9617 |

Anchor windows are more useful than simple tail structural features on stage2/stage3, and they improve stage3 TPR versus no-trace, but they hurt first_batch and overall TNR enough that mean BA stays below the no-trace current best. Seeds 0-4 were therefore not expanded in this round.

Recommendation: keep the no-trace calibrated blend as the experimental best. If trace is revisited, the next trace direction should be an anchor-window transformer or a gated trace feature that only acts on uncertain pairs, but only after preserving the aligned no-trace baseline in the same runner.


### Official-Directed Trace Diagnosis and Direct Evaluation

A new fake official-directed dataset was added at `fake_dataset/official_directed_stage1_valid_failure_only`. Its `sim.log` / `regr.log` surface signals are intentionally very similar across the four gold bugs (`bug_036`, `bug_037`, `bug_038`, `bug_041`): many cases look like `core_ibex_debug_intr_basic_test` with PC/register mismatch or timeout-style failures. This exposes a failure mode for no-trace models: they over-merge, so TPR remains high while TNR collapses.

Direct evaluation on this dataset uses `gold.csv` for scoring and, for trace-supervised rows, for same-dataset pairwise training. These rows are diagnostic experiments, not official prediction settings. The default `regr_fail_bucketing.py` path still does not read trace/gold/meta.

| method | BA | TPR | TNR | clusters | note |
|---|---:|---:|---:|---:|---|
| deterministic | 0.5775 | 0.8366 | 0.3184 | 4 | default no-trace baseline |
| no_trace_best | 0.5899 | 0.8434 | 0.3363 | 4 | 10-seed calibrated blend, alpha=0.88, rt=1.15, et=1.00 |
| trace_policy_veto_boost | 0.5899 | 0.8434 | 0.3363 | 4 | conservative tail policy did not move clustering |
| trace_tail_trace_only | 0.8644 | 0.8383 | 0.8905 | 4 | direct supervised trace GBDT |
| anchor_trace_w32_trace_only | 0.8632 | 0.9991 | 0.7272 | 4 | direct supervised anchor-window GBDT |
| **anchor_trace_w64_trace_only** | **0.9843** | **0.9796** | **0.9891** | 4 | best direct trace-only diagnostic |
| anchor_trace_w128_trace_only | 0.8632 | 0.9991 | 0.7272 | 4 | direct supervised anchor-window GBDT |
| trace_tail_rich | 1.0000 | 1.0000 | 1.0000 | 4 | direct supervised rich+trace GBDT; train=evaluate |
| anchor_trace_w32/w64/w128_rich | 1.0000 | 1.0000 | 1.0000 | 4 | direct supervised rich+trace GBDT; train=evaluate |

The bug-pair confusion confirms the diagnosis. `no_trace_best` merges most of the target bug pairs: `bug_036/bug_037` has 864 FP pairs, `bug_037/bug_038` has 416, and `bug_036/bug_038` has 351. With `anchor_trace_w64_trace_only`, the remaining target confusion is only `bug_036/bug_038` with 28 FP pairs. This is a large TNR improvement and shows that trace contains the missing discriminative signal.

The rule anchor extractor was extended to use PC mismatch / DUT retired PC, ISS retired PC, register mismatch targets, and debug/IRQ/CSR/illegal/timeout tags. On the official-directed dataset, anchor windows were located without tail fallback: across window sizes 32/64/128, `located_by=pc` for 222/261 anchor rows and `located_by=time` for 39/261 rows.

As a sanity check, the same anchor structural route was run on the older fake datasets with seed 0 and window size 64:

| method | first_BA | stage2_BA | stage3_BA | mean_BA | note |
|---|---:|---:|---:|---:|---|
| no_trace_current_best | 0.9348 | 0.8781 | 0.8395 | 0.8841 | aligned current best |
| tail_trace_struct_w500 | 0.8064 | 0.8747 | 0.8455 | 0.8422 | improves stage3 recall but hurts first/TNR |
| anchor_trace_struct_w64 | 0.8671 | 0.8496 | 0.8345 | 0.8504 | below no-trace current best |

Recommendation: keep the current no-trace calibrated blend as the experimental best for general use. For official-directed style data, continue trace-aware work as a separate experimental backend, preferably with anchor-window train/heldout validation rather than same-dataset direct supervision. The most promising next step is an anchor-window trace model or gated trace feature that preserves the aligned no-trace baseline and only activates when sim/regr signals are ambiguous.

Reproduce the direct trace diagnosis:

```bash
source /home/lishixian/miniforge3/etc/profile.d/conda.sh
conda activate collab-overcooked
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_official_directed_trace_eval.py \
  --dataset fake_dataset/official_directed_stage1_valid_failure_only \
  --output-dir /tmp/official_directed_trace_eval \
  --methods deterministic no_trace_best trace_tail trace_policy anchor_trace \
  --window-sizes 32 64 128 \
  --k-from-gold \
  --trace-context rich

python run_official_directed_trace_eval.py \
  --dataset fake_dataset/official_directed_stage1_valid_failure_only \
  --output-dir /tmp/official_directed_trace_eval_traceonly \
  --methods deterministic trace_tail anchor_trace \
  --window-sizes 32 64 128 \
  --k-from-gold \
  --trace-context trace_only
```



### External Trace Transfer Experiments

The official-directed sanitized target dataset `fake_dataset/official_directed_stage1_sanitized_3bugs_85cases` is held out for these experiments. Its `gold.csv` is used only at the final scoring step, never for training, calibration, or parameter selection.

Training sources:

- old fake datasets with gold: `fake_dataset/first_batch_dataset`, `fake_dataset/stage2_dataset_working`, `fake_dataset/stage3_dataset_32bugs_640cases`
- public benchmark traces with manual weak labels: `test_case/problem/benchmark_set_1`, `test_case/problem/benchmark_set_2`
- manual benchmark labels are weak labels from `benchmark_manual_label/benchmark1/*` and `benchmark_manual_label/benchmark2/*`; they are not official ground truth

Target baseline on the current tree:

| method | BA | TPR | TNR | note |
|---|---:|---:|---:|---|
| no_trace_current_best_external | 0.5753 | 0.8501 | 0.3005 | current calibrated no-trace artifacts, seedavg10 |

External transfer results on the sanitized target:

| config | window | model | runs | BA | TPR | TNR | note |
|---|---:|---|---:|---:|---:|---:|---|
| anchor_old_fake | 128 | gbdt | 5 | 0.5932 | 0.7845 | 0.4019 | best anchor structural transfer |
| anchor_old_fake_benchmark | 128 | gbdt | 5 | 0.5932 | 0.7845 | 0.4019 | benchmark manual weak labels did not improve |
| anchor_old_fake_benchmark | 128 | logistic | 5 | 0.5888 | 0.8458 | 0.3318 | small TNR gain, recall mostly preserved |
| anchor_old_fake | 64 | gbdt | 5 | 0.5862 | 0.7930 | 0.3794 | moderate TNR gain, TPR loss |
| trace_encoder_old_fake_benchmark | 64 | gbdt | 1 | 0.5689 | 0.8373 | 0.3005 | rich + anchor + pretrained trace encoder did not transfer |

Anchor location statistics for the full external-transfer grid:

- target located_by: `pc=4380`, `time=720` across all target/window/model/random-state runs
- train located_by top counts: `time=49200`, `pc=9480`
- trace encoder smoke: `trace_encoded_train=996`, `trace_encoded_target=85`

The best transfer model reduces over-merge false positives but not enough: TNR rises from about 0.3005 to 0.4019, while TPR drops from 0.8501 to 0.7845. This is a real but small transfer signal, far below the direct supervised trace diagnostic (`anchor_trace_w64_trace_only` near 0.97 BA when trained on the target itself). The gap indicates dataset shift: current anchor structural features can separate the sanitized target when trained on it, but rules learned from old fake + manual public benchmark data do not yet generalize strongly.

Recommendation: keep this route experimental. Continue only if the next iteration improves transfer supervision, for example by training on official-directed style synthetic traces, doing leave-bug-family-out validation, or learning anchor-window trace embeddings with contrastive objectives. Do not enable trace transfer in the default predictor.

Reproduce:

```bash
source /home/lishixian/miniforge3/etc/profile.d/conda.sh
conda activate collab-overcooked
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_external_trace_transfer_experiments.py \
  --output-dir /tmp/external_trace_transfer_exp \
  --train-datasets fake_dataset/first_batch_dataset fake_dataset/stage2_dataset_working fake_dataset/stage3_dataset_32bugs_640cases \
  --benchmark-datasets test_case/problem/benchmark_set_1 test_case/problem/benchmark_set_2 \
  --target-dataset fake_dataset/official_directed_stage1_sanitized_3bugs_85cases \
  --configs no_trace anchor_old_fake anchor_old_fake_benchmark anchor_blend \
  --window-sizes 32 64 128 \
  --model-types logistic gbdt \
  --betas 0.25 0.50 0.75 1.00 \
  --random-states 0 1 2 3 4

python run_external_trace_transfer_experiments.py \
  --output-dir /tmp/external_trace_transfer_trace_encoder_smoke \
  --train-datasets fake_dataset/first_batch_dataset fake_dataset/stage2_dataset_working fake_dataset/stage3_dataset_32bugs_640cases \
  --benchmark-datasets test_case/problem/benchmark_set_1 test_case/problem/benchmark_set_2 \
  --target-dataset fake_dataset/official_directed_stage1_sanitized_3bugs_85cases \
  --configs no_trace trace_encoder_old_fake_benchmark \
  --window-sizes 64 \
  --model-types gbdt \
  --random-state 0 \
  --device cpu
```



### Official Gold Trace-Assisted Validation

Official `golden.csv` labels are now available for `test_case/problem/benchmark_set_1` and `test_case/problem/benchmark_set_2`. This validation keeps the current no-trace calibrated blend unchanged and tests only zero-shot/unsupervised trace-assisted corrections. It does not train on official trace logs, and it does not change the default `regr_fail_bucketing.py` path.

Current no-trace best:

- `llm_dual_struct_det_summary_dim64`
- residual focal MLP + pairwise soft-voting ensemble
- `alpha=0.88, rich_temp=1.15, ensemble_temp=1.00`
- seed average over seeds `0..9`

Official-gold results:

| dataset | method | BA | TPR | TNR | note |
|---|---|---:|---:|---:|---|
| benchmark_set_1 | no_trace_best | 0.5313 | 0.8125 | 0.2500 | over-merges cases `1,2,3,4,5` with `6,7,9`; isolates case `8` |
| benchmark_set_1 | trace_tail_unsupervised | 0.5313 | 0.8125 | 0.2500 | same score as no-trace |
| benchmark_set_1 | anchor_trace_w32/w64 | 0.5313 | 0.8125 | 0.2500 | same score as no-trace |
| benchmark_set_1 | anchor_trace_w128 | 0.4750 | 0.7500 | 0.2000 | worse split |
| benchmark_set_1 | trace_guided_split_w32/w64 | 0.4563 | 0.5625 | 0.3500 | TNR improves, but TPR drops too much |
| benchmark_set_1 | trace_policy_zero_shot | 0.4563 | 0.5625 | 0.3500 | 11 vetoes, no boosts |
| benchmark_set_2 | no_trace_best | 0.9560 | 0.9516 | 0.9604 | already strong |
| benchmark_set_2 | trace_tail_unsupervised | 0.7831 | 0.9274 | 0.6388 | hurts TNR |
| benchmark_set_2 | anchor_trace_w64 | 0.7013 | 0.7903 | 0.6123 | hurts both metrics |
| benchmark_set_2 | trace_guided_split_w32/w64/w128 | 0.9560 | 0.9516 | 0.9604 | gated no-op, preserves no-trace |
| benchmark_set_2 | trace_policy_zero_shot | 0.8348 | 0.9516 | 0.7181 | hurts TNR |

Existing trace-transformer artifacts were initially skipped because official case ids (`1`, `2`, ...) did not match the feature ids (`case_1`, `case_2`, ...), leaving trace vectors unattached and producing a 302-d feature stack instead of the 366-d stack expected by the trace model. After fixing the experimental runner to match trace paths by case-id alias and input order, the old fake-trained trace embedding model runs on official gold: set1 remains unchanged (`BA=0.5313, TPR=0.8125, TNR=0.2500`), while set2 drops to `BA=0.7288, TPR=0.9113, TNR=0.5463`. This confirms the trained trace embedding does not currently help official-gold evaluation.

Set1 error analysis: official gold groups cases `1..5` as one bug and `6..9` as another. The no-trace model predicts one large mixed bucket containing cases `1,2,3,4,5,6,7,9` and a singleton `8`, giving 15 false-positive cross-bug pairs and 3 false-negative pairs within `bug_7023`. Tail/anchor trace similarities do not form a clean zero-shot `1..5` vs `6..9` separation; for example, cases `4`, `5`, and `9` have very similar anchor-window opcode profiles even though case `9` belongs to the other official bug.

Recommendation: do not enable trace-guided split or zero-shot trace policy as an official-gold backend yet. The trace gate can safely preserve set2 when it no-ops, but it does not solve set1 and it can lower TPR sharply. The next trace direction should either learn a calibrated split policy on non-official data or use official traces only for diagnostic analysis until a held-out validation source exists.

Reproduce:

```bash
source /home/lishixian/miniforge3/etc/profile.d/conda.sh
conda activate collab-overcooked
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"

python run_official_trace_assisted_eval.py \
  --benchmarks test_case/problem/benchmark_set_1 test_case/problem/benchmark_set_2 \
  --output-dir /tmp/official_trace_assisted_eval \
  --methods no_trace_best trace_tail_unsupervised anchor_trace trace_guided_split trace_policy_zero_shot existing_trace_embedding \
  --window-sizes 32 64 128
```


### Official-Style Root-Cause Training

After the fixed official `golden.csv` labels were pulled, the previous no-trace calibrated blend no longer matched the official grouping style on the public benchmarks. With the current artifacts (`llm_dual_struct_det_summary_dim64`, `alpha=0.88`, `rich_temp=1.15`, `ensemble_temp=1.00`, seeds `0..9`), direct official-gold evaluation is:

| dataset | method | BA | TPR | TNR |
|---|---|---:|---:|---:|
| benchmark_set_1 | no_trace_best | 0.4583 | 0.6667 | 0.2500 |
| benchmark_set_2 | no_trace_best | 0.4742 | 0.3419 | 0.6066 |

The main issue is not missing model capacity. The pair probabilities learned from old fake datasets assign high similarity to pairs that the fixed official labels split apart, and low/medium similarity to many same-official-bug pairs. In set1, no-trace predicts a mixed bucket with both `bug_7023` and `bug_234`; in set2, large portions of `bug_107` are fragmented while other bugs are mixed into the same bucket.

A new experimental leave-one-benchmark-out runner was added:

```bash
python run_official_style_training_experiments.py \
  --benchmarks test_case/problem/benchmark_set_1 test_case/problem/benchmark_set_2 \
  --output-dir /tmp/official_style_training_exp_final \
  --variants tags tags_graph tags_graph_anchor \
  --model-types logistic gbdt \
  --window-sizes 64 \
  --blend-alphas 0.25 0.50 \
  --seed 0
```

This route trains only on the other benchmark's fixed official labels and extracts official-style root-cause tags from `sim.log`/`regr.log`: `irq_entry`, `debug_entry`, `dret_return`, `csr`, `mcause_exception`, `core_status_timeout`, PC/register divergence, and related tags. It remains experimental and does not change `regr_fail_bucketing.py`.

LOBO results:

| train | test | method | BA | TPR | TNR |
|---|---|---|---:|---:|---:|
| - | benchmark_set_1 | no_trace_best | 0.4583 | 0.6667 | 0.2500 |
| - | benchmark_set_2 | no_trace_best | 0.4742 | 0.3419 | 0.6066 |
| benchmark_set_2 | benchmark_set_1 | official_style_tags_logistic | 0.7222 | 0.7778 | 0.6667 |
| benchmark_set_2 | benchmark_set_1 | official_style_tags_gbdt | 0.7222 | 0.7778 | 0.6667 |
| benchmark_set_1 | benchmark_set_2 | official_style_tags_logistic_blend0.50 | 0.9213 | 0.9573 | 0.8852 |
| benchmark_set_1 | benchmark_set_2 | official_style_tags_gbdt_blend0.50 | 0.9100 | 0.9402 | 0.8798 |

The useful signal is the root-cause tag layer itself. Adding graph context from the old no-trace probability matrix often reintroduces the old bias; anchor trace structural features are not consistently better in this fixed-official setting. The next architecture should therefore treat official-style root-cause tags as a first-class supervised objective, then only add trace/window signals behind a validation gate.

Recommendation: keep the previous no-trace calibrated blend as the old fake-dataset experimental best, but for the fixed official benchmark style pursue a separate experimental `official_style_tags` backend. Do not replace the formal deterministic default until it is validated on additional held-out official-style data.


Correction: the first five-dataset LODO run exposed a case-id alignment bug for fake datasets without an explicit `Case` column. Fake inputs use log paths such as `cases/case_000001/trace.log`, while `gold.csv` uses `case_000001`; the experimental reader now infers `case_000001` from the path before joining gold labels. After the fix, pair labels are normal: first_batch `pos=360`, stage2 `pos=1680`, stage3 `pos=6080`, benchmark_set_1 `pos=9`, benchmark_set_2 `pos=117`.

Corrected five-dataset LODO (`train = other four datasets`, `test = held-out dataset`, no cross-dataset clustering) shows that `official_style_tags` is useful on the old fake datasets but not yet a universal mainline:

| test | no_trace_best BA | best official_style_tags BA | note |
|---|---:|---:|---|
| first_batch_dataset | 0.8555 | 0.8555 | tie; GBDT alone drops to 0.8430 |
| stage2_dataset_working | 0.8317 | 0.8528 | logistic improves TPR with moderate TNR loss |
| stage3_dataset_32bugs_640cases | 0.8682 | 0.8954 | logistic improves TPR and keeps high TNR |
| benchmark_set_1 | 0.4583 | 0.4583 | no improvement when trained with fake + set2 |
| benchmark_set_2 | 0.4742 | 0.5783 | GBDT improves but remains weak |

Conclusion: do not replace the existing no-trace calibrated blend yet. The corrected result supports an experimental gated architecture: keep no-trace calibrated blend as the base, add official-style root-cause tags as an auxiliary correction when validation or dataset-style detection indicates it helps.


#### Five-Dataset Leave-One-Out Check

To check whether `official_style_tags` is useful beyond the two public official benchmarks, the runner also supports multi-dataset leave-one-out validation:

```bash
python run_official_style_training_experiments.py \
  --benchmarks fake_dataset/first_batch_dataset fake_dataset/stage2_dataset_working fake_dataset/stage3_dataset_32bugs_640cases test_case/problem/benchmark_set_1 test_case/problem/benchmark_set_2 \
  --output-dir /tmp/official_style_lodo_5datasets_tags \
  --eval-mode leave_one_out \
  --variants tags \
  --model-types logistic gbdt \
  --blend-alphas 0.50 \
  --seed 0
```

This trains on four datasets and evaluates on the held-out fifth dataset. It is still experimental and uses gold only in the training/evaluation script.

| held-out test | no_trace_best BA | tags_logistic_blend0.50 BA | tags_gbdt_blend0.50 BA |
|---|---:|---:|---:|
| first_batch_dataset | 0.8555 | 0.8555 | 0.8867 |
| stage2_dataset_working | 0.8317 | 0.8369 | 0.8035 |
| stage3_dataset_32bugs_640cases | 0.8682 | 0.8745 | 0.8658 |
| benchmark_set_1 | 0.4583 | 0.7222 | 0.5139 |
| benchmark_set_2 | 0.4742 | 0.4777 | 0.4626 |

Aggregate:

| method | all 5 mean BA | fake-only mean BA | mean TPR | mean TNR |
|---|---:|---:|---:|---:|
| no_trace_best | 0.6976 | 0.8518 | 0.6597 | 0.7355 |
| tags_logistic_blend0.50 | 0.7534 | 0.8556 | 0.6934 | 0.8133 |
| tags_gbdt_blend0.50 | 0.7065 | 0.8520 | 0.6256 | 0.7874 |

Interpretation: `tags_logistic_blend0.50` is the best current official-style candidate. It slightly improves the three fake datasets, strongly improves official benchmark set1, and raises overall TNR, but it still fails on official benchmark set2. This is not yet strong enough to replace the current experimental best globally. The next iteration should target set2's remaining official-label mismatch, likely by adding a second objective for broad same-root families such as `bug_107` while preserving high-confidence splits for smaller root-cause classes.


### Official-Style Adapter Gated Blend

The corrected five-dataset LODO experiment supports a gated adapter rather than a single mixed model. Training the adapter on official public benchmarks only (`benchmark_set_1` and `benchmark_set_2`) and blending it lightly into the no-trace current best gives the best tradeoff. Mixing fake datasets directly into the adapter, even with official pair weight 100, improves fake stage2/stage3 but largely loses the official benchmark gain, especially on `benchmark_set_2`.

Results from `run_official_style_training_experiments.py` with reused no-trace probability matrices:

| setup | test | no_trace BA | best adapter BA | note |
|---|---|---:|---:|---|
| official_only | first_batch | 0.8555 | 0.8555 | no regression at blend 0.25 |
| official_only | stage2 | 0.8317 | 0.8437 | GBDT blend 0.25; logistic blend 0.25 is 0.8414 |
| official_only | stage3 | 0.8682 | 0.8758 | logistic blend 0.25 |
| official_only | benchmark_set_1 | 0.4583 | 0.7222 | logistic/GBDT adapter or blend 0.50 |
| official_only | benchmark_set_2 | 0.4742 | 0.9213 | logistic blend 0.50 |
| mixed official:fake 10:1 | benchmark_set_2 | 0.4742 | 0.4641 | fake data overwhelms official style |
| mixed official:fake 100:1 | benchmark_set_2 | 0.4742 | 0.4641 | still does not recover set2 |

Candidate gated policy:

- Train adapter on official public benchmarks only.
- Use logistic `official_style_tags` as the adapter model.
- If the dataset has official-style `Case, Regr Log, Sim Log, Trace Log` columns and gzip logs, use `adapter_alpha=0.50`.
- Otherwise use `adapter_alpha=0.25` or keep no-trace if a validation gate is unavailable.

This is still experimental. It should not replace the default deterministic predictor or the no-trace calibrated blend yet, but it is the strongest direction for aligning with fixed official labels while preserving old fake-dataset performance. The next implementation step is to add an experimental prediction backend that trains/loads this official-only adapter and applies the dataset-style gate without reading gold at prediction time.


#### Auto-Gated Adapter Backend

A standalone experimental runner now evaluates the gated adapter without changing the formal predictor:

```bash
python run_official_gated_adapter_eval.py \
  --output-dir /tmp/official_gated_adapter_eval \
  --reuse-base-probs-dir /tmp/official_style_lodo_5datasets_tags/probs \
  --gate auto \
  --fake-alpha 0.25 \
  --official-alpha 0.50 \
  --model-type logistic
```

The adapter is trained only on the public official benchmarks. When the target is one of those official benchmarks, the runner uses leave-target-official-out training. The auto gate uses input format only: official-style `Case, Regr Log, Sim Log, Trace Log` with gzip logs gets `alpha=0.50`; old fake-style path-only CSVs get `alpha=0.25`. Gold is used only for evaluation scoring.

| test | no_trace BA | gated adapter BA | TPR | TNR |
|---|---:|---:|---:|---:|
| first_batch_dataset | 0.8555 | 0.8555 | 0.8167 | 0.8943 |
| stage2_dataset_working | 0.8317 | 0.8414 | 0.7274 | 0.9554 |
| stage3_dataset_32bugs_640cases | 0.8682 | 0.8758 | 0.7794 | 0.9721 |
| benchmark_set_1 | 0.4583 | 0.7222 | 0.7778 | 0.6667 |
| benchmark_set_2 | 0.4742 | 0.9213 | 0.9573 | 0.8852 |
| mean | 0.6976 | 0.8432 | 0.8117 | 0.8748 |

This is currently the strongest source-gated experimental route across the three fake datasets plus the two fixed official benchmarks. It remains experimental because the adapter is trained from public official labels and the input-format gate has only been validated on the available datasets. It should not be treated as the final generalization model, because the long-term goal is a single model and one global calibration policy that works across fake, official, and official-directed data without routing by dataset origin. The formal default remains unchanged.

Reproduce:

```bash
python run_official_style_training_experiments.py \
  --benchmarks fake_dataset/first_batch_dataset fake_dataset/stage2_dataset_working fake_dataset/stage3_dataset_32bugs_640cases test_case/problem/benchmark_set_1 test_case/problem/benchmark_set_2 \
  --output-dir /tmp/official_adapter_official_only \
  --reuse-base-probs-dir /tmp/official_style_lodo_5datasets_tags/probs \
  --eval-mode leave_one_out \
  --train-source official_only \
  --variants tags \
  --model-types logistic gbdt \
  --blend-alphas 0.25 0.50 0.75 \
  --seed 0
```


#### Global Unified Adapter Search

`run_global_unified_adapter_search.py` evaluates a stricter experimental setting: one feature set, one adapter model, one blend alpha, and the same clustering rule across fake datasets, fixed official benchmarks, and the official-directed sanitized diagnostic dataset. It does not use a fake/official source gate. Gold/golden labels are used only inside this experimental runner for leave-one-dataset-out training and scoring; the formal predictor is unchanged.

The GPU MLP variant uses PyTorch for the adapter training step:

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"
CUDA_VISIBLE_DEVICES=7 python run_global_unified_adapter_search.py \
  --output-dir /tmp/global_unified_adapter_gpu_mlp_llm_trace_w64_seed0 \
  --feature-sets tags_structured_llm_trace \
  --models mlp \
  --official-weights 1 3 \
  --alphas 0.25 0.40 0.50 \
  --random-states 0 \
  --folds fake_first fake_stage2 fake_stage3 official_set1 official_set2 sanitized \
  --negative-ratio 2.0 \
  --exclude-sanitized-from-training \
  --device cuda \
  --epochs 30 \
  --batch-size 8192 \
  --hidden-dim 256 \
  --trace-window-size 64
```

Seed-0 LODO results from the first strict unified search:

| feature set | model | official weight | alpha | mean BA | min BA | fake mean | official mean | sanitized | set1 | set2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tags | mlp | 1 | 0.25 | 0.6969 | 0.4583 | 0.8640 | 0.5073 | 0.5753 | 0.4583 | 0.5562 |
| tags_structured_llm | mlp | 1 | 0.50 | 0.7438 | 0.5278 | 0.8810 | 0.6265 | 0.5666 | 0.5278 | 0.7252 |
| tags_structured_llm_trace | mlp | 3 | 0.50 | 0.7518 | 0.5278 | 0.8926 | 0.6256 | 0.5817 | 0.5278 | 0.7233 |

The strict unified model improves fake stage2/stage3 and fixed official set2, and trace-anchor features give a small sanitized gain. However, fixed official set1 remains the bottleneck at BA 0.5278. This suggests the next useful work is not simply a larger MLP or more source weighting; it is an error-aware objective/sampling strategy that targets high-confidence false merges while preserving same-bug recall. Trace-anchor features should be kept as candidate inputs, but they are not yet a standalone solution.


#### Official Full Retraining Diagnostic

`run_official_full_retrain_experiments.py` retrains the full pairwise rich model, not just the lightweight official-style adapter. It samples pairs only within each benchmark so independent datasets are not accidentally linked by shared bug names. This remains experimental and does not affect `regr_fail_bucketing.py`.

Seed-0 results with `llm_dual_struct_det_summary`, reduce dim 64, residual focal MLP:

| train data | eval target | BA | TPR | TNR | note |
|---|---|---:|---:|---:|---|
| set1+set2 | set1 | 1.0000 | 1.0000 | 1.0000 | train-set score, not generalization |
| set1+set2 | set2 | 0.8770 | 0.8632 | 0.8907 | train-set score, not generalization |
| set1+set2 | fake mean | 0.5042 | - | - | severe over-merge / poor transfer |
| set1+set2 | sanitized | 0.4999 | 0.9532 | 0.0467 | severe over-merge |
| fake+set1+set2, official weight 10 | fake mean | 0.8802 | - | - | fake mostly preserved |
| fake+set1+set2, official weight 10 | set1 | 0.5278 | 0.5556 | 0.5000 | still bottleneck |
| fake+set1+set2, official weight 10 | set2 | 0.9380 | 0.9744 | 0.9016 | strong |
| fake+set1+set2, official weight 10 | sanitized | 0.4961 | 0.3654 | 0.6269 | recall collapse |
| set2 only | set1 | 0.7222 | 0.7778 | 0.6667 | official cross-set transfer works |
| set1 only | set2 | 0.9072 | 0.9402 | 0.8743 | official cross-set transfer works |

The diagnostic says the official benchmarks contain useful shared signal: set1 and set2 can transfer to each other. The failure mode appears when fake and official data are mixed naively: the official boundary is diluted by the larger fake distribution, while official-only retraining overfits and collapses fake/sanitized TNR. The next promising direction is domain-balanced or error-aware sampling/objective design, not source-routed inference and not simply enlarging the MLP.


#### Unified Multi-Dataset Episodic Pair Training

`run_unified_multidataset_experiments.py` trains one experimental model across seven independent benchmarks: old fake first/stage2/stage3, VCS official-format fake data, stable official-like multitest data, and fixed official set1/set2.

Each benchmark remains an independent clustering episode. Positive/negative pairs, hard triplets, and transitivity triangles are created only within that benchmark; bug names are never compared across datasets. A dataset-balanced sampler gives small benchmarks comparable gradient exposure without selecting a model by dataset source at inference time.

The shared model uses the existing `llm_dual_struct_det_summary` 294-dimensional pair input with separately reduced features/summary embeddings, followed by a wider residual pair encoder. Experimental objectives are focal classification, hard-pair ranking, probability transitivity consistency, an optional gradient-reversal domain classifier, and conservative two-hop graph refinement.

Seed-0 strict leave-one-dataset-out results:

| config | graph | mean BA | min BA | set1 | set2 | first | stage2 | stage3 | VCS | stable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank | 0.0 | 0.7750 | 0.6111 | 1.0000 | 0.6990 | 0.8150 | 0.7586 | 0.7985 | 0.7428 | 0.6111 |
| **rank+transitivity+domain** | **0.1** | **0.8086** | 0.5556 | **1.0000** | **0.9084** | **0.8517** | **0.8046** | 0.7968 | 0.7428 | 0.5556 |

Three-seed probability averaging improves stability:

| config | graph | mean BA | set1 | set2 | first | stage2 | stage3 | VCS | stable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank | 0.0 | 0.7791 | 1.0000 | 0.7146 | 0.8136 | 0.7880 | 0.8296 | 0.7428 | 0.5648 |
| **rank+transitivity+domain** | **0.1** | **0.8140** | **1.0000** | **0.9170** | **0.8608** | 0.7892 | 0.8233 | 0.7428 | 0.5648 |

This is promising but not a replacement for the current experimental best yet. It improves official-set alignment while retaining useful old-fake performance, but the stable official-like dataset remains weak and individual models have seed variance. The next iteration should stabilize episode construction, use seed/model ensembling, warm up the domain loss, and target fragmented positives without source-routed inference.

Completion LLM use is intentionally deferred from this first objective experiment. The recommended next use is one cached completion per case that produces canonical failure JSON (mechanism, stage, mismatch object, state, trigger, and evidence tags), then uses those fields as teacher/structured features. Do not call completion per pair and do not let it directly output buckets.

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"
CUDA_VISIBLE_DEVICES=0 python run_unified_multidataset_experiments.py \
  --datasets \
    old_fake_dataset/first_batch_dataset \
    old_fake_dataset/stage2_dataset_working \
    old_fake_dataset/stage3_dataset_32bugs_640cases \
    official_format_fake_dataset/official_vcs_stage1_dataset_v1 \
    official_format_fake_dataset/stable_official_like_multitest_v1 \
    test_case/problem/benchmark_set_1 \
    test_case/problem/benchmark_set_2 \
  --holdouts all \
  --output-dir /tmp/unified_multidataset \
  --configs rank rank_trans_domain \
  --seeds 0 1 2 \
  --graph-gammas 0 0.1 \
  --device cuda
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

1. Keep `llm_dual_struct_det_summary_dim64` with global calibrated blend `alpha=0.88, rich_temp=1.15, ensemble_temp=1.00` as the current experimental best, but report both the 0-9 headline mean BA (0.8692) and the stricter 0-19 stability mean BA (0.8525).
2. Do not enable trace policy or lightweight structural postprocess by default; neither beat the no-trace current best under 0-9/0-19 validation.
3. Improve recall without increasing FP on adjacent-looking bug pairs. The most useful next direction is better supervised objectives or sampling that directly targets fragmented same-bug pairs while penalizing high-probability FP pairs such as `bug_012/bug_013`.
4. Investigate per-dataset or validation-learned calibration only if new private/public datasets are available; avoid tuning policies to the current three bundled datasets.
5. Keep all pairwise MLP routes experimental; default submitted baseline remains deterministic `drain + agglomerative`.


### Hosted Completion LLM (Experimental)

Completion experiments do not require a locally deployed model or a project GPU. The preferred exact-model development endpoint is NVIDIA NIM with `qwen/qwen3-coder-480b-a35b-instruct`; the second contest-compatible route is OpenAI `gpt-4o`. Both use an OpenAI-compatible chat-completions API.

API keys are read from environment variables and must not be committed. Example configurations are:

- `tools/llm_config_nvidia_qwen.yaml.example`
- `tools/llm_config_openai_gpt4o.yaml.example`

NVIDIA one-time setup (the prompt is hidden and the key is stored outside the repository with mode `600`):

```bash
tools/configure_nvidia_completion.sh
```

After that, any command can use the hosted completion endpoint without per-terminal exports:

```bash
tools/with_nvidia_completion.sh python tools/check_completion_endpoint.py
tools/with_nvidia_completion.sh python run_unified_trace_completion_experiments.py --help
```

The wrapper loads `~/.config/iccad/nvidia_api_key` and the checked-in YAML template only for the child process.

OpenAI example:

```bash
export OPENAI_API_KEY='...'
export LLM_MODEL_CONFIG="$(cat tools/llm_config_openai_gpt4o.yaml.example)"
python tools/check_completion_endpoint.py
```

`completion_case_features.py` makes at most one cached completion call per case. Before the request, it locally converts sim/regr logs into a strict whitelist of generic enums and booleans. It does not send raw log lines, paths, addresses, identifiers, gold/meta/trace, or bucket labels. The model only canonicalizes these signals into JSON attributes; it is never asked to assign buckets and is never called per pair. Hosted completion remains experimental and does not change the default predictor.


#### Hosted Completion and Trace Modality Experiments (June 2026)

This experimental round used the hosted NVIDIA NIM `qwen/qwen3-coder-480b-a35b-instruct` endpoint; no completion model was deployed on shared GPUs. Completion calls are cached once per case and never output buckets. The formal predictor remains unchanged.

A strict structured-completion input (generic enums and booleans only) was evaluated first. It was too coarse and caused false merges:

| 3-domain seed-0 LODO | stable | set1 | set2 | mean |
|---|---:|---:|---:|---:|
| base | 0.5648 | 0.7222 | 0.7746 | 0.6872 |
| trace | 0.5046 | 0.7222 | 0.8365 | 0.6878 |
| structured completion | 0.5037 | 0.4583 | 0.7840 | 0.5820 |
| trace + structured completion | 0.5648 | 0.5556 | 0.6342 | 0.5848 |

Raw-public completion caching was started separately, but the NVIDIA hosted endpoint reached HTTP 429 rate limits after partial progress. Cached responses are reusable; this route is not yet scored.

Direct trace concatenation was also inconsistent. In seven-domain seed-0 LODO it improved first, stage2, VCS, and set2, but severely damaged set1, reducing mean BA from 0.7571 to 0.7431.

A unified trace-modality-dropout model was then added. During training, the whole trace block is randomly removed for 50% of sampled pairs. At inference, full-trace and missing-trace views are probability-averaged. It uses one shared model rather than source-routed models.

| 7-domain seed-0 LODO | base | trace dropout |
|---|---:|---:|
| first | 0.7800 | **0.8390** |
| stage2 | 0.8067 | **0.8518** |
| stage3 | **0.8232** | 0.7818 |
| VCS | 0.6053 | **0.7428** |
| stable | 0.5648 | **0.5981** |
| set1 | **1.0000** | 0.5278 |
| set2 | 0.7201 | 0.7201 |
| mean | **0.7571** | 0.7231 |

The modality-dropout model is more useful than direct trace concatenation, but cannot replace the base model by itself. A diagnostic base-protection gate that enables trace dropout only under moderate base/dropout clustering agreement reached seed-0 mean BA 0.7905. This gate is not recommended as a default because its thresholds were observed on only seven datasets.

Five-seed probability averaging on stable/set1/set2 showed:

| method | stable | set1 | set2 |
|---|---:|---:|---:|
| base ensemble | 0.5648 | **0.7222** | 0.8944 |
| trace ensemble | 0.5185 | 0.5278 | 0.6895 |
| trace-modality-dropout ensemble | **0.5648** | 0.5278 | **0.9213** |

Next: train the trace model with base-teacher probability distillation and learn a pair-level gate only from inner validation pairs. This should preserve base boundaries on stage3/set1 while retaining trace gains on first/stage2/VCS/set2. Do not promote completion, trace concatenation, or the heuristic gate to the default path yet.


#### Multi-Granular Selective Expert and Signed-Graph Experiments (June 2026)

`run_multigranular_selective_experiments.py` implements a single experimental architecture across all seven independent benchmark episodes. It never routes by dataset name and never constructs pairs across datasets.

The pipeline is:

```text
multi-granular sim/regr evidence
  -> shared residual base pair model
  -> entropy/disagreement/margin/instability difficulty score
  -> selective anchor-trace and/or cached completion evidence
  -> base-distilled expert model
  -> validation-trained pair gate
  -> fixed-k or adaptive signed-graph clustering
```

`multigranular_features.py` extracts ordered fatal/error/mismatch/timeout events, normalized times and positions, PC/opcode/register/CSR/state objects, and separate local sim/regr evidence windows. Local windows use the configured embedding endpoint and are reduced independently; the existing global features/summary dual embedding and deterministic scalar features remain present.

Only the highest-difficulty 15% of cases in each episode receive expert evidence, with a minimum of 2 and maximum of 20 cases. Completion is called once per selected case and cached. It returns canonical evidence JSON, never a bucket. Missing trace, missing completion configuration, HTTP errors, and rate limits leave the base probability unchanged. The expert uses modality dropout and easy-pair base distillation. A learned gate is trained only from held-out training pairs; expert-unavailable pairs have gate weight exactly zero.

The signed-graph clusterer merges using top-k positive logit evidence minus structured conflict penalties. Adaptive k is restricted to `[0.8k, 1.2k]`; it selects the largest standardized merge-gain cliff, falls back to reference k when no cliff is clear, and prefers more buckets on ties.

Loss coefficients are fixed to classification 1.0, hard-pair ranking 0.20, transitivity 0.05, easy-pair distillation 0.30, and separately optimized gate BCE 0.20. Useful ablations are exposed through `--no-event-order`, `--no-local-embeddings`, `--expert-mode`, `--gate-mode`, `--clusterer`, and `--k-policy`.

A short two-epoch set1 LODO smoke test verified the complete base/fine/signed/trace path, both 768-dimensional global embedding channels, local embeddings, GPU training, selective trace extraction, output files, and error analysis:

| smoke method | set1 BA | TPR | TNR |
|---|---:|---:|---:|
| current base | 0.7222 | 0.7778 | 0.6667 |
| fine-grained base | 0.7222 | 0.7778 | 0.6667 |
| fine-grained signed graph | 0.7222 | 0.7778 | 0.6667 |
| selective trace | 0.5278 | 0.5556 | 0.5000 |

These are integration-smoke numbers, not model-selection results. In particular, the selective-trace regression confirms why expert gating and strict multi-seed LODO are required. The current no-trace calibrated blend remains the experimental best until the planned 0-4/0-9 seven-dataset run satisfies the official improvement and fake-data guardrails.

Example seed-0 screening command:

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"
CUDA_VISIBLE_DEVICES=0 python run_multigranular_selective_experiments.py \
  --output-dir /tmp/multigranular_selective_seed0 \
  --holdouts all \
  --seeds 0 \
  --variants current_base fine_base fine_signed selective_trace \
             selective_completion full full_adaptive \
  --device cuda
```

The runner writes `results.csv`, `summary.csv`, `ablation.csv`, `difficulty_debug.csv`, `completion_debug.csv`, `trace_debug.csv`, `cluster_trajectory.csv`, `error_analysis.md`, `preds/`, and `probs/`. The formal `regr_fail_bucketing.py` predictor is unchanged and still does not read gold, meta, or trace by default.

#### Pairwise Classifier Architecture Ablation (June 2026)

This experiment changes only the neural classifier over the existing 294-dimensional `llm_dual_struct_det_summary_dim64` pair vector. Feature construction, half-split episodes, pair sampling, focal loss, calibrated ensemble (`alpha=0.88`, rich temperature `1.15`, ensemble temperature `1.00`), and fixed-k average agglomerative clustering remain unchanged. The formal default predictor is unchanged.

Three experimental backends are available through `train_pairwise_model.py` and `run_arch_ablation.py`:

- `res_mlp`: the existing residual MLP control.
- `gated_mlp`: a feature-wise sigmoid gate followed by the same residual backbone, with gate regularization `1e-4`.
- `ft_transformer`: one scalar-feature token per input dimension, 64-dimensional tokens, two Transformer layers, four heads, FFN multiplier two, and dropout `0.1`. GPU micro-batches use gradient accumulation so all 294 tokens are retained.

Checkpoints contain the architecture name, architecture config, input dimension, feature schema, best clustering-validation BA, and `state_dict`. Scalers and LLM reducers remain in a separate preprocessing sidecar. The loader remains compatible with legacy residual/shallow/deep checkpoints. Early stopping uses mean clustering BA on the held-out half-split episodes, with focal validation loss as a tie-breaker.

Seed-0 screening used eight epochs and at most 50,000 sampled pairs (17,856 pairs were available for this split):

| architecture | evaluation | first BA | stage2 BA | stage3 BA | mean BA |
|---|---|---:|---:|---:|---:|
| res_mlp | model only | 0.9321 | 0.8473 | 0.8587 | 0.8794 |
| gated_mlp | model only | 0.8911 | 0.8615 | 0.9030 | **0.8852** |
| ft_transformer | model only | 0.8911 | 0.7768 | 0.8348 | 0.8342 |
| res_mlp | calibrated blend | 0.8157 | **0.8813** | 0.8355 | **0.8442** |
| gated_mlp | calibrated blend | 0.8064 | 0.8665 | 0.8468 | 0.8399 |
| ft_transformer | calibrated blend | **0.8671** | 0.7896 | **0.8473** | 0.8347 |

The gated model improves model-only mean BA by `+0.0058`, mainly through higher TPR on stage2/stage3, but loses `0.0411` BA on first_batch and reduces mean TNR. More importantly, under the unchanged calibrated-blend pipeline it is worse than the residual control. FT-Transformer also trails the residual control. Therefore this seed-0 screen does not justify a 0-9 expansion or a new experimental best; the existing no-trace calibrated residual blend remains recommended. The Siamese case encoder remains a TODO because introducing separate case-level towers would change the current pair-vector interface rather than isolate classifier architecture.

```bash
export LLM_MODEL_CONFIG="$(cat /tmp/nomic_llm.yaml)"
CUDA_VISIBLE_DEVICES=5 python run_arch_ablation.py \
  --output-dir /tmp/pairwise_arch_ablation_seed0 \
  --seeds 0 \
  --model-arches res_mlp gated_mlp ft_transformer \
  --epochs 8 --max-train-pairs 50000 --batch-size 4096 --device cuda
```

##### Seven-Dataset Strict LODO Architecture Validation

The classifier screen was extended to all seven labeled benchmarks using strict leave-one-dataset-out evaluation. Each held-out benchmark is used only for final clustering and scoring. Training pairs are sampled independently inside each of the other six benchmarks; no cross-dataset pairs are created. Fake labels come from `gold.csv`, while official-format and official benchmark labels come from `golden.csv`. This run uses the same 294-dimensional dual-input pair vector, focal classification, fixed reference k, and average agglomerative clustering. It does not include the historical calibrated soft-voting blend, graph refinement, trace, or completion.

Seed-0 all-architecture screening:

| held-out dataset | res_mlp | gated_mlp | ft_transformer |
|---|---:|---:|---:|
| first_batch | 0.7790 | **0.7914** | 0.7807 |
| stage2 | 0.7473 | **0.8036** | 0.7349 |
| stage3 | 0.7809 | **0.7985** | 0.7720 |
| VCS official-format fake | 0.5869 | **0.7428** | 0.5883 |
| stable official-like | 0.4880 | **0.6380** | 0.4884 |
| official set1 | 0.7222 | **1.0000** | 0.5278 |
| official set2 | 0.7146 | **0.9170** | 0.6850 |
| **macro mean** | 0.6884 | **0.8130** | 0.6539 |

Because gated MLP won all seven seed-0 folds, residual and gated were compared over seeds 0-2:

| held-out dataset | residual BA mean±std | gated BA mean±std | gated delta | gated 3-seed probability-average BA |
|---|---:|---:|---:|---:|
| first_batch | 0.7737±0.0352 | 0.7589±0.0695 | -0.0148 | 0.8003 |
| stage2 | 0.7570±0.0414 | 0.7857±0.0199 | +0.0287 | 0.7765 |
| stage3 | 0.7371±0.0621 | 0.7638±0.0304 | +0.0268 | 0.8208 |
| VCS official-format fake | 0.5869±0.0000 | 0.6908±0.0900 | +0.1039 | 0.7428 |
| stable official-like | 0.5392±0.0444 | 0.5920±0.0400 | +0.0528 | 0.5648 |
| official set1 | 0.8148±0.1604 | 1.0000±0.0000 | +0.1852 | 1.0000 |
| official set2 | 0.7094±0.0090 | 0.7821±0.1169 | +0.0726 | 0.7146 |
| **macro mean** | **0.7026** | **0.7676** | **+0.0650** | **0.7743** |

Gated MLP generalizes better on six of seven benchmarks and raises the official set1/set2 three-seed mean from `0.7621` to `0.8910`. The exception is first_batch, and official set2 remains seed-sensitive: seed 0 reaches `0.9170`, while seeds 1-2 reach `0.7146`; naïve probability averaging does not repair that boundary. Therefore gated MLP is the leading architecture candidate for the unified LODO path, but it does not yet replace the historical no-trace calibrated residual blend. The next fair step is seeds 0-9 plus calibration learned only from training-domain inner validation, with special attention to set2 TPR stability.

Raw outputs:

- `/tmp/all_dataset_arch_lodo_seed0/results.csv`
- `/tmp/all_dataset_gated_lodo_seeds1_2/results.csv`
- `/tmp/all_dataset_residual_lodo_seeds1_2/results.csv`
- `/tmp/all_dataset_arch_lodo_3seed_summary.csv`

##### Gated MLP 0-9 Stability and Inner Calibration

The strict seven-dataset LODO comparison was extended to seeds 0-9. Each fold trains on six independent benchmark episodes and evaluates the seventh; pair sampling remains dataset-local. FT-Transformer was not extended because its seed-0 macro BA (`0.6539`) was already below both residual and gated MLP.

| architecture | macro BA | official set1 BA | official set2 BA | official mean | global worst seed/dataset BA |
|---|---:|---:|---:|---:|---:|
| residual MLP | 0.7194 | 0.8333±0.1434 | 0.7136±0.0054 | 0.7735 | 0.4880 |
| gated MLP | **0.7581** | **0.9250±0.1646** | **0.7539±0.0836** | **0.8395** | **0.5130** |

Per-dataset gated results over ten seeds:

| dataset | BA mean±std | worst BA | mean TPR | mean TNR |
|---|---:|---:|---:|---:|
| first_batch | 0.7822±0.0581 | 0.6791 | 0.7600 | 0.8044 |
| stage2 | 0.7806±0.0235 | 0.7414 | 0.6457 | 0.9155 |
| stage3 | 0.7604±0.0296 | 0.7151 | 0.5810 | 0.9398 |
| VCS official-format fake | 0.7099±0.0650 | 0.5869 | 0.6190 | 0.8008 |
| stable official-like | 0.5946±0.0507 | 0.5130 | 0.4383 | 0.7509 |
| official set1 | 0.9250±0.1646 | 0.5278 | 0.9333 | 0.9167 |
| official set2 | 0.7539±0.0836 | 0.6990 | 0.6188 | 0.8891 |

A ten-seed probability ensemble raises gated macro BA to `0.7813` and official mean BA to `0.8573`. It improves first to `0.8608`, VCS to `0.7428`, and keeps set1 at `1.0000`. Set2 remains `0.7146`, showing that its unstable seeds share a systematic clustering boundary rather than independent probability noise.

Two no-target-leakage calibration variants were evaluated. For each target fold and seed, temperature, logit bias, residual/gated blend beta, and optionally cluster factor were selected using only OOF predictions and labels from the other six training domains. The target gold was never used for selection.

| calibration | macro BA | set1 BA | set2 BA | conclusion |
|---|---:|---:|---:|---|
| raw gated | **0.7581** | **0.9250** | **0.7539** | current candidate |
| inner calibrated, adaptive cluster factor | 0.7410 | 0.9167 | 0.7339 | over-selects factor 0.8 |
| inner calibrated, fixed k | 0.7518 | 0.9250 | 0.7354 | safer, still below raw gated |

Cross-domain calibration therefore does not transfer reliably enough to enable. Keep fixed reference k and raw gated probabilities for this candidate. Fold-local residual blending is also not consistently beneficial; selected beta varies substantially across seeds/domains.

Pair-level error analysis explains the remaining bottlenecks. On first_batch, gated fixes 1,837 residual FP pairs but introduces 1,594 new FP pairs; it fixes 200 FN and introduces 214 FN, so the net change is a broad boundary rearrangement rather than a clean recall gain. On official set2, gated fixes 107 FN and introduces only 10 FN, while FP changes are nearly balanced (20 fixed, 24 new). Its set2 benefit is therefore primarily reduced fragmentation, but the remaining `bug_107` split is systematic across many seeds.

Gated MLP is now the leading unified-LODO architecture candidate, but it does not replace the historical calibrated residual blend default. The next useful work is a validation-trained multi-seed/model selector or a targeted fragmentation objective for set2-like large bugs, not additional global temperature or k-factor sweeps.

Outputs:

- `/tmp/all_dataset_arch_lodo_0_9_summary.csv`
- `/tmp/all_dataset_arch_lodo_10seed_probability_ensemble.csv`
- `/tmp/all_dataset_inner_calibrated_0_9/`
- `/tmp/all_dataset_inner_calibrated_fixedk_0_9/`
- `/tmp/gated_lodo_error_analysis.md`

##### Fragmentation-Targeted Objective Screen

A focused follow-up tested whether static hard-positive quotas and graph-level losses can repair official set2 fragmentation while keeping the same dual-embedding Gated MLP, fixed k, and average agglomerative clustering. Hard positives are the lowest-similarity 40% of same-bug edges. Connectivity loss requires each case to retain at least top-m strong same-bug links; the prototype surrogate separates a bug's sampled positive-edge mean from its nearest sampled negative edges.

Official set2 seed-0 results:

| objective | BA | TPR | TNR |
|---|---:|---:|---:|
| baseline gated | **0.9170** | **0.9487** | 0.8852 |
| static hard-positive quota | 0.7201 | 0.5385 | 0.9016 |
| hard positive + connectivity | **0.9170** | **0.9487** | 0.8852 |
| hard positive + prototype surrogate | 0.6944 | 0.4872 | 0.9016 |

On previously weak seeds 1 and 5, connectivity raises mean BA only from `0.7161` to `0.7201`; TPR remains `0.5385`, so it does not repair the systematic `bug_107` split. Static low-similarity positive oversampling is too broad and severely damages recall. The next implementation should mine same-bug edges from out-of-fold low model probability and known fragmented training clusters, then apply connectivity only to those targeted bridge edges. Do not enable the current hard-positive or prototype variants.

##### OOF Model-Aware Bridge-Edge Mining

Replaced static hard-positive mining with **dataset-aware out-of-fold (OOF) bridge-edge mining**. The idea: use an OOF model to identify which same-bug pairs the model actually splits (bridge edges), then apply a light auxiliary BCE loss to reunite them during final training.

**Core concepts:**

- **OOF prediction**: Leave-one-training-dataset-out. For each training dataset `d`, train on the other 5 datasets, predict pair probabilities on `d`, and cluster to get OOF predicted labels.
- **Bridge edge**: A same-bug pair `(i,j)` where the OOF model splits cases `i` and `j` into different fragments and assigns low `P_oof(i,j)`. These are the fragmentation frontiers the model itself identifies.
- **Bridge loss**: Auxiliary BCE with target=1 applied to bridge edge pairs during final training. Total loss = `focal_loss + bridge_weight * bridge_loss`.

**Implementation files:**

| File | Purpose |
|------|---------|
| `oof_bridge_mining.py` | `mine_oof_bridge_edges()`, `bridge_quality_score()`, `fragmentation_rows()` |
| `run_oof_bridge_experiments.py` | Full experimental pipeline: OOF generation, bridge mining, training, evaluation |
| `analyze_bridge_instability.py` | Root-cause analysis comparing good vs catastrophic seeds |

**Baseline bridge experiment (set2, seeds 0/1/5):**

| Config | Mean BA | Std | Worst BA | TPR | TNR | bug_107 fragments |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| raw gated | 0.6989 | 0.026 | 0.6620 | 0.527 | 0.871 | 2 |
| bridge w=0.3 | **0.8265** | 0.116 | 0.6623 | 0.801 | 0.853 | →1 (2/3 seeds) |

Bridge w=0.3 fixes bug_107 fragmentation on seeds 0 and 5 (BA ~0.91), but seed 1 remains catastrophic (BA 0.662→0.633, bug_107 goes from 2→3 fragments). Mean BA is high but worst-seed BA disqualifies it from mainline consideration.

**Root-cause analysis** (`/tmp/oof_bridge_set2/bridge_instability_report.md`):

The instability is caused by OOF model quality variance across seeds:

| Metric | seed 0 (good) | seed 1 (bad) | seed 5 (good) |
|--------|:---:|:---:|:---:|
| Raw gated BA (set2) | 0.720 | 0.662 | 0.715 |
| stage3 OOF BA | **0.823** | **0.754** | 0.840 |
| stable OOF BA | 0.648 | **0.501** | 0.517 |
| Mined bridge edges | 424 | 511 | 339 |

Seed 1's OOF model quality is systematically worse across training datasets. Worse OOF → more false fragments → more (+noisier) bridge edges → collateral damage. Seed 1's bridges have 87 additional edges (511 vs 424) concentrated in bugs that weren't fragmented in better seeds.

**Stabilization attempts (all failed on seed 1):**

Five mitigation strategies were implemented and tested:

| Strategy | CLI flags | Effect on seed 1 |
|----------|-----------|:---:|
| Quality-weighted bridge loss | `--bridge-quality-weighted` | BA 0.662→0.633 ✗ |
| Budget cap (150 edges) | `--bridge-max-edges-total 150` | BA 0.662→0.633 ✗ |
| Quality + budget combined | both | BA 0.662→0.625 ✗ |
| Hardest fragments only | `--bridge-hardest-fragments` | BA 0.662→0.633 ✗ |
| Quality w=0.2 + budget=150 | combined | **Seed 0/5 BA ~0.91**, but seed 1 still 0.625 |

`bridge_quality_score()` combines OOF confidence, signal agreement (primary_type, mismatch_type), primary_signature match, conflict penalty, fragment size, and OOF reliability into a 0~1 weight. On good seeds, quality-weighted bridge modestly improves over uniform (seed 5: 0.917 vs 0.910). However, no filtering strategy can rescue seed 1 because the underlying OOF predictions are too noisy.

**Key findings:**

1. Bridge edge mining **works extremely well when OOF quality is sufficient** — seeds 0 and 5 achieve BA ~0.91 with bug_107 fully repaired.
2. Bridge edge mining **catastrophically fails when OOF quality is poor** — seed 1's TNR collapses from 0.820 to 0.771.
3. **Quality scoring, budget capping, and hardest-fragment filtering do not fix the instability.** The root cause is OOF model quality, not bridge edge selection.
4. **Budget alone is dangerous** — capping edges without quality guidance loses critical bridges (seed 5: 0.910→0.700 with budget=150 alone).
5. **Quality + budget is the best combo** on good seeds — matches full-bridge BA with 20% fewer edges.

**Verdict: Do not promote bridge loss to mainline.** The catastrophic seed instability violates the "worst BA must improve" criterion. Bridge loss should remain experimental. The correct path forward, if bridge is pursued, requires:

- **OOF quality gate**: skip bridge mining when per-dataset OOF BA < 0.70
- **Multi-seed OOF ensemble**: average OOF predictions across seeds to reduce noise
- **Per-bug OOF check**: don't bridge bugs with >4 OOF fragments (OOF too unreliable for that bug)

**Outputs:**

- `/tmp/oof_bridge_set2/` — baseline bridge experiment (seeds 0/1/5, set2)
- `/tmp/oof_bridge_set2/bridge_instability_report.md` — root-cause analysis
- `/tmp/oof_bridge_stable/` — stabilization experiment (quality, budget, hardest variants)
- `/tmp/oof_bridge_set2/oof_cache/` — cached OOF predictions (reusable)

##### Trace Structured Features (Route A)

Implemented lightweight deterministic trace features injected as extra pairwise feature blocks into the existing Gated MLP. No transformer, no LLM, no new backbone.

**Approach:** Parse RISC-V trace.log files into structured summaries, then derive pairwise features:
- **Tail features** (20 dims): opcode histogram cosine, PC region jaccard, load/store/branch ratios, tight loop detection, exception markers
- **Anchor features** (16 dims): opcode/PC context around mismatch location, branch/CSR patterns
- **Sequence stats** (16 dims): opcode/PC entropy, load-store ratio, compressed ratio, CSR density, loop density
- **Combined tail_anchor** (36 dims) added to the existing 294-dim dual-embedding pair vector → 338-dim input

**Implementation files:**

| File | Purpose |
|------|---------|
| `trace_structured_features.py` | RISC-V trace parser, summary extraction, pairwise feature builders |
| `pairwise_llm_features.py` | Added `FEATURE_MODE=llm_dual_struct_det_summary_trace_struct` (338 dims) |
| `run_trace_completion_experiments.py` | Experiment runner for trace ablation |

**Experiment results (set2/stable/VCS, seeds 0/1/2):**

| dataset | mode | mean BA | std | worst | best | TPR | TNR |
|---------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| set2 | none | **0.7504** | 0.137 | 0.666 | 0.908 | 0.641 | 0.860 |
| set2 | trace | 0.6698 | 0.025 | 0.655 | 0.699 | 0.507 | 0.832 |
| VCS | none | 0.6389 | 0.090 | 0.587 | 0.743 | 0.575 | 0.703 |
| VCS | trace | **0.6908** | 0.090 | 0.587 | 0.743 | 0.608 | 0.773 |
| stable | none | 0.5525 | 0.022 | 0.527 | 0.566 | 0.361 | 0.744 |
| stable | trace | **0.5870** | 0.096 | 0.504 | 0.693 | 0.544 | 0.630 |

**Key findings:**

1. **Trace features are dataset-dependent**: VCS benefits (+5.2% mean BA, +7.0% TNR), stable marginally benefits (+3.5% mean BA), but **set2 is hurt** (-8.1% mean BA).
2. **Trace features are destabilizing**: On set2 seed 2, trace collapses BA from 0.908→0.655. On stable, worst BA drops from 0.527→0.504.
3. **VCS benefits most from trace**: VCS has hardware-level CSR/trap failures where trace behavioral signals (CSR density, exception markers) provide discriminative information beyond sim/regr logs.
4. **set2 regresses with trace**: set2 bugs are primarily logic/debug mismatches where trace behavior is similar across cases; the 36 extra trace dims introduce noise that masks the stronger sim/regr signals.
5. **Variance increases**: Trace features amplify seed-to-seed variance on stable (std 0.022→0.096).
6. **No completion LLM available**: Only an embedding endpoint (nomic-embed-text-v1.5 at port 8001). No completion-capable endpoint was found. Route B (completion JSON) and Route C (trace+completion fusion) are blocked.

**Verdict: Do not enable trace features by default.** Trace is beneficial only for VCS-like hardware-level datasets. For set2 (the primary fragmentation target), trace features are harmful. If selectively enabled, a dataset-aware gate or feature selection mechanism would be needed.

**Outputs:**

- `/tmp/trace_struct_exp/results.csv` — full trace ablation results
- `/tmp/trace_struct_exp/summary.csv` — per-dataset per-mode summary


## Directed Cross v2 Unified LODO Smoke (July 2026)

`official_format_fake_dataset/directed_cross_v2` was added as an eighth independent
LODO episode. It contains 37 cases and five labels; `gold.csv` and `golden.csv`
match exactly. Pair construction remains strictly within each dataset.

The existing beta deterministic fallback with reference `k=5` reached BA 0.647689
(TPR 0.571429, TNR 0.723949). A CPU-only seed-0 strict-LODO smoke trained on the
other seven datasets, used both 768-dimensional features/summary embeddings, and
held directed_cross_v2 out for final scoring:

| objective | BA | TPR | TNR |
|---|---:|---:|---:|
| gated MLP balanced | 0.821463 | 0.798319 | 0.844607 |
| gated MLP hard-positive connectivity | 0.821463 | 0.798319 | 0.844607 |
| gated MLP hard-positive prototype | 0.803742 | 0.764706 | 0.842779 |

This four-epoch smoke is not a model-selection result, but it shows that the unified
gated architecture transfers well to the new domain. The next formal run is an
eight-dataset LODO seeds 0-4 comparison of balanced and hard-positive connectivity,
followed by seeds 0-9 for the winner. It must use at most two genuinely idle GPUs.

## Graph-Aware Clustering and Multi-View Embedding Experiments (July 2026)

This route is experimental only. The formal `regr_fail_bucketing.py --input --output --k`
path is unchanged and still does not read gold/meta/trace by default.

### Stage 1: graph-aware clustering on fixed dual-view probabilities

`graph_clustering.py` adds five clustering backends:

| method | purpose |
|---|---|
| `agglomerative_avg` | existing fixed-k average-linkage baseline |
| `agglomerative_complete` | conservative complete-linkage to reduce single-edge false merges |
| `conservative_merge` | limited postprocess merge from average-linkage clusters |
| `mutual_knn_cc` | mutual-kNN connected components with fixed-k repair |
| `signed_graph_greedy` | fixed-k greedy signed-graph node moves with structured conflict penalty |

On the available eight-dataset LODO probability cache (seeds 0-4 where present),
`signed_graph_greedy` was the best mean-BA graph-only replacement:

| graph | source | mean BA | worst BA | mean TPR | mean TNR | runs |
|---|---|---:|---:|---:|---:|---:|
| signed_graph_greedy | hard_pos_connect | **0.7014** | 0.5283 | 0.6101 | 0.7927 | 40 |
| signed_graph_greedy | balanced | 0.6914 | 0.5383 | 0.5982 | 0.7845 | 40 |
| agglomerative_complete | balanced | 0.6889 | **0.6055** | 0.5508 | **0.8270** | 40 |
| agglomerative_avg | balanced | 0.6766 | 0.5463 | 0.5905 | 0.7628 | 40 |

Key finding: signed-graph improves mean BA mostly by raising TNR, while complete-link
is the most conservative and improves worst BA/TNR at the cost of recall. The current
`conservative_merge` defaults are too aggressive and can collapse clusters on some
datasets, so they are not recommended.

Outputs:

- `/tmp/graph_multiview_stage1_available_s0_4/summary.csv`
- `/tmp/graph_multiview_stage1_available_s0_4/results.csv`
- `/tmp/graph_multiview_stage1_available_s0_4/cluster_diagnostics.csv`

### Stage 2: multi-view embedding ablation

`run_graph_multiview_experiments.py` now supports real multi-view LODO training with
GBDT/logistic pair models. Additional embedding views are built from sim/regr only:

| view | content |
|---|---|
| `features` | existing high-signal feature document |
| `summary` | existing case summary document |
| `event` | failure event order and tags: fatal/error/mismatch/timeout/debug/irq/csr |
| `object` | hardware/software objects: PC region, opcode pair, register, source file, UVM component |
| `context` | compact local sim/regr signal windows around fatal/error/mismatch lines |

Each view is embedded with the existing embedding endpoint, reduced to 64 dimensions,
and converted to pair relation features (`abs(diff)`, product, cosine, distance). No
trace or completion LLM is used.

Four-dataset 0-4 focus run (`set1`, `set2`, `VCS`, `stable`) showed that more views are
useful but not uniformly safe:

| graph | view config | mean BA | worst BA | mean TPR | mean TNR | dataset means |
|---|---|---:|---:|---:|---:|---|
| complete | quad_event_object_context | **0.7454** | **0.5761** | 0.7143 | **0.7766** | set1 0.8889, set2 0.7911, VCS 0.7257, stable 0.5761 |
| average | quad_event_object_context | 0.7085 | 0.5597 | **0.7143** | 0.7027 | set1 0.6611, set2 0.8704, VCS 0.7428, stable 0.5597 |
| average | tri_object | 0.6594 | 0.5174 | 0.6256 | 0.6932 | set1 0.6444, set2 0.8327, VCS 0.6432, stable 0.5174 |
| signed_graph | quad_event_object_context | 0.6685 | 0.5284 | 0.6552 | 0.6818 | set1 0.6500, set2 0.7529, VCS 0.7428, stable 0.5284 |
| average | dual | 0.5901 | 0.5278 | 0.5068 | 0.6735 | set1 0.5278, set2 0.5570, VCS 0.6005, stable 0.6753 |

Eight-dataset seed-0 sanity with `quad_event_object_context + complete` reached mean
BA **0.7563** versus **0.7319** for the same lightweight dual-view GBDT + complete
setup. The follow-up eight-dataset seeds 0-4 run is the stronger estimate:

| view config | mean BA | worst BA | mean TPR | mean TNR | key dataset means |
|---|---:|---:|---:|---:|---|
| quad_event_object_context | **0.7460** | 0.6529 | 0.6205 | **0.8715** | set1 0.8333, set2 0.6690, stage2 0.8161, stable 0.6529 |
| dual | 0.7431 | **0.6672** | **0.6323** | 0.8539 | set1 0.7778, set2 0.6690, stage2 0.7398, stable 0.7218 |
| tri_object | 0.7375 | 0.6459 | 0.6186 | 0.8564 | set1 0.7778, set2 0.6459, stage2 0.7461, VCS 0.6892 |

Per-dataset deltas for `quad_event_object_context` vs dual on seeds 0-4:

| dataset | delta BA | note |
|---|---:|---|
| stage2 | +0.0762 | strong and stable gain, mainly higher TPR while preserving TNR |
| set1 | +0.0556 | improves small official benchmark on average |
| first | +0.0176 | modest gain |
| VCS | +0.0099 | small gain |
| set2 | +0.0000 | neutral overall, high seed variance |
| stage3 | -0.0044 | near-neutral/slight loss |
| directed_cross | -0.0629 | significant TPR loss |
| stable | -0.0689 | significant TPR loss |

Key finding: multi-view embeddings carry real additional signal, especially for
stage2 and set1, but fixed concatenation is not robust enough to become the mainline.
The object/context views can over-separate stable and directed_cross by lowering TPR.
This argues for a learned/selective view gate rather than blindly concatenating all
views as the new mainline.

Current recommendation: keep the existing calibrated dual-input blend as the stable
submission path. Promote graph-aware clustering and multi-view embedding to candidate
experimental backends. The next useful step is a view-gated Gated MLP that learns when
`object/context` views should override the base dual signal, with fake-dataset guardrails.

Outputs:

- `/tmp/graph_multiview_stage2_s0_4_official_vcs_stable/summary.csv`
- `/tmp/graph_multiview_stage2_s0_4_official_vcs_stable_complete/summary.csv`
- `/tmp/graph_multiview_stage2_s0_4_official_vcs_stable_signed/summary.csv`
- `/tmp/graph_multiview_stage2_seed0_all8_complete/summary.csv`


### Stage 3: conservative dual/multi-view probability blend

A learned pair-level view gate was implemented experimentally, but seed-0 all-dataset
sanity did not beat the simpler fixed blend. It reached mean BA 0.7471 with mixed
behavior: set2 and stage3 improved, but first/stable remained weak. The gate tended to
trust the expert view too much on stable-like data.

A cheaper cached-probability blend was then evaluated using the seeds 0-4 dual and
`quad_event_object_context` probability matrices from Stage 2:

```text
P_blend = (1 - beta) * P_dual + beta * P_quad_event_object_context
```

Eight-dataset seeds 0-4, complete-link clustering:

| expert | beta | mean BA | worst BA | mean TPR | mean TNR | key dataset means |
|---|---:|---:|---:|---:|---:|---|
| quad_event_object_context | 0.75 | **0.7543** | 0.6522 | 0.6360 | **0.8726** | set1 0.8333, stage2 0.8111, stage3 0.7981, stable 0.6522 |
| quad_event_object_context | 0.50 | 0.7513 | **0.6693** | **0.6434** | 0.8592 | set1 0.7778, stage2 0.7960, stage3 0.8017, stable 0.6998 |
| quad_event_object_context | 0.25 | 0.7428 | 0.6622 | 0.6343 | 0.8513 | stage3 0.8092, stable 0.7007 |
| tri_object | 0.50 | 0.7371 | 0.6671 | 0.6319 | 0.8422 | directed_cross 0.7905, stage3 0.8013 |

Compared with the Stage 2 dual baseline (`mean BA 0.7431`, worst BA 0.6672), the
`quad_event_object_context` blend is the first multi-view variant that improves mean
BA without worsening worst BA. `beta=0.50` is the preferred candidate because it keeps
stable/directed_cross much safer than `beta=0.75` while still preserving most of the
stage2/stage3 gain.

The follow-up seeds 5-9 run retrained the same lightweight GBDT pair models on all
eight LODO folds, restricted to `dual` and `quad_event_object_context`:

| view config | seeds | mean BA | worst BA | mean TPR | mean TNR | key dataset means |
|---|---:|---:|---:|---:|---:|---|
| quad_event_object_context | 5-9 | **0.7518** | 0.6269 | **0.6310** | **0.8727** | set1 0.7944, set2 0.7327, stage2 0.8040, stable 0.6269 |
| dual | 5-9 | 0.7243 | **0.6606** | 0.6049 | 0.8438 | set1 0.6611, set2 0.6963, stage2 0.7442, stable 0.6606 |

`quad_event_object_context` again improved mean BA, especially on set1/set2/stage2
and VCS, but its stable/directed_cross recall remained weaker. This confirms that the
additional views are useful signals but should be blended conservatively.

Combining the 0-4 and 5-9 probability caches, the fixed-beta blend gives the current
best multi-view 0-9 estimate:

| expert | beta | mean BA | worst BA | mean TPR | mean TNR | key dataset means |
|---|---:|---:|---:|---:|---:|---|
| quad_event_object_context | 0.50 | **0.7587** | **0.6846** | **0.6510** | 0.8664 | set1 0.8056, set2 0.6846, stage2 0.7884, stage3 0.8093, stable 0.6910 |
| quad_event_object_context | 0.75 | 0.7581 | 0.6653 | 0.6410 | **0.8751** | set1 0.8333, set2 0.7004, stage2 0.8117, stable 0.6653 |
| quad_event_object_context | 0.25 | 0.7422 | 0.6709 | 0.6320 | 0.8523 | stage3 0.8136, directed_cross 0.7631, stable 0.6907 |

The LODO beta selector that chooses beta using only the non-target datasets did not
beat the fixed `beta=0.50` setting. Its guarded/worst policies reached mean BA 0.7555
and worst BA 0.6653, mostly because they sometimes selected `beta=0.75` and over-split
stable-like cases. Fixed `beta=0.50` is therefore the preferred multi-view candidate
for now.

Current recommendation: do not promote the learned pair gate yet. Use fixed beta=0.5
as the strongest multi-view candidate. The next improvement should target its remaining
failure mode with hard-positive/connectivity training or a stronger gate that explicitly
protects same-bug recall on stable/directed_cross-like data.

Outputs:

- `/tmp/graph_multiview_gate_seed0_all8/summary.csv`
- `/tmp/graph_multiview_gate_seed0_all8/gate_debug.csv`
- `/tmp/graph_multiview_blend_s0_4_all8_complete/summary.csv`
- `/tmp/graph_multiview_blend_s0_4_all8_complete/results.csv`
- `/tmp/graph_multiview_stage2_s5_9_all8_complete/summary.csv`
- `/tmp/graph_multiview_blend_s0_9_all8_complete/summary.csv`

