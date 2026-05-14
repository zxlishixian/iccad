# EDA Regression Failure Bucketing Baseline

这个项目文件夹用于处理一个 EDA/ICCAD 风格的回归失败聚类赛题：给定一批 Ibex RISC-V CPU 回归仿真的失败日志，自动把 case 分到若干个 bucket 中，使同一根因 bug 导致的失败尽量落在同一个 bucket。

## 文件夹内容

- `B_20260212.pdf`
  - 赛题说明 PDF。
  - 其中定义了输入输出格式、日志类型、benchmark 规模、运行时间限制和提交程序接口。

- `dataset/`
  - 本地样例数据集。
  - 包含三个 benchmark 规模：
    - `first_batch_dataset/`：80 cases，8 个 bug bucket。
    - `stage2_dataset_working/`：240 cases，16 个 bug bucket。
    - `stage3_dataset_32bugs_640cases/`：640 cases，32 个 bug bucket。
  - 每个 benchmark 目录中包含：
    - `input.csv`：待聚类输入，列为 `trace_log,sim_log,regr_log`。
    - `gold.csv`：本地验证用答案，列为 `case_id,bug_id`。
    - `meta.csv`：case 元信息，包括 bug、group、family、test、seed 等。
    - `cases/case_xxxxxx/`：每个 case 的日志目录，包含 `sim.log`、`regr.log`、`trace.log`、`meta.json`。

- `dataset/Ibex 数据集生成手册.md`
  - 说明这些 Ibex 数据集如何生成。

- `dataset/根因说明.md`
  - 说明已经注入并运行过的 32 个 bug，以及它们所属的 group/family。

- `regr_fail_bucketing.py`
  - 当前实现的 baseline 聚类程序。
  - 支持 `simple` / `drain` parser 和 `kmeans` / `agglomerative` / `hdbscan` clustering backend。
  - 正式预测只读取 `input.csv` 指向的 `sim.log` / `regr.log`，不读取 `gold.csv` 或 `meta.csv`。
  - 当前默认主线为 `primary_signature + drain + agglomerative`。

- `regr_fail_bucketing`
  - 轻量 wrapper，提供题面要求的可执行程序形式。
  - 如果当前目录存在 `.venv/bin/python`，会优先使用本地虚拟环境。

- `error_analysis.py`
  - 聚类结果误差分析脚本。
  - 对比 `gold.csv` 和预测 `output.csv`，输出 gold bug fragmentation、predicted bucket purity、top FN bugs、top FP bucket pairs。

- `requirements.txt`
  - Python 依赖列表，目前使用 `scikit-learn`。

- `run_experiments.py`
  - 本地 ablation 脚本，会在三个 bundled dataset 上跑 `simple+kmeans`、`simple+agglomerative`、`drain+kmeans`、`drain+agglomerative`。

- `train_token_weights.py`
  - 使用本地训练集的 `gold.csv` 学习 token 权重，并输出 `token_weights.json`。
  - 只有训练阶段读取 `gold.csv`。

- `run_supervised_experiments.py`
  - leave-one-dataset-out 实验脚本，训练 token weights 后在 held-out dataset 上评估泛化。

## 赛题输入输出理解

题面要求提交程序接口为：

```bash
regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

输入 CSV 每行对应一个 case，包含三个日志路径：

- `trace.log`
  - Ibex tracer 生成的逐指令 retire 日志。
  - 当前 baseline 暂时不使用。

- `sim.log`
  - 仿真控制台日志，包含 Xcelium/VCS 启动信息、UVM 日志、`UVM_INFO`、`UVM_ERROR`、`UVM_FATAL`、测试通过/失败 verdict 等。

- `regr.log`
  - 回归报告摘要，包含 PASS/FAILED 统计、失败测试名、`rtl_sim.log` 摘要、mismatch 抽取等。

输出 CSV 只需要一列：

```csv
bucket
bucket_000
bucket_001
...
```

bucket ID 可以是任意字符串，评价时看 case 两两之间是否被正确判定为同桶或异桶，不要求 bucket 名字和 `bug_id` 一致。

## 已经完成的工作

1. 阅读了赛题 PDF，确认：
   - 输入是三类日志路径 CSV。
   - 输出是单列 bucket CSV。
   - 接口是 `--input <input.csv> --output <output.csv> --k <k>`。
   - benchmark 有不同规模，最大运行时间从 30s 到 300s 不等。
   - 超时或运行失败的 benchmark 得分为 0。

2. 阅读了 dataset 中的说明文档：
   - `Ibex 数据集生成手册.md`
   - `根因说明.md`

3. 检查了 dataset 目录结构：
   - 三档本地数据分别为 80、240、640 cases。
   - 每个 benchmark 都按 `input.csv`、`gold.csv`、`meta.csv`、`cases/` 组织。

4. 打开了小规模 benchmark 的 `input.csv`，确认实际列名为：

```csv
trace_log,sim_log,regr_log
```

5. 随机查看了多个 case 的 `sim.log` 和 `regr.log`，确认日志中常见高信号模式包括：
   - `UVM_FATAL`
   - `UVM_ERROR`
   - `Cosim mismatch`
   - `Register write data mismatch`
   - `PC mismatch`
   - `--- RISC-V UVM TEST FAILED ---`
   - `[FAILED]: error seen in 'rtl_sim.log'`
   - PASS/FAILED 统计行

6. 在完成上述阅读后，实现了一个不使用 `trace.log`、不调用 LLM、基于 `scikit-learn` 的 baseline pipeline。

## 当前 baseline 方法

当前程序流程：

```text
input.csv
  -> 读取 sim.log / regr.log
  -> 抽取 primary_signature
  -> 选择高信号日志行
  -> SimpleDrain 或 fixed-depth Drain 模板化
  -> case-level template/token/count 特征
  -> 可选加载 token_weights.json 做 repeat/drop 加权
  -> sklearn FeatureHasher
  -> sklearn TfidfTransformer
  -> sklearn MiniBatchKMeans 或 AgglomerativeClustering 聚类
  -> output.csv
```

模板化会归一化：

- 绝对路径
- case 编号
- seed
- 时间戳
- 大整数
- 十六进制地址/数据
- 寄存器名
- 行号前缀

程序的日志解析部分使用 Python 标准库，向量化和聚类使用 `scikit-learn`。本地已经创建 `.venv` 并安装了 `scikit-learn`、`numpy`、`scipy`、`joblib` 等依赖。

如果 `scikit-learn` 不存在，程序会打印 warning 到 stderr，并回退到标准库 hashing TF-IDF + k-means fallback，不会直接崩溃。

如果需要重新安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 使用方式

直接运行 wrapper：

```bash
./regr_fail_bucketing \
  --input dataset/first_batch_dataset/input.csv \
  --output /private/tmp/first_out.csv \
  --k 8
```

或者直接用本地虚拟环境运行 Python 脚本：

```bash
.venv/bin/python regr_fail_bucketing.py \
  --input dataset/stage3_dataset_32bugs_640cases/input.csv \
  --output /private/tmp/stage3_out.csv \
  --k 32 \
  --parser drain \
  --cluster agglomerative
```

输出文件行数应当等于输入 case 数加 1 行表头。

新增可调参数：

- `--parser simple|drain`
- `--cluster kmeans|agglomerative|hdbscan`
- `--drain-depth <int>`
- `--drain-st <float>`
- `--drain-max-children <int>`
- `--svd-dim <int>`
- `--token-weights <token_weights.json>`
- `--token-weight-mode repeat|none`
- `--cluster-factor <float>`

默认配置为：

```text
--parser drain
--cluster agglomerative
--drain-depth 4
--drain-st 0.45
--drain-max-children 100
--svd-dim 128
--token-weight-mode repeat
--cluster-factor 1.0
```

训练 token weights：

```bash
.venv/bin/python train_token_weights.py \
  --datasets dataset/first_batch_dataset dataset/stage2_dataset_working dataset/stage3_dataset_32bugs_640cases \
  --output /tmp/token_weights_all.json
```

使用 token weights 推理：

```bash
.venv/bin/python regr_fail_bucketing.py \
  --input dataset/stage3_dataset_32bugs_640cases/input.csv \
  --output /tmp/stage3_out.csv \
  --k 32 \
  --parser drain \
  --cluster agglomerative \
  --token-weights /tmp/token_weights_all.json \
  --cluster-factor 1.25
```

`token_weights.json` 是由本地训练集 `gold.csv` 学得的特征权重。正式预测只加载 `token_weights.json`，不会读取 `gold.csv` 或 `meta.csv`。

运行所有本地 ablation：

```bash
.venv/bin/python run_experiments.py \
  --python .venv/bin/python \
  --output-dir /private/tmp/regr_fail_bucketing_experiments
```

运行 supervised leave-one-dataset-out：

```bash
.venv/bin/python run_supervised_experiments.py \
  --python .venv/bin/python \
  --output-dir /tmp/supervised_exp \
  --cluster-factors 1.0 1.25 1.5
```

## Error Analysis

当 TNR 很高但 TPR 很低时，通常说明模型能把不同 bug 区分开，却没有把同一个 bug 的不同表现合并起来，也就是 false negative 多、golden bucket 被拆碎。可以用 `error_analysis.py` 定位具体问题：

```bash
python3 error_analysis.py \
  --gold dataset/stage3_dataset_32bugs_640cases/gold.csv \
  --pred /private/tmp/sklearn_stage3_final.csv \
  --top 12 \
  --out /private/tmp/stage3_error_analysis.md
```

报告包含：

- `Gold Bug Fragmentation`：每个 golden bug 被拆到多少个预测 bucket、最大预测桶占比、贡献了多少 FN pairs。
- `Predicted Bucket Purity`：每个预测 bucket 混入了多少个 golden bug、主导 bug 占比、贡献了多少 FP pairs。
- `Top FN Bugs`：最严重的同源 case 拆分问题。
- `Top FP Bucket Pairs`：同一个预测 bucket 中最严重的不同 bug 混合对。

## 已验证结果

本地粗略验证结果。当前新版默认包含 `primary_signature`。

无权重 baseline，`drain + agglomerative + cluster_factor=1.0`：

| dataset | BA | TPR | TNR | pred clusters | runtime_sec |
|---|---:|---:|---:|---:|---:|
| `first_batch_dataset` | 0.765119 | 0.741667 | 0.788571 | 8 | 1.284 |
| `stage2_dataset_working` | 0.740608 | 0.715476 | 0.765741 | 16 | 1.909 |
| `stage3_dataset_32bugs_640cases` | 0.736150 | 0.849342 | 0.622959 | 32 | 3.089 |

全数据训练 token weights 后的 sanity check：

| dataset | cluster_factor | BA | TPR | TNR | pred clusters |
|---|---:|---:|---:|---:|---:|
| `first_batch_dataset` | 1.0 | 0.743452 | 0.808333 | 0.678571 | 8 |
| `first_batch_dataset` | 1.25 | 0.756627 | 0.786111 | 0.727143 | 10 |
| `first_batch_dataset` | 1.5 | 0.745079 | 0.744444 | 0.745714 | 12 |
| `stage2_dataset_working` | 1.0 | 0.676937 | 0.579762 | 0.774111 | 16 |
| `stage2_dataset_working` | 1.25 | 0.676615 | 0.576786 | 0.776444 | 20 |
| `stage2_dataset_working` | 1.5 | 0.675681 | 0.573214 | 0.778148 | 24 |
| `stage3_dataset_32bugs_640cases` | 1.0 | 0.687832 | 0.638980 | 0.736683 | 32 |
| `stage3_dataset_32bugs_640cases` | 1.25 | 0.689299 | 0.590296 | 0.788301 | 40 |
| `stage3_dataset_32bugs_640cases` | 1.5 | 0.690443 | 0.579934 | 0.800953 | 48 |

Leave-one-dataset-out supervised token weighting：

| train | test | cluster_factor | BA | TPR | TNR | pred clusters |
|---|---|---:|---:|---:|---:|---:|
| stage2+stage3 | first | 1.0 | 0.737183 | 0.797222 | 0.677143 | 8 |
| stage2+stage3 | first | 1.25 | 0.727063 | 0.755556 | 0.698571 | 10 |
| stage2+stage3 | first | 1.5 | 0.722500 | 0.725000 | 0.720000 | 12 |
| first+stage3 | stage2 | 1.0 | 0.672693 | 0.586905 | 0.758481 | 16 |
| first+stage3 | stage2 | 1.25 | 0.669393 | 0.576786 | 0.762000 | 20 |
| first+stage3 | stage2 | 1.5 | 0.668980 | 0.574405 | 0.763556 | 24 |
| first+stage2 | stage3 | 1.0 | 0.521272 | 0.877303 | 0.165242 | 32 |
| first+stage2 | stage3 | 1.25 | 0.531874 | 0.875164 | 0.188584 | 40 |
| first+stage2 | stage3 | 1.5 | 0.642865 | 0.778618 | 0.507112 | 48 |

当前整体最稳的是无权重 `primary_signature + drain + agglomerative + cluster_factor=1.0`。监督 token weights 对部分小数据集能提升 TPR，但 leave-one 泛化存在过拟合，尤其 first+stage2 训练后测试 stage3 时 TNR 很低。

这些分数只是当前轻量 baseline 的 sanity check，不代表最终可提交最优方案。

## 当前限制

- 暂时没有使用 `trace.log`，因此没有利用逐指令执行轨迹中的行为模式。
- 没有使用 LLM 或 embedding。
- 当前使用的是通用文本特征和 `MiniBatchKMeans`，还没有针对硬件日志做强监督或半监督优化。
- 主要依赖 `sim.log` / `regr.log` 中的文本症状，遇到同一 bug 多种表象或不同 bug 相似表象时容易混淆。
- 当前没有做针对 `meta.csv` 或 `gold.csv` 的训练式特征学习；正式评测时也不应依赖答案文件。

## 后续可改进方向

- 加入 `trace.log` tail/head 特征，特别是失败前的 PC、指令类型、访存模式、寄存器写回模式。
- 对 `UVM_FATAL`、`Cosim mismatch`、`Register write data mismatch`、`PC mismatch` 等错误类型做更细粒度解析。
- 将 case 的第一处 mismatch 结构化，例如 mismatch 类型、寄存器编号、DUT/expected 值形态。
- 对不同日志来源设置不同权重：`regr.log` 的失败摘要通常比仿真启动头部更重要。
- 增加聚类后处理，例如合并过小簇、按近邻关系重新分配孤立 case。
- 在允许依赖的环境中尝试 `sklearn` 的 `TfidfVectorizer`、`MiniBatchKMeans` 或层次聚类。
