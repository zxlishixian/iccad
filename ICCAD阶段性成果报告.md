# ICCAD Regression Failure Bucketing 阶段性成果报告

更新时间：2026-07-07

## 1. 项目目标与正式约束

本项目对应 ICCAD/EDA regression failure bucketing 任务：给定一组 regression failure case 的日志，将同一个真实 bug 导致的失败聚到同一个 bucket。

正式预测入口保持不变：

```bash
python regr_fail_bucketing.py --input <input.csv> --output <output.csv> --k <k>
```

核心约束：

- 默认正式路径不能读取 `gold.csv` / `golden.csv` / `meta.csv`。
- 默认正式路径不读取 `trace.log`。
- 默认 baseline 必须稳定可运行。
- LLM completion 只能作为 experimental，不直接输出 bucket。
- LLM embedding 可以作为 experimental 特征输入。
- Alpha/Beta 提交包必须保证 `Case,bucket` 输出格式、运行时间和异常 fallback。

当前正式默认 baseline 仍然是 deterministic no-trace pipeline：

```text
sim.log / regr.log
-> primary_signature + FixedDepthDrain
-> TF-IDF / SVD
-> clustering
-> Case,bucket
```

所有深度学习、LLM、trace、completion、signed graph 路线都仍标记为 experimental。

## 2. 数据集现状

目前项目中实际使用的数据大致分为四类。

### 官方公开 benchmark

- `test_case/problem/benchmark_set_1`
- `test_case/problem/benchmark_set_2`

官方后来补充了标准答案：

- `test_case/problem/benchmark_set_1/golden.csv`
- `test_case/problem/benchmark_set_2/golden.csv`

这两个集合非常小，但风格最接近真实评测，是当前优化优先级最高的数据。

### 旧 fake datasets

- `fake_dataset/first_batch_dataset`
- `fake_dataset/stage2_dataset_working`
- `fake_dataset/stage3_dataset_32bugs_640cases`

这些数据规模更大，适合训练 pairwise 模型和做稳定性验证，但和官方日志分布存在明显差异。

### official-format fake datasets

- `official_format_fake_dataset/official_vcs_stage1_dataset_v1`
- `official_format_fake_dataset/stable_official_like_multitest_v1`
- `official_format_fake_dataset/directed_cross_v2`

这些是为了模拟官方格式和日志链路重新构造的数据。它们对检测泛化能力有价值，但仍不能等同官方 hidden set。

### Alpha/Beta 提交包相关

- `alpha_test_submission`：Alpha 提交包，不再修改。
- `beta_test_submission`：后续优化提交包，应在此基础上继续迭代。

## 3. 已完成的主要技术路线

### 3.1 Deterministic baseline

特点：

- 不依赖 torch / sklearn 深度模型 / LLM endpoint。
- 不读 trace。
- 不读 gold/meta。
- 可作为正式 fallback。

作用：

- 保底可运行。
- 大规模 case 或 LLM/API 出问题时用于 runtime fallback。

### 3.2 LLM embedding baseline

已实现 OpenAI-compatible embedding endpoint 支持。

历史较优配置：

```text
--llm-mode embedding
--llm-fusion concat
--llm-doc-style features
--llm-weight 4.0
--cluster-factor 0.875
```

结论：

- embedding 对 sim/regr 表层语义有帮助。
- 单纯 concat embedding 不足以解决官方 set1/set2 与 fake 数据分布差异。

### 3.3 Pairwise learning 与 soft voting ensemble

已实现：

- logistic
- GBDT
- MLP
- soft voting ensemble

历史 5-seed half-split 结果显示，pairwise learning 在 stage2/stage3 等中大规模 fake benchmark 上显著强于 deterministic baseline。

局限：

- 传统 21 维 summary features 信息量不足。
- 在官方 benchmark 上容易出现分布不匹配。

### 3.4 Rich Pairwise MLP

新增 richer pair feature：

- deterministic vector relation
- LLM embedding relation
- structured pair features

后来发现 `rich_no_det` 更好，说明将 deterministic SVD vector 整段拼入 MLP 可能会干扰泛化。

### 3.5 Dual-input no-trace calibrated blend

当前历史最强 no-trace experimental best：

```text
llm_dual_struct_det_summary_dim64
+ residual focal MLP
+ pairwise soft voting ensemble
+ alpha=0.88
+ rich_temp=1.15
+ ensemble_temp=1.00
```

特点：

- 同时使用 features embedding 和 summary embedding。
- 不使用 trace。
- 不使用 completion。
- 通过 calibrated blend 融合 rich MLP 与传统 ensemble。

该路线在旧 fake datasets 上表现最好，是目前仍值得保留的 experimental 主线之一。

### 3.6 Trace 路线

已探索：

- tail/global trace structural features
- trace policy veto/boost
- trace refiner
- trace transformer embedding
- anchor-guided trace window

结论：

- trace 中确实存在可分信号。
- 直接全量 trace 或全局 trace embedding 不稳定。
- 更合理的方向是 anchor-window trace，只对困难 case 或不确定 pair 使用。
- 目前 trace 不建议默认启用。

### 3.7 Completion LLM 路线

已配置 NVIDIA API/OpenAI-compatible completion endpoint。

设计原则：

- 不把整份日志发给 completion。
- 只抽取 top failure events、LOG-EXTRACT、局部上下文和结构字段。
- 每个困难 case 最多调用一次。
- 输出固定 JSON evidence tags，不直接输出 bucket。

目前 completion 仍处于 experimental，尚未形成稳定主线。

### 3.8 多粒度输入、Selective Expert 与 Gated MLP

最新统一架构方向：

```text
多粒度 sim/regr 编码
-> base pair model
-> 困难样本筛选
-> selective trace/completion expert
-> learned gate
-> graph / fixed-k clustering
```

当前已实现或部分实现：

- `multigranular_features.py`
- `completion_case_features.py`
- `selective_expert.py`
- `signed_graph_clustering.py`
- `run_unified_multidataset_experiments.py`
- `run_multigranular_selective_experiments.py`
- `run_selective_completion_experiments.py`

最新重点是 Gated MLP：

- 使用 dual embedding。
- 使用多粒度结构特征。
- 使用统一模型，不按数据来源切模型。
- 用 LODO 验证泛化。

## 4. Alpha Test 结果与提交包经验

官方 Alpha 反馈中，我们的潜在模型分数约为：

```text
POTENTIAL score ≈ 0.7220
```

但最初 REAL score 因提交包结构问题为 0：

- 官方 strict runner 期望 `regr_fail_bucketing` 位于提交包顶层。
- 我们最早打包时多包了一层目录。

该问题已经定位并修复。

官方公布 2026-07-02 Alpha Top 5 REAL score：

| Rank | REAL score |
|---:|---:|
| 1 | 0.7794 |
| 2 | 0.7704 |
| 3 | 0.7684 |
| 4 | 0.7676 |
| 5 | 0.7339 |

我们的 0.72 左右距离 Top 5 第五约 1.4 个点，距离第一约 6 个点。说明方向并非完全错误，但要进入前列，需要提升 hidden/worst-case 稳定性。

## 5. Beta 提交包与运行时间优化

已创建：

```text
beta_test_submission/
```

并明确不再修改：

```text
alpha_test_submission/
```

Beta 包当前采用 runtime-safe router：

```text
small n     -> full model
medium n    -> deterministic agglomerative fallback
large n     -> deterministic kmeans fallback
```

主要目标：

- 防止超时。
- 防止 API/LLM 不可用导致崩溃。
- 保证输出格式合法。

已验证：

- 官方小 benchmark 可走 full model。
- 大规模 stress 可走 fast deterministic route。
- 输出格式保持 `Case,bucket`。

## 6. 当前正在进行的实验

当前正在跑 8-dataset LODO：

```text
datasets:
  first_batch_dataset
  stage2_dataset_working
  stage3_dataset_32bugs_640cases
  official_vcs_stage1_dataset_v1
  stable_official_like_multitest_v1
  directed_cross_v2
  benchmark_set_1
  benchmark_set_2

configs:
  gated_mlp balanced
  gated_mlp hard_pos_connect

seeds:
  0..4
```

运行方式：

- GPU 4：seeds 0/1/2
- GPU 5：seeds 3/4
- embedding server：CPU Nomic endpoint
- 已确认 `features` 与 `summary` 两路 embedding 都为 768 维，无 fallback。

当前 partial 结果显示：

- `hard_pos_connect` 不全面优于 `balanced`。
- 在 stage3 上略有提升，主要来自 TNR 提高。
- 在 first/stage2/directed_cross_v2 上暂时不稳定。
- benchmark_set_1 上 seed0 已出现 BA 1.0。
- benchmark_set_2 上 seed0/seed3 的 Gated MLP 仍偏弱，说明官方 set2 是当前泛化瓶颈之一。

因此，目前还不能把 hard-positive/connectivity 方案定为新主线。需要等待完整 seeds 0..4 结果，再决定是否扩展 seeds 0..9。

## 7. 当前关键问题

### 7.1 官方数据与 fake 数据分布差异明显

旧 fake datasets 上表现好的模型，在官方 set1/set2 或 official-format fake 上可能下降明显。

主要原因：

- fake 数据的 sim/regr 信号更规整。
- 官方日志中表层 failure pattern 更容易重叠。
- 同一 bug 可能出现多表象。
- 不同 bug 可能共享 testname、mismatch type、PC mismatch、REG mismatch 等表层模式。

### 7.2 当前模型存在 fragmentation 与 over-merge 双重问题

典型错误：

- fragmentation：同一 bug 被拆成多个 bucket，TPR 低。
- over-merge：相似但不同 bug 被合并，TNR 低。

单纯调 temperature、cluster factor 或 k-policy 难以同时解决这两个问题。

### 7.3 Completion 和 trace 需要 selective 使用

trace 和 completion 都有潜在价值，但不能作为默认全量主线：

- trace 全局使用不稳定。
- completion 成本和 latency 不确定。
- full log completion 容易过慢且信息噪声大。

更合适的方式是：

- 先由 base model 筛出困难 case。
- 只对困难 case 调 trace/completion expert。
- expert 只输出 evidence tags，不直接分桶。
- learned gate 决定 expert 影响概率。

### 7.4 运行时间与打包稳定性同等重要

Alpha 说明：REAL score 严格依赖可执行性、超时和格式。

因此后续不能只追求 local score，还必须保证：

- 无 API 时 fallback。
- 大数据不超时。
- package 顶层结构正确。
- 环境依赖不踩 GLIBC / symlink / hidden import 问题。

## 8. 当前推荐路线

### P0：保住 Beta 提交稳定性

继续维护：

- `beta_test_submission`
- runtime router
- fast fallback
- no-LLM/no-trace safe path

这是避免 REAL score 掉零的底线。

### P0：完成 Gated MLP LODO seeds 0..4

当前正在跑。

完成后需要汇总：

- macro mean BA
- official set1/set2 mean BA
- worst dataset BA
- worst seed BA
- TPR/TNR balance

如果 `hard_pos_connect` 不稳定，则不要继续盲目扩大，而应改采样和 loss。

### P1：训练目标从 pair BCE 升级到 bug graph connectivity

建议继续探索：

- hard positive mining
- hard negative mining
- cross-view positive
- positive connectivity loss
- prototype / anchor loss

目标不是让所有同 bug pair 都高，而是保证同 bug 图足够连通，减少系统性 fragmentation。

### P1：Selective Completion Expert

建议只对 difficulty top 15% case 使用 completion。

输入限制为：

- top failure events
- local sim/regr context
- LOG-EXTRACT
- PC/register/opcode/CSR 结构字段

输出固定 JSON evidence tags。

### P1：Anchor-window Trace Expert

不做 full trace。

优先做：

- failure time anchor
- PC anchor
- register mismatch anchor
- ±64 / ±128 retired instruction window
- opcode ngram / PC region / CSR/system/branch/load/store features

只对困难 case/pair 启用。

### P2：Conservative signed graph clustering

固定 k 聚类不一定最优，但 adaptive-k 不能太激进。

建议：

- 在 `[0.8k, 1.2k]` 中选择。
- 没有明显 merge-gain 断崖就回退 k。
- 并列时选择更多桶，避免过度合并。
- strong conflict edge 作为 cannot-link penalty。

## 9. 近期工作清单

1. 等待当前 8-dataset LODO seeds 0..4 完成。
2. 汇总 balanced vs hard_pos_connect。
3. 若 Gated MLP 仍不稳，优先优化 hard-pair sampling 和 connectivity/prototype loss。
4. 加入 completion JSON tags 作为 optional expert feature。
5. 加入 anchor-window trace expert。
6. 做 old blend 与 Gated MLP 的 probability ensemble。
7. 完成 beta package runtime smoke。
8. 将最终候选方案在官方 set1/set2、official-format fake、旧 fake 上统一验证。

## 10. 阶段性结论

目前项目已经从 deterministic baseline 发展到 rich pairwise / dual embedding / calibrated blend / Gated MLP / selective expert 的多路线实验阶段。

最重要的阶段性判断：

1. 旧 fake dataset 上的高分不能直接代表官方 hidden 泛化。
2. 当前 Alpha potential score 约 0.72，距离 Top 5 不远，但要提升需要解决 hidden/worst-case 稳定性。
3. 不应按数据来源切换模型，应该训练统一模型。
4. 只加深网络不是主要方向，关键是更细粒度证据、hard-pair training、graph connectivity、selective expert 和稳健 clustering。
5. Beta 阶段必须同时优化分数和运行时间，防止 REAL score 因超时/打包/API 问题损失。

一句话总结：

```text
当前最有希望的方向是：
统一多粒度 pair model
+ hard-pair/connectivity training
+ selective trace/completion expert
+ conservative graph clustering
+ runtime-safe beta submission router
```

这条路线既保留已有 no-trace calibrated blend 的稳定性，也为官方 hidden set 中表层日志重叠、同 bug 多表象、相似 bug 混淆等问题提供了继续提升的空间。
