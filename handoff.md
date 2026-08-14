# ICCAD 竞赛交接文档（写给零上下文的新会话）

> 最后更新：2026-08-13

## 1. 我们在做什么

**ICCAD 2026 Problem B：EDA Regression Failure Bucketing（回归失败聚类）**

- 任务：给定一组 Ibex RISC-V CPU 回归仿真的失败日志（`sim.log` / `regr.log` / `trace.log`），把由**同一个根因 bug** 导致的 case 聚到同一个 bucket。
- 提交接口（bash 入口 `regr_fail_bucketing`，或 python `regr_fail_bucketing.py`）：

```bash
regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

- `input.csv` 列名：`trace_log,sim_log,regr_log`（日志路径）；输出必须是 `Case,bucket` 两列 CSV。
- **k 是软提示**（真实 bug 数），不是硬约束；打分是 **Balanced Accuracy = (TPR+TNR)/2**，pairwise 位置对齐（gold[i] vs pred[i] 逐行 zip）。
- 官方公开 dev benchmark：`test_case/problem/benchmark_set_1`（7 例）、`benchmark_set_2`（25 例），带 `golden.csv` 标签。
- **最终评测用的是我们拿不到的 hidden set**（据 `ICCAD阶段性成果报告.md`：「仍不能等同官方 hidden set」）。赛题 PDF：`B_20260601.pdf` / `B_QA_20260612.pdf`（当前目录下 *.pdf 可能已被清理，历史文档在 `MATERIALS_INDEX.md` 有索引）。

**用户优先级（必须遵守）：**
1. **安全第一**：保证 final 提交任何情况下都能拿到有效分数（valid output > 高分）。
2. BA 最大化其次。
3. GPU 限制（原文）：「最多可以使用两张卡 但是注意要保证使用时是空闲的 不能占用别人的，用完记得释放」。
4. LLM endpoint 一律通过 `LLM_MODEL_CONFIG` 环境变量（YAML）读取，**不要硬编码 localhost**。

## 1.5 核心战略原则（2026-08-13 用户强调，建模前必读）

这些是用户在调研过程中反复强调的**方法论红线**，直接影响模型怎么设计，务必先读再动手：

1. **分布差距警告**：自造 fake 数据与官方数据有分布差距，**官方公布的 dev 集与最终隐藏评测集也很可能有分布差距**。因此模型必须**分布鲁棒**，不能只在 fake/official 上刷分。

2. **自造数据的作用**：主要是**对齐日志格式 / 预训练**，不是用来刷最终 supervised 分。不要把 fake 上的高分当成目标。

3. **bug 分类依据官方没明说**：官方没明确说明 bug 划分标准，甚至出现过「三种日志（sim/regr/trace）完全相同但属于不同 bug」的数据（后来被剔除）。**这说明 bug 身份可能根本不可从日志观察**。

4. **不要纠结「如何精确划分 bug」**：因为相同日志可能是不同 bug，追求「设计一个签名把日志精确映射到 bug」是**死路**（数据上做不到）。

5. **从 ML 聚类角度出发**：把相似 case 聚到一起，接受「有些 bug 天然不可分」（BA 有天花板）。

6. **EDA 底层原理用在「特征设计」上，不用在「划分规则」上**：
   - 特征：功能单元、指令 opcode、寄存器、分歧类型、PC/cycle、trace 上下文、LLM 语义嵌入——**尽量丰富地喂给聚类模型，让模型自己学权重**
   - 不要手设计「黄金指纹签名」（第一个 mismatch 的 opcode 会被 `lui` 这类初始化指令污染，见 §7 坑 #13）

7. **LLM 要充分利用但别嵌 boilerplate**：嵌**语义片段**（regr.log 的 mismatch 行 `pc[...] ori x16,x3,903`、trace 分歧点附近），而不是整段日志（UVM header/VCS banner 是分布依赖的负资产）。

## 1.6 竞赛关键文件 + hidden 集规模（2026-08-14 用户强调）

**最重要的竞赛文件都在 `/home/lishixian/iccad/information_files/`**（不是项目根目录）：
- `B_20260601.pdf` — 赛题规格（接口、打分、benchmark 规模、运行时）
- `QA/B_20260601.pdf`、`B_QA_20260612.pdf`、`B_QA_20260727.pdf` — 官方答疑（更新运行时、benchmark 等）
- `submission_files/Alpha Test Submission Guideline_ABC.pdf`、`beta_test_submission_ABC.pdf` — 提交格式
- `result/alpha.docx`、`result/beta.docx` — 官方反馈

**hidden 集规模（B_20260601.pdf §3.2，共 10 个 benchmark，BA 平均为最终分）**：

| # | N(case) | k(bug) | 运行时 | 大小 |
|---|---|---|---|---|
| 1 | 2 | 2 | 30s | 1M |
| 2 | 30 | 4 | 30s | 10M |
| 3 | 100 | 8 | 100s | 10M |
| 4 | 300 | 16 | 100s | 10M |
| 5 | 1000 | 32 | 100s | 10M |
| 6 | **3000** | **64** | **100s** | 10M |
| 7 | 100 | 8 | 300s | 100M |
| 8 | 300 | 16 | 300s | 100M |
| 9 | 1000 | 32 | 300s | 100M |
| 10 | **3000** | **64** | **300s** | 100M |

**关键结论**：hidden 集有 **N=3000、k=64** 的 benchmark，与我们造的 **benchmark6（2944 例、64 bug）规模完全一致**。所以 benchmark6 不是「大 fake 误导」，**它就是 hidden 集规模的真实预演**，64-bug 分簇（TPR=0.134）是真问题，必须攻克。运行时（N=3000 给 100s/300s）下，benchmark6 的 70.6s 推理绰绰有余。Alpha/Beta 测试的运行时是 final 的 3×/2×（§3.2 PS）。

## 2. 环境与关键路径

- **项目目录**：`/home/lishixian/iccad`（所有代码、数据、提交包都在这）
- **正确 python**：`/home/lishixian/miniforge3/envs/collab-overcooked/bin/python`（有 torch/joblib/sklearn）。**系统 `python3` 没有 torch，会直接 ModuleNotFoundError**。跑任何脚本都显式用这个解释器。
- **LLM embedding**：`nomic-embed-text-v1.5`（768 维）。开发环境本地服务器 `127.0.0.1:8001` 可用；正式环境靠 `LLM_MODEL_CONFIG`（NVIDIA NIM 兜底）。缓存 `/tmp/regr_fail_llm_cache`（按 `cache_key_for_llm_doc` 键控）。
- **Trace 缓存**：`/tmp/theta_trilog_trace_cache`。
- **推理只跑 CPU**（`final_inference.py` 里 `torch.load(map_location="cpu")`、`device="cpu"`）。
- 机器：128 核、503GB RAM、多用户共享（load 常年 30+）。**不是 git 仓库**（改代码前注意备份或自己 diff）。

**当前产物（都在）：**
- 训练输出：`/tmp/theta_final_real_5seed/`（results.csv 45 行 + manifest.json + models/ 45 个 .pt + 45 个 .pkl + preds/）
- 打包好的最终提交：`/home/lishixian/iccad/final_submission/`（746MB，结构见 §4）
- 验证输出：`/tmp/theta_final_validate2/`（定向验证结果）、`/tmp/theta_final_validate/`（部分）

## 3. 数据集（11 个，⚠️ 有嵌套泄漏）

| 数据集 | 例数 | 类型 | 说明 |
|---|---|---|---|
| first_batch_dataset | 80 | 旧 fake | ❌ **已删除**（⊂ stage2 ⊂ stage3，嵌套副本）|
| stage2_dataset_working | 240 | 旧 fake | ❌ **已删除**（⊂ stage3，嵌套副本）|
| stage3_dataset_32bugs_640cases | 640 | 旧 fake | ✅ 保留（含前两个的全部 case）|
| official_vcs_stage1_dataset_v1 | 40 | official 格式 fake | ✅ 保留（独立，与 official 模板近似）|
| directed_cross_v2 | 37 | official 格式 fake | ❌ **已删除**（⊂ directed_cross_v4）|
| directed_cross_v4 | 65 | official 格式 fake | ✅ 保留（含 directed_cross_v2 全部）|
| stable_official_like_multitest_v1 | 24 | official 格式 fake | ✅ 保留（独立）|
| benchmark5_final | **96** | official 格式 fake | ✅ 保留但**剪枝到 96 独有例**（原 933，删 837 与 benchmark6 重复的）|
| benchmark6_final | 2944 | official 格式 fake | ✅ 保留 |
| benchmark_set_1 | 7 | **官方 dev**（有 golden.csv）| ✅ 保留，微调目标之一 |
| benchmark_set_2 | 25 | **官方 dev**（有 golden.csv）| ✅ 保留，微调目标之一 |

> **当前训练集 = 6 个 fake + 2 个 official，共 3809 个唯一 case（内容级零重叠，已实测）。**

**数据泄漏现状（2026-08-13 已实锤，content hash = `regr.log` + 解压后的 `sim.log.gz`）：**

自造 fake 数据**不是独立集，是「扩展」关系**——小集的 case 与大集的部分 case 逐字节相同：

| 关系 | 验证 |
|---|---|
| `first_batch(80) ⊂ stage2(240) ⊂ stage3(640)` | 80/80、240/240 完全包含 |
| `directed_cross_v2(37) ⊂ directed_cross_v4(65)` | 37/37 完全包含 |
| `benchmark5(933) ∩ benchmark6(2944) = 837` | 90% 重叠，但 benchmark5 有 96 例独有（**非纯子集**）|
| `official_vcs(40)`、`stable(24)` | 与任何集无逐字节重复（独立）|
| `benchmark_set_1(7)`、`benchmark_set_2(25)` | 与 fake 无逐字节重复（仅模板近似 = 软泄漏）|

**去重总量：9 个 fake 共 5003 case-slot → 唯一 3809 个 case，1194 slot（24%）是重复。**

三个后果：
1. 对 fake 做「留一数据集」LODO 是假的——留出 first_batch 时 stage3 里还有它一模一样的 case，泄漏。
2. 训练时同一 case 在多个集里被重复计数（first_batch 的 80 例在 3 个旧 fake 里各出现一次），权重被抬高。
3. 「9 数据集」实际 ≈ 5 个唯一家族 + 96 个 benchmark5 独有例。

详情见 `GENERALIZATION_AUDIT_20260713.md`。**结论：目前没有任何一个分数能代表 hidden set 真实表现。** fake 集 1.0 是背答案，official 集 0.46/0.77 是 train-on-dev 过拟合数。

## 3.5 去重已执行（2026-08-13）

按上面的泄漏关系，已对数据集做物理去重（**用 `mv` 移到 quarantine，未 `rm`，可恢复**）：

- **整目录移走 3 个嵌套子集** → `dataset/_removed_20260813/`：
  - `old_fake_dataset/first_batch_dataset`（⊂ stage2/stage3）
  - `old_fake_dataset/stage2_dataset_working`（⊂ stage3）
  - `official_format_fake_dataset/directed_cross_v2`（⊂ directed_cross_v4）
- **benchmark5 剪枝**：837 个与 benchmark6 重复的 case 目录移到 `dataset/_removed_20260813/benchmark5_duplicates/`，并重生成 `benchmark5_final/{input,golden,meta}.csv`（各 96 行，Case 序列与 case 目录一一对应）。
- **脚本同步**：`run_final_submission_train.py` 的 `FINAL_DATASETS`、`run_final_submission_validate.py` 的 `DATASETS` 都已删掉这 3 个集，benchmark5 保留（现 96 例）。

**结果：剩余 6 fake + 2 official 内容级零重叠，训练集 = 3809 唯一 case（640+40+65+24+96+2944）。** 之前的 5-seed 45 模型（`/tmp/theta_final_real_5seed` 与 `final_submission/`）仍是在**未去重的 9 集**上训的，**尚未用去重后的数据重训**——重训留作下一步。

## 3.6 真实桶数严查（2026-08-13）

所有数据集的真实桶数（k = `len(set(gold))`）与每桶 case 分布，**数据改动后必须重新数，不能沿用旧值**：

| 数据集 | cases | k(桶数) | 每桶 min/mean/max | 稀疏桶(≤2例) | 结论 |
|---|---|---|---|---|---|
| stage3 | 640 | 32 | 20/20/20 | 0 | 均衡 ✅ |
| official_vcs | 40 | 3 | 8/13.3/16 | 0 | 均衡 ✅ |
| directed_cross_v4 | 65 | 10 | 4/6.5/8 | 0 | 均衡 ✅ |
| stable | 24 | 4 | 6/6/6 | 0 | 均衡 ✅ |
| benchmark5(剪枝后) | 96 | 32 | 1/3/6 | **12 (37.5%)** | ⚠️ **极稀疏，已排除** |
| benchmark6 | 2944 | 64 | 46/46/46 | 0 | 均衡 ✅ |
| benchmark_set_1 | 7 | 2 | 3/3.5/4 | 0 | 小但均衡（official）✅ |
| benchmark_set_2 | 25 | 4 | 3/6.2/15 | 0 | 均衡（official）✅ |

**决策：benchmark5 剪枝后 32 桶只有 96 例、37.5% 的桶 ≤2 例，是极稀疏聚类任务，先排除出训练。** 排除后训练集 = **5 fake + 2 official = 3713 唯一 case**（640+40+65+24+2944）。benchmark5 的数据没删，仍在 `dataset/` 里，只是不进训练。

## 4. 完成了什么

> **⚠️ 以下「Theta TriLog pair 模型」已退役（2026-08-14 被 siamese 取代），仅作历史参考。** 它的资产（trace 特征管线、SupCon 概念、软 k 聚类）被 siamese 继承，但 pair 架构（O(N²)）、LODO 折结构、`run_final_submission_train.py`/`final_inference.py`/`package_final_submission.py` 都已不再使用。

### 模型：Theta v4 TriLog（双塔）【已退役】
- `TriLogPairNet(base_dim, trace_dim, dropout, fusion)`，`fusion=concat`；`forward_features` 把 `value[:, :base_dim]`（base 视图）和 `value[:, base_dim:]`（trace 残差）分开。
- 特征：features/summary/event/object/context 五视图（各 64 维，LLM 768 维经 reducer 降到 64）+ trace residual。
- 训练两阶段：**9 fake 集 LODO pretrain → 2 official 集冻结 backbone 只训末层**。5 seeds × 9 folds = 45 个模型。
- 聚类：correlation clustering，`cannot_link_weight=100`，入口 `cluster_with_fallback`。
- **软 k 后处理**：`_enforce_requested_k` 只在簇数严重偏离 k（>2× 或 <½）时 merge/split 到 k。教训：k 是软提示，不要硬卡成精确 k。

### Siamese + 原型模型（2026-08-14 新主线，替代 pair 模型）

按 §1.5 战略原则（ML 聚类 + EDA 特征）重做的 O(N) 模型，**推荐以此为当前主线**：

- 架构：`run_siamese_train.py` + `theta_siamese_model.py`——per-case 编码器（Siamese）→ SupCon + 原型损失 → k-means。**O(N) 推理**（pair 模型是 O(N²)）。
- 特征（per-case）：LLM 降维 64 + 失败签名（家族 11 + 分歧类型 3）+ **测试名语义类别 flag（csr/interrupt/debug/mmu/... 14 类）** + trace residual 96。
- **关键突破（测试名类别 flag）**：benchmark_set_1 的 2 个 bug 靠「测试名功能域」区分——bug_7023 = riscv_{interrupt_csr,multiple_interrupt,single_interrupt,csr}_test（CSR/中断域），bug_234 = riscv_debug_basic_test + mismatch（debug 域）。精确测试名 one-hot 会把 interrupt_csr/csr 打散（有害），**语义类别 flag（csr/interrupt/debug 关键词）才对**。加了这个特征后 set1 从 0.54 → 0.83，official mean 从 0.696 → **0.868**（train-on-dev，5-seed）。
- 关键修复（3 个 bug，都在 theta_siamese_model.py / run_siamese_train.py）：
  1. SupCon 的 `log(0)` 梯度 NaN（对无正样本的行 `0×inf=NaN` 导致权重全 NaN 坍缩）→ numerator 加 `1e-12`。
  2. `_find_regr` 的 case id 规范化（`read_cases` 返回 `'1'`，目录是 `case_1`）→ 数字 id 补 `case_` 前缀。
  3. bug 标签跨数据集碰撞（`bug_037` 在不同集是不同 bug）→ 加数据集名前缀 `{ds}::{bug}`。
- 验证集早停：`train_siamese_model` 有 stratified train/val split + val-loss 早停，自动选最优 epoch（40ep 会过拟合，别手调）。
- **分数（干净，只训 4 fake 测 official）**：official mean ≈ 0.53（set1≈0.55、set2≈0.51）；fake LODO 小集 0.60~0.70，benchmark6(64bug)=0.52（难，TPR=0.056 严重欠聚类）。train-on-dev（官方进训练）会飙到 0.868 但那是过拟合（见坑 #16），不可取。

### 当前提交包（final_submission/，2026-08-14 打包完成）

- **入口** `final_submission/regr_fail_bucketing`（PyInstaller onedir，mode 755，顶层）
- **模型**：siamese 5-seed 集成（NumPy 前向，无 torch）
- **接口** `--input --output --k` → `Case,bucket`
- **LLM** 通过 `LLM_MODEL_CONFIG`，失败回落确定性基线
- **GLIBC** 已修到 2.28 上限（替换 libstdc++ 2.38→2.17、libgcc 2.35→2.14）
- **打包步骤**：`siamese_predict.py` + `failure_signature.py`（torch-free）→ 转 NumPy encoder → PyInstaller

**提交包实测（train-on-dev，过拟合）：** benchmark_set_1=1.0、benchmark_set_2=0.91、official_vcs=0.80、directed_cross_v4=0.78、stable=0.78、benchmark6=0.56（64 bug，TPR=0.134）。运行时 2.9s~70.6s，均在官方 100s/300s 限制内。

### 模型迭代结论（2026-08-14，损失/特征实验均边际）

针对 benchmark6 的 TPR 低（碎片化）问题，试了 3 个前沿方法，**都边际或负面**：

| 方法 | benchmark6 TPR | 结论 |
|---|---|---|
| 基线 | 0.134 | — |
| 子中心原型（SwAV 多原型）| 0.144 | 边际 |
| 正样本聚合损失 | 0.139 | 边际 |
| sim.log UVM_FATAL 语义特征 | 0.124 | 负面（给 mismatch bug 加噪声）|
| 域对抗适配（DANN）| ~0.134 | 边际负面（set2 0.877→0.871）|

**根因（数据限制，非模型）**：fake benchmark6 的 **56 个 mismatch bug 在 sim/regr/trace 三种日志里都无可区分信号**（同 bug 不同 PC 显形、第一个 mismatch 是 lui 污染、测试名只有 6 个、agent 名只有 3 个）。对应官方 QA 的「renew benchmarks to **avoid diff-bugs-same-syndrome**」——官方 hidden 集专门消歧过，我们的 fake benchmark6 没消歧，**比官方更不可分**。所以 benchmark6 的 TPR=0.134 大概率是 fake 数据固有上限，不是模型问题。

**域对抗适配（DANN）是错配**：DANN 假设「域是 nuisance（干扰），要移除」，但我们的 fake vs official 域是 **informative 的**（official 的 bug 分布和 fake 不同，域信号本身携带 bug 信息）。强行学域无关特征反而丢掉有用信号。所以 DANN 对本任务不适用。

## 4.5 alpha 基线 + 对标（2026-08-14）

用户要求：新模型要在**本地数据集上干净地跑过 alpha**。alpha 的 0.72 是**官方隐藏测试集**的泛化分（不是本地分）。

**alpha 两套本地分数**（alpha 冻结二进制 `submission_files/alpha/alpha_test_submission/regr_fail_bucketing`，PyInstaller，默认 `llm_mode=none` 确定性，CLI 只有 `--input --output --k`，无 `--llm-mode`）：

| 数据集 | alpha 确定性 | alpha LLM（VALIDATION_RESULTS）|
|---|---|---|
| benchmark_set_1 | 0.528 | 0.722 |
| benchmark_set_2 | 0.931 | 0.921 |
| official_vcs | 0.587 | — |
| directed_cross_v4 | 0.649 | — |
| stable | 0.597 | — |

**⚠️ 关键：alpha 官方集分数（set1=0.722/set2=0.921）是 train-on-dev 过拟合**——它的 adapter（`official_style_tags_logistic`）就在 benchmark_set_1/2 上训练，是背答案，不代表泛化。**不要拿它当对标目标。**

**fake 集干净对比（双方都没过拟合）：**

| 数据集 | alpha 确定性 | 我们（干净 LODO）|
|---|---|---|
| official_vcs | 0.587 | **0.598** ✅ |
| directed_cross_v4 | 0.649 | **0.699** ✅ |
| stable | 0.597 | **0.674** ✅ |
| **fake 平均** | 0.611 | **0.657** ✅ |

**结论：fake 集（诚实泛化）上我们已全面超过 alpha。** 唯一输的是 benchmark_set_2（干净 0.513 vs alpha 0.931），但那是 alpha train-on-dev 背答案。

**剩下真问题**：干净 official 迁移（只训 fake、测 official）只有 0.51~0.56，离 alpha hidden 0.72 还远。这是 fake→official 分布跨越，不是过拟合能解决的。

## 5. 卡在哪儿（当前阻塞点）

1. **benchmark6 的 64-bug 分簇是 fake 数据固有上限**：TPR=0.134 是因为 fake benchmark6 的 56 个 mismatch bug 在日志里不可分（见 §4 模型迭代结论），3 个损失/特征实验都救不了。官方 hidden 集已消歧，大概率比 fake 好分，但无法本地验证。
2. **干净 official 迁移只有 0.53**：只训 fake、测 official 时，set1≈0.55/set2≈0.51。这是 fake→official 分布跨越，是真实泛化瓶颈。alpha 靠「训 official（adapter）」拿到 hidden 0.72，我们是「干净迁移」0.53。
3. **train-on-dev 分（0.868）不可信**：官方集进训练是背答案（见坑 #16），不能当泛化证据。

## 6. 最终状态（2026-08-14 收尾）

**决策：接受当前模型为最终版本（用户选定）。** 5 类方法（子中心/正样本聚合/UVM_FATAL/加数据/域对抗）都边际或负面，模型迭代已达数据与方法瓶颈。

**最终提交**：`final_submission/`（siamese + 原型 + trace + 测试名，B 版 fakes+official 一起训）

- 架构：per-case 编码器 → SupCon + 原型损失 → k-means，O(N)，PyInstaller 自包含，GLIBC 2.28 达标
- 特征：LLM 64 + 失败签名（家族+分歧类型）14 + 测试名类别 14 + trace residual 96 = 188 维
- 本地分数（train-on-dev，过拟合上界）：set1=0.83、set2=0.90、official_vcs=0.80、directed_cross_v4=0.78、stable=0.78、benchmark6=0.56
- fake LODO（诚实泛化）：小集 0.60~0.70、benchmark6=0.52（64 bug 数据上限）
- hidden 集估计：对标 alpha 的 ~0.72（同样训了 official）

**遗留（可选，不再主动投入）：** 干净 official 迁移 0.53（fake→official 分布跨越）；benchmark6 64-bug 不可分（fake 数据未消歧）。

## 7. 采过的坑（不要再踩）

1. **LLM 混合宽度崩溃**：`np.vstack` 把 768 维（成功抓到）和 0 维（LLM 超时兜底产生的零向量）拼一起直接 ValueError（index 0 size 768, index 1979 size 0）。**修复**：`pairwise_llm_features.py` 里所有 reducer/apply 函数改用 `_stack_llm_vectors`（按公共最大宽度补零）替代 `np.vstack`；兜底零向量必须用 `llm_expected_dim`（768）造满维，不能造 0 维。**不要再用裸 `np.vstack` 拼 LLM 向量。**
2. **train/hold scaler 宽度不匹配**：`X has 903 features, but StandardScaler is expecting 3719`。根因：`build_multiview_pair_feature_matrix` 的 `_safe_vstack` 取所有特征的最大宽度；train 侧 llm_vec 已被 reducer 降成 64 维，hold 侧还是 768 维原向量，宽度被撑爆。**修复**：在 `run_final_submission_train.py` 里 fit 完 reducer 后，对**全部** features（不只 train 子集）先 `apply_llm_reducer` / `apply_llm_summary_reducer` 再建矩阵。**记住：reduce-all-before-build。**
3. **大数据集 CPU 推理慢到离谱**：benchmark6（2944 例）全对全 ~433 万对 × 45 模型，CPU 上预计 ~20 小时（stage3 641 例实测 54 分钟）。瓶颈是**逐 case LLM 抓取（每 case 5 视图 5 次 fetch）+ O(N²) 对矩阵**，不是模型加载。**不要在 CPU 上对 N>1000 的集跑全对全推理**；竞赛 hidden set 规模很小（几十例），小集推理很快（7 例全链路 13s）。
4. **模型加载其实很快**：45 个模型加载 ~10s，别想当然以为是加载慢。小输入的总耗时主要由 LLM fetch 决定。
5. **python 环境**：系统 `python3` 无 torch。永远用 `/home/lishixian/miniforge3/envs/collab-overcooked/bin/python`，或设 `PY=$(...)` 变量后统一 `$PY`。
6. **GPU 纪律**：最多 2 张、先确认空闲（`nvidia-smi --query-compute-apps`）、用完释放。别碰别人占用的卡。GPU 0 上常年有别的用户（wanghaoming 等）的 30GB 进程。
7. **LLM endpoint 不能硬编码**：一律 `LLM_MODEL_CONFIG`。`final_inference.py` 里 `llm_timeout_sec=60`（训练时 transient 超时曾把 benchmark6 全量打成 0 维向量，见坑 #1）。
8. **嵌套副本陷阱**：first_batch/stage2/stage3 是同一批数据的递增超集，任何 LODO/留一策略对它们无效。做「干净」实验必须走内容级 guard（`evaluation_leakage_guard.py`）。
9. **验证脚本顺序陷阱**：`run_final_submission_validate.py` 顺序跑、结果 csv 只在最后统一写——中途 kill 会丢掉所有已算分数（输出 csv 还在，可用 `pairwise_scores(read_gold(...), pred)` 手动补分）。
10. **官方集分数是乐观上界**：official 集被 9 个 fold 全部见过（微调目标），任何「official 验证分数」都是 train-on-dev。向用户报告时**必须注明这一点**，不要拿它当泛化证据。
11. **算 case 内容重叠必须解压 `sim.log.gz`**：official 格式 fake 的 sim 日志是 `sim.log.gz`（gzip），而旧 fake 是明文 `sim.log`。若只哈希 `regr.log`（官方格式 fake 里只有 118~735 字节、信息量极低），不同 case 会**撞哈希**，误报「官方集之间重叠 4 例」「benchmark5/6 是纯子集」这类假结论。正确做法：`regr.log` + 解压后的 `sim.log.gz` 一起哈希。case 目录有两类结构（旧 fake 在 `cases/` 子目录、official 格式 fake 在顶层 `case_*`），要用 `rglob` 递归找。
12. **传 k 前必须先 `len(set(gold))` 确认真实 bug 数，绝不硬编码**：曾把 benchmark_set_1 想当然当 k=3，真实是 k=2（bug_234、bug_7023），导致所有 benchmark_set_1 的结论都错了。代码里 validate 用 `k = len(set(gold))`、训练用 `len(set(labels))` 是对的，但手工测试时硬编码 `--k 3` 是错的。**每个数据集的 k 都要从 gold 数出来再用。**
13. **「第一个 mismatch 的 opcode」不是 bug 指纹**：benchmark6 的 case 从 cycle 0 就分歧（`Mismatch[1]` 是 `lui`/`c.li` 初始化指令），导致第一个 mismatch 的 opcode 被 `lui`（74%）污染，对 64 个 bug 没区分度。硬编码 opcode 独热签名反而**有害**（把 74% 的 case 都标成 lui，成了常值噪声）。**别手设计「黄金指纹」，把特征丰富地喂给 ML 聚类（siamese+原型）让模型自己学。**
14. **测试名用「语义类别」别用「精确 one-hot」**：benchmark_set_1 的 bug_7023 有 4 个不同测试名（interrupt_csr/multiple_interrupt/single_interrupt/csr），精确 one-hot 会把这 4 个 case 打散（有害）；用**关键词类别 flag**（csr/interrupt/debug/mmu...）才对（它们都是 CSR/中断域）。这是本轮最大突破：set1 从 0.54 → 0.83。
15. **集成别直接平均 embedding**：不同 seed 的 siamese embedding 空间没对齐（各学各的方向），直接平均会模糊小数据集的细微区别（benchmark_set_1 集成后 set1 从 0.54 掉到 0.43）。小数据集别用 embedding 平均集成；要用 co-association（各 seed 同簇一致度）或直接对 prediction 投票。
16. **train-on-dev（官方集进训练）是过拟合，不可取**：把 benchmark_set_1/2 加进训练，official mean 从 0.53 飙到 0.868——但这是背答案，不是泛化。用户明确「train-on-dev 不可取」。真实泛化要看**干净迁移**（只训 fake、测 official）和 **fake LODO**。
17. **「per-seed 平均」≠「集成（平均 embedding 再聚类）」**：per-seed 平均是 5 个 seed 各自聚类取平均 BA；集成是 5 个 encoder 先平均 embedding、再聚类一次。两者分数不同（集成通常更稳、更高，如 set1 per-seed 0.833 → 集成 1.0）。**最终提交 `siamese_predict.py` 用的是集成**，别拿 per-seed 平均当提交分数。
18. **GLIBC 打包风险（关键）**：官方机要求 ELF 符号上限 GLIBC 2.28（RHEL/Alma/CentOS 8），但本开发机是 glibc 2.39，PyInstaller 打包出的 `_internal/libgcc_s.so.1`（GLIBC_2.35）和 `_internal/libstdc++.so.6`（GLIBC_2.38）**超标**。**已修复（2026-08-14）**：用 conda 的 `libgcc-15.2.0`（GLIBC_2.14）和 `libstdcxx-15.2.0`（GLIBC_2.17）替换，替换后全包最高 GLIBC = 2.28（达标）。注意：libstdc++ ABI 是稳定的（GCC 3.4 起），旧 glibc 版本可安全替换新版本。

## 8. 关键命令速查

```bash
cd /home/lishixian/iccad
PY=/home/lishixian/miniforge3/envs/collab-overcooked/bin/python

# siamese 训练（B 版：4 fake + 2 official 一起训，train-on-dev）
$PY run_siamese_train.py --train-datasets \
  dataset/fake_dataset/official_format_fake_dataset/official_vcs_stage1_dataset_v1 \
  dataset/fake_dataset/official_format_fake_dataset/directed_cross_v4 \
  dataset/fake_dataset/official_format_fake_dataset/stable_official_like_multitest_v1 \
  dataset/fake_dataset/official_format_fake_dataset/benchmark6_final \
  dataset/real_dataset/benchmark_set_1 dataset/real_dataset/benchmark_set_2 \
  --eval-datasets dataset/real_dataset/benchmark_set_1 dataset/real_dataset/benchmark_set_2 \
  --output-dir /tmp/siamese_seed0 --seed 0 --device cpu --epochs 40 --use-trace

# 5-seed 集成评估（平均 embedding 再聚类）
$PY run_siamese_ensemble_eval.py --seed-dirs /tmp/siamese_seed0 ... /tmp/siamese_seed4 \
  --eval-datasets dataset/real_dataset/benchmark_set_1 dataset/real_dataset/benchmark_set_2 \
  --output-dir /tmp/ens --use-trace

# 打包（PyInstaller onedir，排除 torch）
$PY -m PyInstaller --name regr_fail_bucketing --onedir \
  --add-data "/home/lishixian/iccad/final_submission/models:models" \
  --exclude-module torch --exclude-module torchvision --exclude-module torchaudio \
  --paths /home/lishixian/iccad siamese_predict.py

# 提交包实测
final_submission/regr_fail_bucketing --input <csv> --output <csv> --k <k>
```

## 9. 关键文件索引

**当前主线（siamese）：**
- `run_siamese_train.py` + `theta_siamese_model.py`：siamese + 原型模型（SupCon/原型/子中心/正样本聚合损失，验证集早停）
- `siamese_predict.py` + `failure_signature.py`：torch-free 推理 + 失败签名提取（提交包用）
- `run_siamese_ensemble_eval.py`：5-seed 集成评估
- `theta_trace_features.py` + `trace_anchor.py`：层次化 trace（mismatch 聚焦，已被 siamese 用）
- `pairwise_llm_features.py` + `run_graph_multiview_experiments.py`：LLM 特征 + 降维 reducer
- `information_files/`：**竞赛关键文件**（B_20260601.pdf 等，见 §1.6）
- `final_submission/`：打包好的提交包

**已退役（pair 模型，仅历史参考）：**
- `run_final_submission_train.py` / `final_inference.py` / `package_final_submission.py` / `theta_trilog_model.py`
- `graph_clustering.py`（软 k 聚类，siamese 现在用 k-means，不再用它）
- `evaluation_leakage_guard.py` / `GENERALIZATION_AUDIT_20260713.md`：泄漏审计（历史）
- `ICCAD阶段性成果报告.md` / `MATERIALS_INDEX.md` / `README.md`：项目背景与历史
- `handoff.md`：本文件
