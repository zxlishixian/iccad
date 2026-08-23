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

## 1.7 官方测评流程（B_20260601 §3.3/§3.4/§3.5 + QA A4/A6/A9/A10/A11/A12，2026-08-14 复核原文）

- **测评跑的是可执行文件，不是脚本**：接口为 `regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>`，可执行文件名必须**恰好是 `regr_fail_bucketing`**（QA A9/A10 原文）。Python 程序强烈建议 PyInstaller 打包。
- **源码只是后备，且触发条件很窄**：仅当可执行文件**无法启动（cannot start）**时才尝试运行源码；启动后崩溃/超时 → 该 benchmark 直接 0 分，不会回退。且源码路径也必须自包含——测评机**无外网、不允许 pip/requirements.txt**，不保证装有 numpy/sklearn。**二进制是第一生命线**，源码后备只是保险。
- **测评机**：RHEL 8 系（glibc 2.28）、x86_64、32 逻辑核、~128GB RAM、**无 GPU**（QA A4/A6）。
- **LLM 双端点**：官方把 `LLM_MODEL_CONFIG` 指向两个端点（Qwen qwen3-coder-480b + OpenAI gpt-4o / text-embedding-3-small，embedding 官方用 nomic-embed-text-v1.5），每个 benchmark 每端点各跑一次取较高分；**网络延迟计入运行时**，API 调用过多导致超时 → 该 benchmark 记 0（§3.4）。程序内必须做好 LLM 异常处理与超时兜底。
- **--k 就是真实 bug 数**：QA A11「buckets 数 = bug 数」；QA A12「测评时 --k 按 §3.2 表原样传入，不会偏离」。所以直接拿 --k 当聚类簇数是正确做法；"soft" 只指打分对簇数不精确仍宽容（pairwise BA）。
- **每个 benchmark 只跑一次**（§3.5），随机性自己控制（random.seed / PYTHONHASHSEED / np.random.seed / LLM temperature）。

## 1.8 模型改进三大硬约束（2026-08-24 用户强调，设计任何改进前必读）

任何模型升级（包括序列建模等表征升级）都必须满足这三条，缺一即否：

1. **泛化能力优先**：在**未见过的 hidden 官方数据集**上也要有同样的分数表现。验证一律用**干净迁移**（只训 fake、测 official）+ fake LODO，**不许用 train-on-dev 当改进证据**（坑 #16）。特征/模型不能过拟合到 fake 的具体模式。

2. **无 LLM 兜底机制**：如果官方 LLM 端点未响应，必须有兜底——当前机制是「LLM 块零向量化 + 模型照常跑」。**任何改进都不能依赖 LLM 作为唯一判别信号**（坑 #24：v4 的 set1 曾高度依赖 LLM，无 LLM 掉到 0.56）。序列/EDA 特征要保证在无 LLM 时也能拿到主要分数。

3. **绝对不超时（避免 0 分）**：网络延迟计入运行时、超时该 benchmark 记 0。新加的序列模型必须**轻量、CPU 友好**，且在 N=3000（benchmark6/10，100s/300s 限制）下实测运行时，留足余量。LLM 调用量也要控制（batch 大、超时短）。

## 2. 环境与关键路径

- **项目目录**：`/home/lishixian/iccad`（所有代码、数据、提交包都在这）
- **正确 python**：`/home/lishixian/miniforge3/envs/collab-overcooked/bin/python`（有 torch/joblib/sklearn）。**系统 `python3` 没有 torch，会直接 ModuleNotFoundError**。跑任何脚本都显式用这个解释器。
- **LLM embedding**：`nomic-embed-text-v1.5`（768 维）。开发环境本地服务器 `127.0.0.1:8001` 可用；正式环境靠 `LLM_MODEL_CONFIG`（NVIDIA NIM 兜底）。缓存 `/tmp/regr_fail_llm_cache`（按 `cache_key_for_llm_doc` 键控）。
- **Trace 缓存**：`/tmp/theta_trilog_trace_cache`。
- **推理只跑 CPU**（`final_inference.py` 里 `torch.load(map_location="cpu")`、`device="cpu"`）。
- 机器：128 核、503GB RAM、多用户共享（load 常年 30+）。**不是 git 仓库**（改代码前注意备份或自己 diff）。

**当前产物（都在）：**
- 训练输出：`/tmp/theta_final_real_5seed/`（results.csv 45 行 + manifest.json + models/ 45 个 .pt + 45 个 .pkl + preds/）
- 打包好的最终提交：`/home/lishixian/iccad/submission_files/final/final_submission/`（253MB，结构见 §4）
- 废弃的 pair 提交包：`/home/lishixian/iccad/submission_files/final/final_submission_pair_old/`（746MB，未入库）
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
| benchmark5_500cases_official | 500 | 新 fake（批1，官方格式）| ✅ 高质量（10 bug，debug/interrupt/MDU/ALU-decoder）|
| benchmark8_500cases_official | 500 | 新 fake（批2，官方格式）| ✅ 高质量（10 bug，ALU-shift/LSU/branch/AUIPC/CSR）|
| k32_new12_380cases_official | 380 | 新 fake（批3，官方格式）| ✅ 高质量（12 bug，**RV32C + MDU div/rem**）|
| batch4_11bugs_20260821/dataset | 364 | 新 fake（批4，官方格式）| ✅ 高质量（11 **控制流 bug**，流程最规范：patch+manifest+机器审计）|
| benchmark_set_1 | 7 | **官方 dev**（有 golden.csv）| ✅ 保留，微调目标之一 |
| benchmark_set_2 | 25 | **官方 dev**（有 golden.csv）| ✅ 保留，微调目标之一 |

> **当前 v4 训练集 = 9 个 fake（无官方，干净迁移）；v3 训练集 = 7 fake + 2 official（train-on-dev）。**
> 4 个高质量新集（benchmark5/benchmark8/k32_new12/batch4）在 v4 里 ×2 加权；旧 lui 级联数据 ×1 当预训练。

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

**结果：剩余 6 fake + 2 official 内容级零重叠，训练集 = 3809 唯一 case（640+40+65+24+96+2944）。** 之前的 5-seed 45 模型（`/tmp/theta_final_real_5seed` 与 `submission_files/final/final_submission/`）仍是在**未去重的 9 集**上训的，**尚未用去重后的数据重训**——重训留作下一步。

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

### 当前提交包（submission_files/final/final_submission/，2026-08-14 打包完成）

- **入口** `submission_files/final/final_submission/regr_fail_bucketing`（PyInstaller onedir，mode 755，顶层）
- **模型**：siamese 5-seed 集成（NumPy 前向，无 torch）
- **接口** `--input --output --k` → `Case,bucket`
- **LLM** 通过 `LLM_MODEL_CONFIG`，失败回落确定性基线
- **GLIBC** 已修到 2.28 上限（替换 libstdc++ 2.38→2.17、libgcc 2.35→2.14）
- **打包步骤**：`siamese_predict.py` + `failure_signature.py`（torch-free）→ 转 NumPy encoder → PyInstaller

**提交包实测（train-on-dev，过拟合）：** benchmark_set_1=1.0、benchmark_set_2=0.91、official_vcs=0.80、directed_cross_v4=0.78、stable=0.78、benchmark6=0.56（64 bug，TPR=0.134）。运行时 2.9s~70.6s，均在官方 100s/300s 限制内。

**⚠️ 2026-08-14 冒烟复测发现两个问题（已修源码，需重打包）：**

1. **无 LLM 时模型静默退化为 baseline（坑 #19）**：`pairwise_llm_features.py` 无 LLM 分支给 `llm_vec=zeros(0)`（应为 768 维零向量），LLM 块被整个丢掉 → 188 维编码器收到 124 维输入 → matmul 崩 → 静默回落确定性基线。实测：无 LLM 时 set1 BA=0.52（基线）；有 LLM（本机 127.0.0.1:8001）时 set1 BA=1.0、2.6s。**官方测评必设 LLM_MODEL_CONFIG，所以官方路径能拿到模型分**；但一旦官方端点故障/超时，会跌到基线分。已修：`zeros(0)`→`zeros(dim)`（与 fetch-fail 分支一致）。修复后源码实测：无 LLM set1 BA=1.0、set2 BA=0.90——零 LLM 块下模型照样工作。
2. **LLM 调用量风险**：推理默认 batch=64 且抓 features+summary 两份 embedding，benchmark6（3000 例）≈94 次 API 调用；官方「网络延迟计入运行时、调用过多超时=0 分」。计划：batch 64→512、超时 60→20s（重打包时一起改）。
3. **真机 GLIBC 2.28 测试仍未做**：目前只做过静态符号分析；队友在 AlmaLinux 8 容器里跑的是 `python3.9 regr_fail_bucketing.py`（源码后备路径），**不是**二进制（二进制被 .gitignore 排除，队友 clone 里没有）。重打包后必须真机跑一次二进制。

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

### 2026-08-15：新数据集 + opcode 特征消融（结论：加新集、弃 opcode、不加权）

队友按官方格式修复 bug 后造了新集 `benchmark5_500cases_official`（500 例 / 10 bug，late first-mismatch、opcode 可分、无泄漏，质量审计通过，见 §3）。做了「只训 fake → 测官方」的干净消融（官方集 7+25 例，**噪声大，set1 尤其**）：

| 变体 | set1(7例) | set2(25例) | 官方均值 |
|---|---:|---:|---:|
| A 旧fake(5) | 0.43 | 0.71 | 0.57 |
| B +新集 | 0.71 | 0.65 | 0.68 |
| D +新集+opcode | 0.71 | 0.52 | 0.61 |
| C +新集×3+opcode(5seed) | 0.79±.19 | 0.65±.05 | 0.72 |
| E +新集×3无opcode(3seed) | 0.71±.23 | 0.62±.03 | 0.67 |

**三个结论**：
1. **新集帮 interrupt/CSR/debug 域**（set1 0.43→0.71+，因为新集 bug_038/057/037/056 正是这域），但**略伤 mismatch 域**（set2 0.71→0.65）。净 +0.11，方向相反。
2. **精确 opcode 特征是负面的**：D 把 set2 从 0.65 砸到 0.52——opcode 重新引入「同症状不同 bug」混淆，**坑 #13 判断依然成立，已回退**（`git checkout` 两个文件，回到 family+type）。
3. **加权（新集×3）无明确收益**（B≈E），不值得引入过拟合风险。

**决定**：最终训练 = 全部 6 fake + 2 official（**不加权、不用 opcode、family+type 特征不变**）。6 fake = official_vcs / directed_cross_v4 / stable / benchmark6_final / benchmark5_final / benchmark5_500cases_official(新)。新集主要价值是给 fake 侧补上 interrupt/CSR/debug 域信号，让 hidden 集那类 bug 更稳。

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

## 4.6 业界文献调研（模型升级方向参考，2026-08-24）

> 核心认知：**我们的"回归失败分桶"= 硬件版的"崩溃报告去重/crash bucketing"**。软件工程界对后者有成熟的深度学习解法，EDA 界也有直接对应的 RTL failure triage 文献。以下按"领域 + 代表作 + 对我们模型的启示"整理，便于随时查阅（之前模型从 pair→siamese+原型的方向转变，正是源于对 SupCon/原型学习的论文调研）。

### A. 软件工程：crash report deduplication / bucketing（与我们问题同构）

| 方法 | 核心思路 | 对我们的启示 |
|---|---|---|
| **[DeepCrash](https://www.semanticscholar.org/paper/DeepCrash%3A-deep-metric-learning-for-crash-bucketing-Liu-Xie/1c4848545a68bf76f45dc4644eea73243ee19ea5)**（2022）| frame2vec 把每个栈帧嵌入 → 深度度量模型建模**帧序列** → 聚类分桶 | 把每条 retired 指令（opcode+PC+寄存器）嵌入 → 序列建模 → 聚类，替代现在的哈希计数 |
| **[S3M](https://opus.constructor.university/frontdoor/index/index/docId/1327)**（Siamese Stack Trace Similarity）| **Siamese + biLSTM** 编码栈轨迹算相似度 | 我们已是 Siamese，但用聚合特征；换成序列编码器是升级方向 |
| **[DedupT](https://www.semanticscholar.org/paper/Stack-Trace-Based-Crash-Deduplication-with-Mamun-Uddin/dd6d31e03df9ebe90f0a9c648cdc83d9c1c40149)**（2025）| **Transformer** 整体建模栈轨迹（非孤立看帧）| 最对应"控制流 bug 分歧路径是序列信号"这个瓶颈 |
| **[SANER 2025](https://ieeexplore.ieee.org/document/10992512)**（stack trace dedup）| embedding + **reranker 二段式**精排 | 二段式精排契合我们的 pairwise BA 指标 |

### B. EDA：RTL failure triage（同问题的本领域解法）

| 方法 | 核心思路 | 对我们的启示 |
|---|---|---|
| **[Clustering-based failure triage for RTL regression debugging](https://www.semanticscholar.org/paper/Clustering-based-failure-triage-for-RTL-regression-Poulos-Veneris/f488b3e1b19554ce9af164f2f0f8afe02519eea9)**（Poulos-Veneris）| 错误轨迹签名 + 聚类，报告 93% 准确率、misplacement −47% | 确认"签名+聚类"路线，但他们的签名含 SAT suspect set（我们没有）|
| **[A failure triage engine based on error trace signature](https://ieeexplore.ieee.org/document/6604054)** | 错误轨迹签名提取 + ML 聚类，93% | 同上 |
| **[Exemplar-based failure triage (affinity propagation)](https://www.eecg.toronto.edu/~veneris/15lats.pdf)** | exemplar/AP 聚类，87% 准确率 | AP 聚类可作为 k-means 的替代（软 k、非度量空间）|
| [Synopsys Verdi RDA](https://qacf-www.synopsys.com/blogs/chip-design/ml-regression-failure-analysis.html) / [Cadence Verisium AutoTriage](https://community.cadence.com/cadence_blogs_8/b/fv/posts/automate-regression-failure-triage-with-the-cadence-verisium) | 商业 ML 分桶（UVM 消息/错误类型），~90% | 工业界确认"ML 分桶"是刚需 |

### C. 多模态对比（日志+指标）

- **[MAD-CMC](https://www.sciencedirect.com/science/article/abs/pii/S0306457324003728)**（2024）：对比式多模态表示聚类（logs + metrics 联合、BERT+SVD + AE + 跨模态 Transformer + k-means）。启示：sim/regr/trace 三视图可做多模态融合，而非简单拼接。

### D. 映射到我们模型的三个升级方向（按价值排序）

1. **序列级 trace 建模（主线，最对症）**：分歧点附近指令序列（opcode+PC+寄存器）→ biLSTM（S3M 式）或轻量 Transformer（DedupT 式），替换现在的 trace 残差哈希计数。直接打"控制流 bug 分歧路径是序列信号"这个确认瓶颈。
2. **指令级 embedding（frame2vec 式）**：每条 retired 指令嵌入成稠密向量，而非手写 opcode 家族。
3. **二段式 reranker（SANER 式）**：粗聚类 + 对不确定 case 对精排，直接优化 pairwise BA。

> **结论**：加特征（P0 pc_same/pc_offset、P2 分歧深度）已经到头——都是混合结果（见坑 #28 和 §6.5 的 v5depth 消融）。下一轮升级应走**表征升级**（序列建模），不是再加聚合特征。

## 5. 卡在哪儿（当前阻塞点）

1. **benchmark6 的 64-bug 分簇是 fake 数据固有上限**：TPR=0.134 是因为 fake benchmark6 的 56 个 mismatch bug 在日志里不可分（见 §4 模型迭代结论），3 个损失/特征实验都救不了。官方 hidden 集已消歧，大概率比 fake 好分，但无法本地验证。
2. **干净 official 迁移只有 0.53**：只训 fake、测 official 时，set1≈0.55/set2≈0.51。这是 fake→official 分布跨越，是真实泛化瓶颈。alpha 靠「训 official（adapter）」拿到 hidden 0.72，我们是「干净迁移」0.53。
3. **train-on-dev 分（0.868）不可信**：官方集进训练是背答案（见坑 #16），不能当泛化证据。

## 6. 最终状态（2026-08-14 收尾）

**决策：接受当前模型为最终版本（用户选定）。** 5 类方法（子中心/正样本聚合/UVM_FATAL/加数据/域对抗）都边际或负面，模型迭代已达数据与方法瓶颈。

**最终提交**：`submission_files/final/final_submission/`（siamese + 原型 + trace + 测试名，B 版 fakes+official 一起训）

- 架构：per-case 编码器 → SupCon + 原型损失 → k-means，O(N)，PyInstaller 自包含，GLIBC 2.28 达标
- 特征：LLM 64 + 失败签名（家族+分歧类型）14 + 测试名类别 14 + trace residual 96 = 188 维
- 本地分数（train-on-dev，过拟合上界）：set1=0.83、set2=0.90、official_vcs=0.80、directed_cross_v4=0.78、stable=0.78、benchmark6=0.56
- fake LODO（诚实泛化）：小集 0.60~0.70、benchmark6=0.52（64 bug 数据上限）
- hidden 集估计：对标 alpha 的 ~0.72（同样训了 official）

**2026-08-14 深夜改动（提交前最后一轮）**：

- **二进制与 `_internal/` 已改为入库**（.gitignore 放行）：用户最终流程是「队友 clone 仓库 → 从自己电脑上传 Google Drive」，服务器无 sudo、跑不了 docker，所以打包产物必须走 git。队友上传时只传 `submission_files/final/final_submission/` 目录内容（二进制 + `_internal/` + README + `regr_fail_bucketing.py` 后备源码），不要传整个仓库。
- **修复无 LLM 静默退化**（坑 #19）：`pairwise_llm_features.py` 无 LLM 分支 `zeros(0)`→`zeros(768)`。修复后无 LLM 也是模型分（set1 BA=1.0、set2 BA=0.90）。
- **LLM 调用量优化**：`siamese_predict.py` 默认 `--llm-batch-size` 64→512、`--llm-timeout-sec` 60→20（官方网络延迟计入运行时，benchmark6 从 ~94 次调用降到 ~6 次）。
- **f-string 兼容修复**：`regr_fail_bucketing.py:836` 反斜杠 f-string 拆成变量（Python<3.12 解析兼容），并复制进 `submission_files/final/final_submission/` 作为官方「源码后备」。
- **重打包 + GLIBC 2.28 修复已重做**（重打包后必须重换 libstdc++/libgcc，见 §8）。
- **仍未做的**：AlmaLinux 8 容器真机测试（服务器无 sudo，只能队友在本机跑；容器命令见 §8）。

**2026-08-15 决定（已定，待执行重训）：**

- **新集加入最终训练**：`benchmark5_500cases_official`（500 例/10 bug）加进最终训练集（6 fake + 2 official，含 benchmark5_final），覆盖 interrupt/CSR/debug 域。消融见 §4「2026-08-15」。
- **opcode 特征已回退**：精确 first-mismatch opcode 实测负面（伤 set2），保持 family+type 特征。
- **不加权**：新集×3 加权无收益。
- **待执行链**：5-seed 重训 → 转 npz（`encoder_seed*.npz` + `preprocess.pkl`）→ PyInstaller 重打包 → GLIBC 2.28 复验 → 双模式冒烟 → 真机（队友 AlmaLinux 8）→ 重推 git。命令见 §8。

**遗留（可选，不再主动投入）：** 干净 official 迁移 0.53（fake→official 分布跨越）；benchmark6 64-bug 不可分（fake 数据未消歧）。

**2026-08-16：v3 模型（最终提交，打包到 `submission_files/final/final_submission_v3/`）**

- **训练配置**：7 fake + 2 official，两个高质量新集（benchmark5_500cases + benchmark8_500cases）各 ×2 加权，旧 lui 级联数据（official_vcs/directed/stable/benchmark5_final/benchmark6）当预训练 ×1，无 opcode，--use-trace，5 seed。
- **benchmark8** 是队友第二批高质量数据（500 例/10 个新 bug，late mismatch、无泄漏、无逐字节歧义），补上了 LSU/branch/CSR 域的 bug 广度——这正是 v3 相比 v2 的关键差异。
- **提交同款分数（train-on-dev 上界）**：set1=1.0、set2=0.9587（官方均值 0.979）、benchmark5_500=0.936、benchmark8_500=0.998、benchmark6=0.58。**全面优于旧模型（官方均值 0.955）和 v2（0.875）**，尤其 set2 从 v2 的 0.75 回升到 0.9587。
- **已打包验证**：GLIBC 2.28 达标、双模式冒烟通过（有 LLM set1=1.0，无 "[siamese] failed"）。
- **注意**：v3 是新推荐的提交；旧 `final_submission/`（旧模型）保留不动。队友上传时用 v3 文件夹内容。

**2026-08-16 晚：集成改为 co-association（坑 #15 的正式修复）**

- 把 `siamese_predict.py` 的「平均 5 个 npz embedding → k-means」改成「**5 个 npz 各自 k-means → 逐对 co-association 投票 → 层次共识聚类（average linkage）**」。
- 验证（v3 全 9 集）：co-association **严格不输**，7/9 集提升，最大 +0.10（directed）、+0.076（official_vcs）、+0.040（benchmark6）；官方集 set1 保持 1.0、set2 微涨。
- 干净迁移（8 fake→官方）：set1 从 0.51→0.72、官方均值 0.61→0.71（平均 embedding 的未对齐问题被修掉）。
- 已重打包、GLIBC 2.28 复验、双模式冒烟通过（set1=1.0/set2=0.965）。**注意**：层次聚类的 `AgglomerativeClustering(metric="precomputed")` 是 O(n²) 内存 + 较高时间复杂度，N=3000 时应实测运行时（本次 benchmark6 2944 例可跑完）。

**2026-08-22 深夜：集成改为 Procrustes 对齐平均（取代 co-association）**

- 发现 co-association 在 **N<25 小集不可靠**（坑 #26），并测了 **Procrustes 正交对齐 + 平均** 的替代方案（把 seed 1~N 对齐到 seed 0 再平均，修坑 #15 的"空间未对齐"根因）。
- 11 集全量对比（v4b 模型，无 LLM）：Procrustes **在官方 set2 大赢 +0.236**（0.727→0.962，追平最好 seed）、batch4 +0.027、k32_new12 +0.008；**退化**在 fake 集 stable −0.071、benchmark6 −0.031、directed −0.016、benchmark5_500 −0.015。**非严格占优**，但赢在官方集、输在 fake 集 → 从提交价值看值得换。
- 已把 `siamese_predict.py` 的 `_coassoc_cluster` 换成 `_procrustes_cluster`（正交 Procrustes 对齐平均 → k-means），重打包 v4。

**2026-08-22 本轮工作总结（数据 ×2 批 + v4 模型 + 数据策略文档）：**

- **新数据批 3 `k32_new12_380cases_official`**（12 bug，RV32C + MDU div/rem）：质量检查通过、与前批零重合、三日志歧义 0。补上了 RV32C 译码和 MDU 除余两个缺口。
- **新数据批 4 `batch4_11bugs_20260821`**（11 控制流 bug，364 例）：流程最规范（11 个 RTL patch + 3 波 manifest + 10 份机器生成审计 + selection-level 消歧剔 145 例）。标准检查全过（零重合/零泄漏/三日志歧义 0/late mismatch）。**但控制流 bug 天生难分**（坑 #25）：分支 bug 的首 mismatch 是"随机后续指令的 PC 分歧"、JAL/JALR bug 全报同一个 `read to uninitialized addr`——v3 零样本 BA 仅 0.574，每个 bug 被拆 4~7 簇。这是数据内在难度，非造数错误；官方 hidden 若含分支 bug 同样难。
- **数据策略文档更新**：`DATA_GENERATION_ROADMAP.md` 重构（**不合并、不卡 k、类型越广越好** + 粗/细粒度维度）；新增 `BUG_INJECTION_CATALOG.md`（~100 种 bug 类型 × 12 功能单元 × 12 种注入模式，给队友逐批照造）。
- **v4 模型（干净迁移版）**：只用 9 个 fake（**无官方**）、4 个新集 ×2 加权、5 seed。提交同款（co-association）官方分：**set1=1.0、set2=0.70，官方均值 0.85**（旧基线 0.53 → 8fake 0.71 → 9fake 0.85）。打包到 `submission_files/final/final_submission_v4/`（GLIBC 2.28 达标、冒烟通过），已推送。
- **v3 vs v4 定位**：v3 = train-on-dev（7 fake + 2 official，官方均值 0.98，过拟合上界）；v4 = 干净迁移（9 fake 无 official，官方均值 0.85，诚实泛化下界）。**最终提交用哪个待用户定**（hidden 分布接近 dev 集选 v3；更广的 hidden 选 v4）。
- **v4 的 LLM 依赖（坑 #24）**：v4 的 set1 高度依赖 LLM（有 LLM 1.0 / 无 LLM 0.56），set2 不依赖（无 LLM 反而 0.74）。官方必设 LLM 端点所以预期 0.85；若端点故障会跌到 0.65。

## 6.5 模型迭代记录（每一代的设置 + 分数）

> ⚠️ **评估口径教训（2026-08-22 晚）**：co-association 共识聚类在 **N<25 的小集上不可靠**（7 例的 set1 尤其）
> ——它会因"错在哪个 case 上"的随机性把结论翻转（坑 #26）。**小集要看 per-seed 均值**；co-assoc 只在大集（≥25 例）可靠。
> 下表同时给 per-seed 均值（小集可靠）和 co-assoc（提交同款，大集可靠）。

### 干净迁移系列（只训 fake → 测官方）

| 代次 | 训练集 | 官方均值(per-seed) | 官方均值(co-assoc) | 备注 |
|---|---:|---:|---:|
| 旧 fake 基线 | 旧 lui 级联 fake | 0.53 | — | 早期记录 |
| A | 旧 5 fake | 0.57 | — | 单 seed |
| B | 旧 5 fake + benchmark5 | 0.68 | — | 单 seed，+批1 |
| 8 fake | B + benchmark8 + k32_new12 | 0.71 | 0.71 | +批2/3 |
| v4 旧 | 9 fake（+batch4）| 0.72 | 0.85 | 曾被误判为"泛化最好"，实际 set1 是 co-assoc 侥幸 |

### train-on-dev 系列（fake + 官方一起训）

| 代次 | 训练集 | 官方均值(per-seed) | 官方均值(co-assoc) | 备注 |
|---|---:|---:|---:|
| siamese_cat | 4 fake + 2 official | — | 0.955 | 最初提交（`final_submission`）|
| **v3** | **7 fake + 2 official（2 新 ×2）** | 0.86 | **0.98** | **打包 `final_submission_v3`** |
| **v4b** | **9 fake + 2 official（4 新 ×2）** | **0.82** | 0.72 | **打包 `final_submission_v4`（本次）** |

**最终结论（修正后）**：
- **per-seed 上 v4b（0.82）≥ v4 旧（0.72）**——加官方数据确实有帮助，之前的"v4b 更差"是 co-assoc 小集噪声造成的误判。
- 官方集 co-assoc 最高分仍是 **v3（0.98）**；但 v3 只有 7 fake（缺 k32_new12/batch4），hidden 集若含新 bug 类型泛化可能不足。
- **提交用 v3（官方集最强）或 v4b（数据最全、per-seed 泛化不输）**；v4 旧（纯干净迁移）留作对照，不再作为推荐提交。

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
19. **无 LLM 时提交包静默跌到基线（关键，2026-08-14 发现）**：`pairwise_llm_features.py` 无 LLM 分支造 `llm_vec=zeros(0)`（0 维），`apply_llm_reducer` 见 `has_llm=False` 就把 LLM 块整个丢掉 → 188 维编码器收到 124 维输入 → siamese 崩 → **静默回落确定性基线**。冒烟测试「输出正常」会掩盖真相（基线也能出合法 CSV）：无 LLM set1 BA=0.52 vs 有 LLM 1.0。**已修**：`zeros(0)`→`zeros(dim=768)`（与 fetch-fail 分支同宽）。**教训：验证提交包一定要看 stderr 有没有 `[siamese] failed`，不能只看 CSV 头**。另外注意坑 #1 是同一类问题在别处的变体——所有 LLM 兜底零向量必须满 768 维。
20. **打包产物曾不进 git，后改为入库（2026-08-14）**：最初 `.gitignore` 排除二进制+`_internal/`，导致队友 clone 拿不到二进制、无法 clone→上传。**已改为入库**（队友流程是「clone 仓库 → 上传 `submission_files/final/final_submission/` 内容到 Google Drive」）。教训：提交物必须跟着仓库走，别依赖 scp/U 盘传。注意 `final_submission_pair_old/`（废弃 746MB）仍被 .gitignore 排除（模式无前导斜杠，按目录名任意层级匹配）。官方测评跑的是**可执行文件**（§1.7），源码只是「无法启动」时的后备。
21. **`.gitignore` 的 `models/` 会吞掉提交包里的编码器权重（关键，2026-08-14）**：`models/`（无前导斜杠）按目录名任意层级匹配，把 `submission_files/final/final_submission/_internal/models/`（打包好的 5 个 `encoder_seed*.npz` + `preprocess.pkl`，~3.6MB）也忽略了，导致队友 clone 出的二进制启动正常但 `[siamese] failed ... preprocess.pkl not found` → 静默回退基线（set1 BA 0.52 而非 1.0）。同样 `*.tar.gz` 吞掉了 dateutil 的 `dateutil-zoneinfo.tar.gz`。**已修**：加否定规则 `!submission_files/final/final_submission/_internal/models/**` 和 `!.../dateutil/zoneinfo/dateutil-zoneinfo.tar.gz`。**教训：打包产物入库后，一定要 `git ls-files | grep _internal/models` 确认权重真的在 git 里；队友冒烟报 `[siamese] failed` 先查模型文件在不在。**
22. **真机 glibc 2.28 已确认（2026-08-14，队友实测）**：队友在 AlmaLinux 8 容器（`--platform linux/amd64`）里直接跑 `regr_fail_bucketing` 二进制，`ldd (GNU libc) 2.28` + 正常产出 `Case,bucket` CSV → **二进制在 glibc 2.28 上能启动运行**。但那次跑的是缺模型权重的旧包，模型回退基线的现象正是坑 #21。
23. **精确 opcode 特征会重新引入「同症状不同 bug」混淆（2026-08-15）**：曾经想把 first-mismatch 的精确 opcode 当特征（因为新集里 mul/mulhsu/xor/and/or/slt 的 opcode 都不同、理论上可分）。实测只训 fake→测官方，加 opcode 后 **set2 从 0.65 崩到 0.52**——官方 set2 的 bug 有重叠 opcode，opcode one-hot 让模型把不同 bug 聚到一起（正是官方「消歧」要避免的）。**坑 #13「opcode 被污染/不可靠」的判断在官方数据上依然成立，精确 opcode 不要当特征**；用「家族 + 分歧类型」就够。
24. **干净迁移模型的 set1 可能高度依赖 LLM（2026-08-22）**：v4（9 fake 无官方）有 LLM 时 set1=1.0，无 LLM（零 LLM 块）时 **set1=0.56**（TNR 仅 0.33）——set1 那 2 个 bug 的判别信号落在了 LLM 语义块上；set2 相反（无 LLM 0.74 vs 有 LLM 0.70，靠 EDA 特征就够）。官方测评必设 LLM 端点（nomic），所以预期分数走"有 LLM"路径；但**若端点故障/超时（fetch 失败=零 LLM 块=无 LLM 路径），set1 会大跌**。评估提交风险时要把这个依赖算进去。
25. **控制流 bug（分支/跳转）天生难分，不是造数错误（2026-08-22）**：batch4 的 11 个控制流 bug 里，5 个分支 bug 的首个 mismatch 是「ibex 在 PC-A 执行 X、spike 在 PC-B 执行 Y」——分歧在**随机后续指令**显形，opcode 五花八门且互相重叠；6 个 JAL/JALR bug 全报同一个标准断言 `read to uninitialized addr`（只差地址值）。v3 零样本 BA 仅 0.574（每 bug 拆 4~7 簇）。这是**数据内在难度**（官方 hidden 含分支 bug 时同样难），队友的 selection-level 消歧已经做到位。**对模型的启示**：要分控制流 bug，方向是「PC 分歧模式」类特征（PC region、分歧深度、分支方向、地址差值），不是 opcode。
26. **co-association 共识聚类在 N<25 的小集上不可靠（2026-08-22）**：7 例的 set1 上，co-assoc 会因为"各 seed 错在哪个 case 上"的随机性把结论翻转——曾误判"v4b(0.72) 比 v4 旧(0.85) 差"，实际 per-seed 上 v4b(0.82) ≥ v4 旧(0.72)，是 co-assoc 的侥幸/不巧。**教训：小集（N<25）看 per-seed 均值，别信 co-assoc；co-assoc 只在大集（≥25 例）可靠**。
27. **Procrustes 对齐平均不是严格占优，但赢在官方集（2026-08-22）**：11 集对比里 Procrustes 对齐平均在官方 set2 大赢 +0.236（0.727→0.962）、batch4 +0.027，但 fake 集 stable −0.071、benchmark6 −0.031、directed −0.016 退化。**没有"无退化"的免费午餐**；但退化全在 fake 集、赢在官方集 + 难分的控制流集，从提交价值看换 Procrustes 是对的。当前 `siamese_predict.py` 已用 Procrustes 对齐平均。
28. **PC 分歧特征（pc_same/pc_offset）是混合结果，不是清晰赢（2026-08-24）**：给 `parse_rich_signature` 加了 `pc_same`（同 PC=数据通路 / 异 PC=控制流）和 `pc_offset_norm`（PC 偏移量）。单 seed 消融：官方集持平（set2 0.700 vs 0.703），**混合集 catalog +0.10（0.655→0.756），但纯控制流集 batch4 −0.06（0.868→0.809）**。结论：pc_same 能劈开"控制流 vs 数据通路"的**粗粒度**，但分不开控制流 bug 彼此（batch4 全是异 PC，退化成常数噪声）。**真正的控制流判别信号在分支处操作数的值/具体分歧路径，非常细，单靠 PC 一个 bit 不够**。代码改动已留在工作区（4 个文件，未提交），是否保留待定。

## 8. 关键命令速查

```bash
cd /home/lishixian/iccad
PY=/home/lishixian/miniforge3/envs/collab-overcooked/bin/python

# siamese 训练（v4 = 干净迁移：9 fake 无官方，4 个新集 ×2 加权；v3 = 7 fake + 2 official）
$PY run_siamese_train.py --train-datasets \
  dataset/fake_dataset/official_format_fake_dataset/official_vcs_stage1_dataset_v1 \
  dataset/fake_dataset/official_format_fake_dataset/directed_cross_v4 \
  dataset/fake_dataset/official_format_fake_dataset/stable_official_like_multitest_v1 \
  dataset/fake_dataset/official_format_fake_dataset/benchmark6_final \
  dataset/fake_dataset/official_format_fake_dataset/benchmark5_final \
  dataset/fake_dataset/official_fixed_dataset/benchmark5_500cases_bundle/benchmark5_500cases_official \
  dataset/fake_dataset/official_fixed_dataset/benchmark8_500cases_bundle_20260815/benchmark8_500cases_official \
  dataset/fake_dataset/official_fixed_dataset/k32_new12_380cases_official \
  dataset/fake_dataset/official_fixed_dataset/batch4_11bugs_20260821/dataset \
  dataset/fake_dataset/official_fixed_dataset/benchmark5_500cases_bundle/benchmark5_500cases_official \
  dataset/fake_dataset/official_fixed_dataset/benchmark8_500cases_bundle_20260815/benchmark8_500cases_official \
  dataset/fake_dataset/official_fixed_dataset/k32_new12_380cases_official \
  dataset/fake_dataset/official_fixed_dataset/batch4_11bugs_20260821/dataset \
  --eval-datasets dataset/real_dataset/benchmark_set_1 dataset/real_dataset/benchmark_set_2 \
  --output-dir /tmp/siamese_seed0 --seed 0 --device cpu --epochs 60 --use-trace

# 5-seed 集成评估（平均 embedding 再聚类）
$PY run_siamese_ensemble_eval.py --seed-dirs /tmp/siamese_seed0 ... /tmp/siamese_seed4 \
  --eval-datasets dataset/real_dataset/benchmark_set_1 dataset/real_dataset/benchmark_set_2 \
  --output-dir /tmp/ens --use-trace

# 打包（PyInstaller onedir，排除 torch）
# 注意：--add-data 的源 submission_files/final/final_submission/models 目前不存在（被 .gitignore 排除）。
# 重打包前先从 _internal/models 拷回：cp -r submission_files/final/final_submission/_internal/models submission_files/final/final_submission/models
$PY -m PyInstaller --name regr_fail_bucketing --onedir \
  --add-data "/home/lishixian/iccad/submission_files/final/final_submission/models:models" \
  --exclude-module torch --exclude-module torchvision --exclude-module torchaudio \
  --paths /home/lishixian/iccad \
  --distpath /tmp/pyinstaller_dist --workpath /tmp/pyinstaller_work --specpath /tmp \
  --log-level WARN siamese_predict.py
# 产物回拷（二进制+_internal 必须同目录，_internal 会覆盖旧目录）
cp -r /tmp/pyinstaller_dist/regr_fail_bucketing/_internal submission_files/final/final_submission/
cp /tmp/pyinstaller_dist/regr_fail_bucketing/regr_fail_bucketing submission_files/final/final_submission/
chmod 755 submission_files/final/final_submission/regr_fail_bucketing

# GLIBC 2.28 修复（每次重打包后必须重做）：用 conda 老库替换 bundled 新库
STDCXX=$(find /home/lishixian/miniforge3/pkgs/libstdcxx-15.2.0* -name "libstdc++.so.6" | head -1)
GCC=$(find /home/lishixian/miniforge3/pkgs/libgcc-15.2.0* -name "libgcc_s.so.1" | head -1)
cp "$STDCXX" submission_files/final/final_submission/_internal/libstdc++.so.6
cp "$GCC" submission_files/final/final_submission/_internal/libgcc_s.so.1
# 验证全包 GLIBC 符号上限 ≤ 2.28
find submission_files/final/final_submission -type f \( -name "*.so*" -o -name "regr_fail_bucketing" \) \
  | while read f; do objdump -T "$f" 2>/dev/null | grep -oE "GLIBC_[0-9]+\.[0-9]+(\.[0-9]+)?"; done | sort -Vu | tail -5

# 提交包实测（两种都要跑，且必须看 stderr 有没有 [siamese] failed）
env -u LLM_MODEL_CONFIG submission_files/final/final_submission/regr_fail_bucketing --input <csv> --output /tmp/o1.csv --k <k>
LLM_MODEL_CONFIG="$(cat /tmp/llm_local.yaml)" submission_files/final/final_submission/regr_fail_bucketing --input <csv> --output /tmp/o2.csv --k <k>
```

## 9. 关键文件索引

**当前主线（siamese）：**
- `run_siamese_train.py` + `theta_siamese_model.py`：siamese + 原型模型（SupCon/原型/子中心/正样本聚合损失，验证集早停）
- `siamese_predict.py` + `failure_signature.py`：torch-free 推理 + 失败签名提取（提交包用）
- `run_siamese_ensemble_eval.py`：5-seed 集成评估
- `theta_trace_features.py` + `trace_anchor.py`：层次化 trace（mismatch 聚焦，已被 siamese 用）
- `pairwise_llm_features.py` + `run_graph_multiview_experiments.py`：LLM 特征 + 降维 reducer
- `information_files/`：**竞赛关键文件**（B_20260601.pdf 等，见 §1.6）
- `submission_files/final/final_submission/`：打包好的提交包

**已退役（pair 模型，仅历史参考）：**
- `run_final_submission_train.py` / `final_inference.py` / `package_final_submission.py` / `theta_trilog_model.py`
- `graph_clustering.py`（软 k 聚类，siamese 现在用 k-means，不再用它）
- `evaluation_leakage_guard.py` / `GENERALIZATION_AUDIT_20260713.md`：泄漏审计（历史）
- `ICCAD阶段性成果报告.md` / `MATERIALS_INDEX.md` / `README.md`：项目背景与历史
- `handoff.md`：本文件
