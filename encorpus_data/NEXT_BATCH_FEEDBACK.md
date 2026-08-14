# benchmark5_500cases_official 质量反馈 & 下一批造数据建议

> 给数据生成的同学。这份是对 `benchmark5_500cases_official`（500 例 / 10 bug）独立质量审计的结论，
> 重点放在「下一批数据先优化什么」。整体结论：**质量比旧 fake 有质的提升，可以直接用**，
> 但有几个设计层面的问题值得在下一批修掉。

## 一、做得好、请保持的（不要回退）

1. **late first-mismatch**：300 个 mismatch case 的首个 mismatch 都在 index 32+（251 个在 32–255、49 个 256+，最大 1486），
   **彻底去掉了旧 benchmark6 的 cycle-0 `lui` 污染**。这是本轮最大的改进。
2. **每个 bug 的日志都可区分**：6 个 mismatch bug 的首个 opcode 各不相同
   （mul / mulhsu / xor / c.and / c.or / sltu），4 个 UVM/debug bug 的 UVM_FATAL 消息各不相同
   （"Did not receive IN_DEBUG_MODE" / "mstatus.mpie not set" / "read to uninitialized addr" / "mcause wrong"）。
3. **格式 100% 对齐官方**：`input.csv` 头 `Case,Regr Log,Sim Log,Trace Log`、`golden.csv` 头 `Case,Bug`、
   `case_N/{regr.log,sim.log.gz,trace.log.gz}`、Mismatch 块结构，全部与官方 benchmark_set_1 同构。
4. **无泄漏**：日志里 0 个 `bug_` 标签、0 个真实绝对路径（`/workspace/ibex/...` 占位）、gzip 全部可读。
5. **均衡**：10 bug × 50，无偏差。

## 二、下一批要优先修的（按重要性排序）

### 问题 0（最根本，最重要）：消除歧义数据（diff-bugs-same-syndrome）

官方 Q5（`information_files/QA/B_QA_20260612.pdf`）明确说：他们造 benchmark 时专门**剔除了
「不同 bug 产生完全相同症状」的冲突 case**——原话 "we have excluded all these conflicting cases
... diff-bugs-same-syndrome"。原因很直接：**如果两个 bug 在 sim / regr / trace 三种日志里打出来的
东西一模一样，那任何算法都分不对它们**；这类 case 只会污染训练、拉低分数，也无法用来评估模型。

我们旧 benchmark6 就是活生生的反例：56 个 mismatch bug 全部从 cycle-0 `lui` 开始级联，第一个
mismatch 全是 `lui`/`c.li`，日志同质到不可分 → TPR 只有 0.13。**这不是模型不行，是数据本身没有
可区分的信号**，正是官方「消歧」要避免的那类数据。

**造数据的铁律：每个 bug 必须在至少一个特征维度上有唯一、稳定的「症状签名」**（首个 mismatch
opcode / UVM_FATAL 消息文本 + file:line / 分歧 PC 区域 / trace 分歧模式 / 测试名类别…任选其一，越多越稳）。
生成后必须跑一个「歧义自检」：

> **若两个不同 bug 的所有 case 在全部可观测维度上两两不可区分（same syndrome），就删掉其中一个
> bug，或删掉导致冲突的那些 case——绝不留 diff-bugs-same-syndrome 的数据。**

这条优先级高于下面所有问题：下面问题 1/2 本质上是「捷径 / 区分度不足」，而歧义数据是「根本不可能
分对」，属于数据里的硬伤。

### 问题 1（重要）：3 个测试名 → bug 的 1:1 捷径

下面 3 个测试各只覆盖 **1 个 bug**，共 53/500 = **10.6%** 的 case 可以靠「背测试名」直接分对：

| 测试 | 覆盖 bug |
|---|---|
| `riscv_debug_csr_entry_test` (17) | 只有 bug_037 |
| `riscv_debug_ebreak_test` (16) | 只有 bug_037 |
| `riscv_debug_branch_jump_test` (20) | 只有 bug_056 |

**危害**：官方 hidden 集不会给这种捷径（test 和 bug 不是 1:1），模型若在训练时学会「这个测试名 → 这个 bug」，
在 hidden 集上会失效。而且这会让「测试名类别」特征在这 53 例上成为作弊器，掩盖模型真实能力。

**建议**：下一批让**每个 bug 至少出现在 2 个测试里、每个测试至少覆盖 2 个 bug**。
具体说，把 bug_037 和 bug_056 各再挂 1~2 个不同的 debug/interrupt 测试，别让 `csr_entry`/`ebreak`/`branch_jump`
独占一个 bug。

### 问题 2（重要）：6 个 ALU/multdiv bug 同质化过强，只能靠精确 opcode 区分

bug_073/074/078/079/080/081 这 6 个 bug：
- 全是 ALU/MUL 家族（`_family_of` 把 xor/and/or/slt 归为 ALU，mul/mulhsu 归为 MDU）；
- 触发测试高度重叠（arithmetic / machine-rand / rand-instr / RV32IM）；
- mismatch case 的 regr.log 里没有测试名（只有 `Mismatch[1]` + `[FAILED]: N matched`，和官方一致）。

结果：我们的模型用「家族 + 测试名类别」特征时，这 6 个 bug 在**这两类特征上完全同质**，
只能靠「精确 opcode」或「trace 残差 + LLM」区分。实测提交模型在这批数据上 BA=0.69，
**6 个 ALU bug 每个都被拆成 4~5 簇**（最大簇纯度仅 40~72%）——模型分不开它们。

**建议（二选一）**：
- （推荐）**让这 6 个 bug 的测试覆盖错开**：例如 bug_073 偏 arithmetic、bug_074 偏 machine-rand、
  bug_078/079/080/081 各偏不同测试。这样「测试名类别」就能辅助区分，不依赖精确 opcode。
- 或**接受**它们本质是「同家族、同测试、仅 opcode 不同」的解码器 bug——这本身很真实，
  但要知道在这种数据上只有 opcode 级特征能分开，家族级特征天然失效。

### 问题 3（低优先级）：两处轻微格式差异

1. `sim.log` 里 `Compiler version VERSION; Runtime version VERSION`——官方是
   `U-2023.03-SP1_Full64` 这种具体版本串。
2. `sim.log` 的 Command 行路径用 `/workspace/ibex/...`，官方是 `/global/apps/vcs_2023.03-SP1/...`。

**说明**：这两处用占位符反而可能利于泛化（模型不会过拟合到具体版本串/路径），所以**不强求改**。
但如果想让分布和官方尽可能一致，可以把版本串填成真实的 VCS 版本号。

## 三、下一批数据的检查清单（建议内置到生成脚本）

1. **歧义自检（最优先）**：确认任意两个不同 bug 在所有可观测维度（opcode / fatal 消息 / PC 区域 /
   trace 模式 / 测试名）上至少有一个可区分信号；若有 same-syndrome 的冲突 case，**删掉**（见问题 0）。
2. 每个 bug 出现在 ≥2 个测试、每个测试覆盖 ≥2 个 bug（**别再有 1:1 测试名捷径**）。
3. 同家族 bug（ALU/MUL/CSR/...）的测试覆盖尽量错开，别让 6 个 bug 共用同一套测试。
4. 保持 late first-mismatch（首个 mismatch index > 32，杜绝 cycle-0 `lui`）。
5. 日志里 0 个 bug label / 真实绝对路径。
6. 每个 bug 的 case 数均衡（50 左右），每个 bug 的首个 opcode / UVM_FATAL 消息可区分。

## 四、参考

- 官方消歧说明：`information_files/QA/B_QA_20260612.pdf` Q5（「excluded all these conflicting
  cases ... diff-bugs-same-syndrome」）。
- 旧数据教训：`handoff.md` 坑 #13（opcode 被 lui 污染）、#14（测试名用语义类别别用精确 one-hot）。
- 本数据的自审产物：`benchmark5_500cases_bundle/analysis/*.json`。
