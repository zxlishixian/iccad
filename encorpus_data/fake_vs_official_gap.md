# 我们 fake 数据 vs 官方数据 的差异清单（给队友修复用）

> 目的：列出我们自造 fake 数据和官方数据的**具体差异**，让队友按此修复数据生成流程。
> 关联文档：`official_bug_injection.md`（官方注入方法）、`official_bug_distribution.md`（官方 bug 分布）

## 核心差异（按重要性排序）

### 🔴 差异 1：mismatch 是「cascade」不是「单点」—— 最致命

| | 我们 benchmark6 | 官方 |
|---|---|---|
| 一个 case 的 mismatch 数 | **28004 个**（`[FAILED]: 131 matched, 28004 mismatch`）| **单个** `Mismatch[1]` |
| regr.log 打印 | 6 个 Mismatch 头（Mismatch[1..5] + Mismatch[28004]）| 1 个 Mismatch[1] |
| 原因 | `mismatch_print_limit` 高 + bug 从 cycle 0 就开始分歧 | `mismatch_print_limit=1`（官方 QA Q13 明确）|

**问题**：我们的 bug 从 cycle 0（第一条 `lui` 初始化指令）就开始 cascade 分歧，导致**每个 bug 的症状都是「从 lui 开始全错」**，互相不可区分。

**修复**：把 `mismatch_print_limit` 调成 1，且注入的 bug 要**在具体功能单元指令处才显形**（不是从第一条指令就全错）。

### 🔴 差异 2：第一个 mismatch 是「初始化指令」不是「功能单元指令」

| | 我们 benchmark6 | 官方 benchmark_set_2 |
|---|---|---|
| 第一个 mismatch 指令 | **lui（1761 例）、c.li（612 例）** —— 全是初始化 | remu、mulhsu、ori、srai、c.addi —— 功能单元指令 |

**问题**：我们的第一个 mismatch 全是 `lui`（load upper immediate，程序初始化），说明 bug 在**极早期（取指/PC）**就触发，导致后面全部 cascade。官方第一个 mismatch 是**具体功能单元指令**（MDU/ALU），说明 bug 在**具体指令执行**时才显形。

**修复**：bug 注入要针对**具体功能单元**（乘法器、ALU、移位器等），让分歧在**对应的指令**（remu/mulhsu/srai 等）处发生，而不是在初始化就错。

### 🟠 差异 3：没做「消歧」

官方 QA A5 明确：**官方主动剔除了「不同 bug 但相同 syndrome」的 case**（避免 diff-bugs-same-syndrome）。

我们的 benchmark6 **没做这一步**：56 个 mismatch bug 症状雷同（都从 lui cascade），日志层面不可分。

**修复**：造数据后，**检查每个 bug 的症状是否可区分**（不同 bug 的第一个 mismatch 指令/测试名/分歧类型是否不同），剔除「不同 bug 相同 syndrome」的 case。

### 🟡 差异 4：bug 类型分布

官方分两类（见 `official_bug_distribution.md`）：

| 类别 | 症状 | 区分信号 | 官方例子 |
|---|---|---|---|
| 功能单元缺陷 | mismatch | 指令家族（MDU/ALU）| bug_107 = remu/mulhsu/rem |
| 测试断言失败 | test_fail（无 mismatch）| 测试名（CSR/中断/debug）| bug_7023/7021 = CSR/中断测试 |

**修复**：两类 bug 都要覆盖，且：
- mismatch 类：同 bug 不同 case 的**指令家族稳定**（都是 MDU 或都是 ALU）
- test_fail 类：同 bug 不同 case 的**测试域稳定**（都是 CSR 或都是中断）

### 🟢 差异 5：Ibex 版本

官方基于 commit `8ce399dbe678f0a66856ac302ec7609ba366d8fd`（QA A3），bug 注入是**这个 commit 之上的本地 RTL 修改**。

**修复**：如果队友用的 Ibex commit 不同，切换到官方这个 commit，避免版本漂移导致日志格式/指令集差异。

## 给队友的一句话总结

**核心问题**：我们的 bug 注入太「早期」了——bug 从第一条 `lui` 指令就开始 cascade 分歧，导致所有 mismatch bug 症状雷同（都从 lui 全错），不可区分。

**要修的是**：
1. bug 注入在**具体功能单元**（MDU/ALU/CSR）的逻辑里，让分歧在**对应指令**处才发生（不是初始化就错）
2. `mismatch_print_limit=1`，regr.log 只打印单个 Mismatch[1]
3. **消歧**：造完数据检查每 bug 症状可区分，剔除「不同 bug 相同 syndrome」
4. 用官方 commit `8ce399d...` 作为 baseline
