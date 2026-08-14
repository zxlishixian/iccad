# 官方 public 数据集的 bug 分布分析（2026-08-14）

> 分析 benchmark_set_1（7例/2bug）和 benchmark_set_2（25例/4bug）的实际 bug 症状，推断官方 bug 注入的类型分布。

## 官方 bug 分两大类

### 类别 1：mismatch 类（功能单元缺陷 → Ibex vs Spike 指令分歧）

| bug | 例数 | 分歧指令 | 特征 |
|---|---|---|---|
| bug_107 (set2) | 15 | remu/mulhsu/rem/sra/ori/srai/c.* | **MDU/ALU 家族**，同 bug 不同 case 指令家族稳定 |
| bug_234 (set1) | 3 | addi/sltu | ALU 家族 + debug test |
| bug_304 (set2) | 3 | c.addi | compressed + debug test |

**关键**：mismatch 类 bug 的**指令家族（MDU/ALU）是稳定信号**——同 bug 的不同 case 用同一功能单元的指令（bug_107 全是乘除/移位指令）。

### 类别 2：test_fail 类（测试断言失败 → 无 mismatch）

| bug | 例数 | 测试名 | 特征 |
|---|---|---|---|
| bug_7023 (set1) | 4 | interrupt_csr / multiple_interrupt / single_interrupt / csr | **CSR/中断测试** |
| bug_2014 (set2) | 3 | single_interrupt / mem_error / multiple_interrupt | 中断/内存错误测试 |
| bug_7021 (set2) | 4 | csr / single_interrupt / multiple_interrupt / interrupt_csr | **CSR/中断测试** |

**关键**：test_fail 类 bug 靠**测试名（功能域）区分**——同 bug 的不同 case 用同一功能域的测试（CSR/中断）。

## 核心结论（给队友）

官方 bug 注入是**两类**：

1. **功能单元缺陷**（MDU/ALU/CSR 逻辑改动）→ 产生 mismatch，症状 = **特定指令家族**（remu/mulhsu 等）
2. **测试断言失败**（CSR/interrupt/debug 功能改动）→ 产生 test_fail（无 mismatch），症状 = **特定测试名/功能域**

且官方做了**消歧**：同一 bug 的不同 case 症状**稳定可区分**（同指令家族 / 同测试域），不会像我们 fake benchmark6 那样「56 个 bug 都从 cycle 0 的 lui cascade 分歧、症状雷同」。

## 和 EnCorpus 的对应

EnCorpus 的两种 bug 类型正好对应：
- **Signal Mix-Ups（信号交换）** → 功能单元逻辑错（信号接错）→ 可能产生 mismatch 类症状
- **Broken Conditionals（条件破坏）** → 控制流/条件错 → 可能产生 test_fail 或 debug 类症状

但 EnCorpus 是 **Yosys 网表级注入**（`passes/inject/`），官方是 **RTL 源码级注入**（A7）。建议队友在 **RTL 源码级**做类似的「信号交换 / 条件破坏 / 逻辑改动 / 位宽错误」修改。

## 给队友的具体造数据清单

1. **RTL 源码级注入**（不是网表级），基于 commit `8ce399dbe678f0a66856ac302ec7609ba366d8fd`
2. **两类 bug 都要覆盖**：
   - 功能单元缺陷（MDU/ALU/CSR 的算术/逻辑/移位改动）→ mismatch 症状
   - 控制/中断/调试逻辑缺陷 → test_fail 症状
3. **消歧**：同 bug 的不同 case 症状稳定（同指令家族 / 同测试域），剔除「不同 bug 相同 syndrome」的 case
4. **mismatch_print_limit=1**：regr.log 只打印单个 Mismatch[1] 块
5. **bug 数量/k 分布**对标官方 hidden 表（§1.6）：k = 2/4/8/16/32/64，每 bug 多 case 多 seed
