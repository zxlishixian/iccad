# 官方数据集的 bug 注入方法 + 数据分布调研（2026-08-14）

> 目的：搞清楚官方 hidden 集的 bug 注入方法和数据分布，让队友的 fake 数据集从底层原理上更像官方。

## 官方明确披露的（来自 B_QA_20260727.pdf）

### 1. Ibex 版本（A3）
- 官方基于 upstream lowRISC Ibex 的这个 commit：
  ```
  git clone https://github.com/lowRISC/ibex.git
  git checkout 8ce399dbe678f0a66856ac302ec7609ba366d8fd
  ```
- **「the RTL was locally modified for bug injection on top of this commit」** —— bug 注入是在这个 commit 之上做的本地 RTL 修改。

### 2. bug 注入方式（A2 + A7）
- **A2：「There is no predefined fault taxonomy or fixed list of coarse subsystems」** —— 没有预定义的故障分类法，bug 是**任意的 RTL 修改**，不同 bug 可能产生跨粗粒度子系统的重叠症状。**不要假设固定故障模型。**
- **A7：「The bugs are injected into the RTL source code where the assembly trace is dumped from simulation」** —— bug 注入在 RTL 源码里（汇编 trace 从仿真 dump 出来的那个位置）。VCS 仿真器 + UVM 测试台是 oracle。

### 3. 数据分布的关键特点

**A. 官方「消歧」了（A5）—— 这是最关键的一条：**
> 「The same syndrome you see is actually coming from two different bugs... we have excluded all these conflicting cases from the benchmark sets (both public and private).」

即官方**主动剔除了「不同 bug 但相同 syndrome」的 case**。这对应赛题 QA 里的「renew benchmarks to avoid diff-bugs-same-syndrome」。

**⚠️ 我们 fake benchmark6 的问题正在于此**：56 个 mismatch bug 有相同的 syndrome（都从 cycle 0 的 lui 开始 cascade 分歧），日志层面不可分。**官方是「消歧」过的，所以每个 bug 的症状可区分。**

**B. mismatch 只打印 1 个（Q13）：**
> 「all officially provided non-UVM failure cases contain only a single Mismatch[1] block in regr.log」（upstream 默认 `mismatch_print_limit=5`）

**⚠️ 我们 fake benchmark6 的 regr.log 有 28004 个 mismatch（`Mismatch[1..28004]`），而官方只有单个 `Mismatch[1]` 块。** 这是 fake 和官方之间一个明显的格式/分布差异。

**C. 环境是黑盒（A13）：**
> 「we have made several changes over years... please treat it as a black box with limited hints... the data-driven clustering algorithm needs to be tolerant for some minor modifications.」

官方环境在 Ibex commit 之上做了多年修改（mismatch_print_limit、testlist 等），无法完全复现，算法要容忍这些差异。

## 给队友造数据的建议（让 fake 更像官方）

1. **用同一个 Ibex commit**（`8ce399dbe678f0a66856ac302ec7609ba366d8fd`）作为 baseline RTL，避免版本漂移。
2. **bug 注入做「任意的 RTL 修改」**，不要局限于固定子系统分类（信号交换、条件破坏、逻辑改动、位宽错误等都可）。
3. **消歧**：造数据时剔除「不同 bug 相同 syndrome」的 case（这是官方最关键的分布特征，也是我们 fake benchmark6 缺的）。
4. **mismatch 只打印 1 个**：把 `mismatch_print_limit` 调成 1，让 regr.log 只有单个 `Mismatch[1]` 块，而不是像 benchmark6 那样 cascade 出 28004 个 mismatch。

## 参考

- 赛题 QA：`information_files/QA/B_QA_20260727.pdf`（A2/A3/A5/A7/A13）
- Ibex 验证文档：https://ibex-core.readthedocs.io/en/latest/03_reference/verification.html
- bug 注入参考（RTL 修改方式）：EnCorpus 的 Signal Mix-Ups / Broken Conditionals（见 README.md）
