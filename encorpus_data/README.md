# EnCorpus 数据源调研（2026-08-14）

> 目的：找「和本竞赛高度相似的公开数据集 / 可生成类似数据的项目」，用于扩充训练数据（尤其按 §1.5 战略原则做格式对齐/预训练）。

## EnCorpus / Encarsia（ETH Zurich COMSEC）

- 仓库：https://github.com/comsec-group/encarsia
- Zenodo：https://doi.org/10.5281/zenodo.14664723
- 论文：encarsia_sec25.pdf

### 内容

| 项 | 说明 |
|---|---|
| CPU | **Ibex、Rocket、BOOM**（含本竞赛的 Ibex）|
| bug 类型 | Signal Mix-Ups（`driver/`）、Broken Conditionals（`multiplexer/`），每 CPU 各约 1000 个 |
| 每个 bug 目录 | `host.v`（带 bug 的 RTL）、`fuzz.log`、`check_summary.log`（DETECTED/NOT DETECTED）、`verify.log`（JasperGold）、`yosys_verify.log`、`yosys_proof.S`、`prefilter/fuzz.log` |

### ⚠️ 关键结论：日志格式不同，不能直接当同分布训练数据

EnCorpus 的日志是 **fuzzing + 形式化验证日志**（fuzz.log / verify.log / check_summary.log），**不是**本竞赛的 UVM co-simulation 日志（sim.log / regr.log / trace.log）。

- 本竞赛：Ibex + Spike 黄金模型 co-simulation，sim.log（UVM 测试台输出）/ regr.log（mismatch 汇总）/ trace.log（指令 trace）
- EnCorpus：DifuzzRTL / Processorfuzz / Cascade fuzzer 的输出 + Yosys/JasperGold 形式化验证日志

**所以 EnCorpus 的价值在于「bug 注入方法论参考」和「RTL 层 bug 类型」，不是「可改造的日志数据集」。**

### 更匹配的源：Ibex 项目本身

本竞赛的日志来自 Ibex 的 UVM 测试台（`dv/uvm/core_ibex`），这正是生成 sim/regr/trace 的**原始管线**：

- 仓库：https://github.com/lowRISC/ibex
- 验证文档：https://ibex-core.readthedocs.io/en/latest/03_reference/verification.html
- nightly regression 报告（含失败详情）：https://ibex.reports.lowrisc.org/opentitan/latest/report.html
- 依赖：RISCV-DV（随机指令生成）+ Spike（黄金 ISS，`--enable-commitlog --enable-misaligned`）+ VCS 仿真器

**若能自己跑 Ibex UVM 测试台 + 注入 bug，就能生成和竞赛**完全同格式**的训练数据。** 这是扩充数据的最正路，但需要搭仿真环境（VCS 或 Verilator + RISCV-DV + Spike）。

## 下一步（待定）

1. 评估 Ibex 仿真环境搭建成本（VCS 是商业的，Verilator 是开源的，但 Ibex UVM 测试台主要用 VCS）。
2. 若搭不起来，退而求其次：用 EnCorpus 的 `host.v`（带 bug 的 RTL）理解 bug 类型，或看 Ibex nightly regression 的失败日志（虽是 summary 不是完整 sim/regr/trace）。
