# Benchmark8 500 Cases 数据集说明

## 1. 数据集目标

本批数据用于 Benchmark 5 规格下的回归失败聚类训练与验证，目标规模为 500 cases、10 个根因，每个根因 50 cases。数据采用官方目录布局与日志格式，并重点避免上一批数据中出现的启动阶段过早分叉、测试名与根因一一对应、日志模板泄漏等问题。

基线为 lowRISC Ibex 上游提交：

```text
8ce399dbe678f0a66856ac302ec7609ba366d8fd
```

仿真与参考模型工具链：

- Synopsys VCS 2023.12-SP2
- Spike ISS
- Ibex UVM / riscv-dv 回归框架

## 2. 根因清单

| Bucket | 根因 | 主要影响域 |
|---|---|---|
| `bug_041` | SRA 被错误执行为逻辑右移 | ALU / shift |
| `bug_048` | ADD/SUB 结果低字节发生条件位翻转 | ALU / arithmetic |
| `bug_051` | 执行一定数量 load 后，LB 符号扩展错误 | LSU / load sign extension |
| `bug_053` | 执行一定数量指令后，OP-IMM 结果发生条件位翻转 | ALU / immediate decode |
| `bug_066` | BGEU 错误采用有符号比较 | Branch comparator |
| `bug_083` | Branch immediate 错误使用 J-type 编码 | Branch target generation |
| `bug_085` | AUIPC 计算时忽略当前 PC | ALU / PC-relative operation |
| `bug_096` | IRQ 向量地址计算产生错误对齐 | Interrupt / trap vector |
| `bug_100` | 异常入口未正确清除 `mstatus.MIE` | CSR / exception state |
| `bug_102` | Store 地址立即数错误使用 I-type 编码 | LSU / store address generation |

这些根因不与上一批 500-case 数据集使用的 `bug_037/038/056/057/073/074/078/079/080/081` 重复。

## 3. 测试配额

| Bucket | Test 分布 |
|---|---|
| `bug_041` | arithmetic 13, machine-mode 13, rand-instr 11, RV32IM 7, user-mode 6 |
| `bug_048` | arithmetic 14, machine-mode 11, rand-instr 11, RV32IM 14 |
| `bug_051` | MMU stress 25, unaligned load/store 25 |
| `bug_053` | arithmetic 24, machine-mode 12, rand-instr 14 |
| `bug_066` | jump stress 11, machine-mode 14, rand-instr 13, rand-jump 12 |
| `bug_083` | jump stress 19, rand-jump 31 |
| `bug_085` | arithmetic 25, machine-mode 25 |
| `bug_096` | interrupt CSR 15, multiple interrupt 17, single interrupt 18 |
| `bug_100` | interrupt CSR 16, multiple interrupt 17, single interrupt 17 |
| `bug_102` | machine-mode 7, MMU stress 14, rand-instr 7, unaligned load/store 15, user-mode 7 |

共有 12 种测试。每个根因至少覆盖 2 种测试，每种入选测试至少覆盖 2 个根因，因此不存在 test name 到 bucket 的一一映射。

## 4. 关键质量指标

- 总 case 数：500
- Bucket 数：10，每个 50 cases
- 完整 `ibex[n]` / `spike[n]` 双端 mismatch：236 cases
- 双端 mismatch 中首个 Ibex index 均大于 32
- 首个 mismatch 最大 index：9054
- 失败类型同时包含 `REGR_MISMATCH`、`UVM_ERROR` 和 `UVM_FATAL`
- 日志中不包含 `bug_` 标签或生成机真实绝对路径
- 单日志最大行数低于赛事 10,000,000 行限制
- 官方布局：`input.csv`、`golden.csv`、`case_N/regr.log`、`case_N/sim.log.gz`、`case_N/trace.log.gz`

配额优化后 exact-test pair gap 从 0.2695 降至约 0.2641。该值未降到理想的 0.25 以下，主要原因是 LSU、branch 和 IRQ/CSR 根因的有效定向测试容量有限；但所有测试均有跨根因覆盖，pair-separability 审计未产生捷径警告。

## 5. 使用注意事项

1. `golden.csv` 仅用于训练或离线评估，正式提交输入应只使用 `input.csv` 和 case 日志。
2. 聚类特征应优先使用 DUT/ISS opcode pair、首个 mismatch 位置、PC region、UVM failure signature 和动态上下文，不应只记忆精确测试名。
3. 本批同时包含架构比较失败与 UVM/debug/interrupt 类失败，解析器需要兼容三种 primary type。
4. 发布包内的 `benchmark8_analysis` 保存选择报告、完整审计、格式验证和 pair-separability 结果，可用于复现实验前的数据验收。
