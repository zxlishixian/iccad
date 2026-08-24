# Benchmark 5 规格 500-case 数据集说明

## 1. 数据集规格

- 目标规格：最大 10,000,000 行、最大 1,000 cases、最多 32 buckets、单 case 目标运行时间不超过 100 s。
- 本数据集：500 cases、10 buckets、每个 bucket 50 cases。
- Ibex 基线：upstream commit `8ce399dbe678f0a66856ac302ec7609ba366d8fd`。
- 仿真与参考模型：Synopsys VCS 2023.12-SP2、Spike、RISC-V GCC、Ibex UVM/riscv-dv。
- 目录格式：官方 `input.csv`、`golden.csv`、`case_N/regr.log`、`case_N/sim.log.gz`、`case_N/trace.log.gz`；`meta.csv` 仅用于内部可追溯性。

## 2. 根因分布

| Bug | Group | Root cause | Cases | Tests |
|---|---|---|---:|---|
| bug_037 | debug_exception | debug entry priority decision is wrong | 50 | debug CSR entry, EBREAK, EBREAK M/U |
| bug_038 | csr_exception | MRET restores `mstatus.MIE` incorrectly | 50 | single/multiple interrupt |
| bug_056 | debug_exception | debug EBREAK CSR output is stuck low | 50 | debug branch/jump, EBREAK M/U |
| bug_057 | interrupt | external IRQ is reported with timer IRQ cause | 50 | single/multiple interrupt |
| bug_073 | multdiv | low multiply is decoded as high multiply | 50 | arithmetic, machine rand, rand instr, RV32IM |
| bug_074 | multdiv | `MULHSU` signedness mode is wrong | 50 | arithmetic, machine rand, rand instr, RV32IM |
| bug_078 | decoder_alu | XOR is decoded as OR | 50 | arithmetic, machine rand, rand instr, RV32IM |
| bug_079 | decoder_alu | AND is decoded as OR | 50 | arithmetic, machine rand, rand instr, RV32IM |
| bug_080 | decoder_alu | OR is decoded as XOR | 50 | arithmetic, machine rand, rand instr, RV32IM |
| bug_081 | decoder_alu | SLT signedness is swapped | 50 | arithmetic, machine rand, rand instr, RV32IM |

## 3. Test 配额

- `bug_037`: 17 CSR-entry + 16 EBREAK + 17 EBREAK-M/U.
- `bug_038`: 25 multiple-interrupt + 25 single-interrupt.
- `bug_056`: 20 debug-branch/jump + 30 EBREAK-M/U.
- `bug_057`: 5 multiple-interrupt + 45 single-interrupt. Multiple-interrupt 对该根因触发率较低，因此保留少量有效样本用于 test 多样性。
- `bug_073/074/078/079/080/081`: 每个 bug 分别为 13 arithmetic + 13 machine-rand + 12 rand-instr + 12 RV32IM。

## 4. 质量审计

- 500/500 cases 日志完整且存在真实 failure marker。
- 300 cases 为完整 `ibex[n]`/`spike[n]` 双端比较，`op_pair` 非空率为 100%（在这 300 个 case 内）。
- 200 cases 为 directed UVM failure；特征提取结果为 183 `UVM_FATAL`、17 `UVM_ERROR`。
- 300 个双端 case 的首个 mismatch 均晚于索引 31：251 个位于 32-255，49 个位于 256+，最大索引 1486。
- matched 数范围 42-6897；同时包含 `pc_same` 230 cases 与 `pc_diff` 70 cases。
- 500 个 sim 日志内容均唯一；307 个 regr 模板；500 个 trace 内容均唯一。
- 最大日志为 case 149 的 trace，共 499,812 行，低于 10M 上限。
- 无原始绝对路径、bug label 或 raw bucket 路径泄漏；无 Unicode replacement character；所有 gzip 可正常读取。
- 10-case 分层抽样：日志完整、failure marker、trace 头、路径清洗和内容唯一性全部通过。

## 5. Pair-separability

- `same_test` 正负 pair gap：0.2556。
- token Jaccard gap：0.3722。
- deterministic cosine gap：0.1802。
- same opcode gaps：Ibex 0.3439、Spike 0.3275。
- same `op_pair` gap：0.2767。
- 质量门禁结论：通过，无 hard warning。

仍有三个低风险提示：`riscv_debug_branch_jump_test`、`riscv_debug_csr_entry_test`、`riscv_debug_ebreak_test` 各只覆盖一个 bug。它们不是对应 bug 的唯一 test，且共享的 EBREAK-M/U、single-interrupt、multiple-interrupt 以及四种功能 test 已降低 test-name 捷径风险。

## 6. 复现与审计产物

- `selection_report.json`：精确 10 x 50 的选择结果。
- `validation.json`：官方目录和文件完整性验证。
- `final_audit.json`：case/bucket/test/failure/行数/泄漏审计。
- `final_feature_summary.json`：日志与 mismatch 特征统计。
- `pair_separability.json`：正负 pair 判别信号及捷径风险。
- `sample_audit_10.json`：每个 bucket 抽取一个 case 的详细审计。
