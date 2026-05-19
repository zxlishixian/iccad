# 公开 Benchmark 根因分析

本文档记录对公开数据集 `problem/benchmark_set_1` 和 `problem/benchmark_set_2` 的人工根因分析与自然子类划分。这里的 bucket 按日志证据和失效机制划分，不强行等同赛事表格里的参考 bucket 数；参考 bucket 数只作为 soft hint。

## 分析方法

输入目录：

```text
D:\ICCAD\drive-download-20260518T133804Z-3-001\problem
```

每个 case 使用三类证据：

```text
regr.log      regression 汇总、first mismatch、failed test name
sim.log.gz    UVM_FATAL/UVM_ERROR/assertion/failure status
trace.log.gz  retire trace，重点看 first mismatch 附近和 tail loop
```

判定优先级：

```text
1. sim.log 中明确的 UVM_FATAL / assertion / Check failed
2. regr.log 中的 first mismatch 位置和双方 PC/指令
3. trace 中 first mismatch 附近的控制流、特殊指令、CSR/LSU 行为
4. trace tail 是否进入固定 completion loop
```

## Benchmark 1 结论

benchmark1 共 9 个 case。折中采用 3 个自然子类：

```text
debug_or_ebreak_trap_entry_failure:
  case_1, case_2, case_3

irq_entry_timeout:
  case_4, case_5, case_7, case_9

xprop_exception_state_failure:
  case_6, case_8
```

### debug_or_ebreak_trap_entry_failure

覆盖 case：

```text
case_1, case_2, case_3
```

证据：

```text
case_1: riscv_debug_ebreak_test.0 failed
        UVM_FATAL Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period

case_2: riscv_debug_ebreakmu_test.0 failed
        UVM_FATAL Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period

case_3: regr.log 出现 Mismatch[1]
        trace 中 c.ebreak 后 Ibex 继续顺序执行
        Spike 已进入 trap handler 附近地址 0x80011080
```

判断：

这三个 case 都是 debug/ebreak/trap 入口相关问题。case_1 和 case_2 是显式 debug-mode entry timeout；case_3 是 ebreak 没有按预期进入 trap/debug 处理路径，表现为 Ibex 和 Spike 的 PC 流分叉。

### irq_entry_timeout

覆盖 case：

```text
case_4, case_5, case_7, case_9
```

证据：

```text
case_4: riscv_multiple_interrupt_test.0 failed
case_5: riscv_single_interrupt_test.0 failed
case_7: riscv_interrupt_csr_test.0 failed
case_9: riscv_single_interrupt_test.0 failed

共同 fatal:
UVM_FATAL Did not receive core_status HANDLING_IRQ within 750 cycle timeout period
```

判断：

这组是中断入口/中断状态转换失败。即使 case_7 的 test 名带 CSR，第一故障证据仍是 `HANDLING_IRQ` timeout，而不是 CSR assertion。

### xprop_exception_state_failure

覆盖 case：

```text
case_6, case_8
```

证据：

```text
case_6:
  riscv_csr_test.0 failed
  IbexBranchDecisionValid
  IbexDataOffsetKnown
  IbexWbStateKnown
  IbexCsrWdataIntKnown

case_8:
  riscv_multiple_interrupt_test.0 failed
  IbexDataOffsetKnown
  IbexCsrWdataIntKnown
  later fatal: mcause.exception_code is encoding the wrong exception type
```

判断：

这两个 case 都有 CSR/LSU/control-state 的 X 传播或未知态 assertion。case_8 后面表现为 `mcause` 错误，但更早的根因证据是 `data_offset` 和 `csr_wdata_int` unknown。

## Benchmark 1 输出建议

如果按自然子类提交，可使用：

```csv
Case,bucket
1,debug_or_ebreak_trap_entry_failure
2,debug_or_ebreak_trap_entry_failure
3,debug_or_ebreak_trap_entry_failure
4,irq_entry_timeout
5,irq_entry_timeout
6,xprop_exception_state_failure
7,irq_entry_timeout
8,xprop_exception_state_failure
9,irq_entry_timeout
```

## Benchmark 2 结论

benchmark2 共 27 个 case。自然子类划分为 6 类：

```text
branch_ge_condition_failure:
  case_1, case_2, case_3, case_4, case_5, case_6, case_7, case_8,
  case_9, case_10, case_11, case_12, case_13, case_14, case_15

debug_entry_timeout:
  case_16, case_17

memory_fault_status_mismatch:
  case_18

irq_entry_timeout:
  case_19, case_20

debug_dret_return_failure:
  case_21, case_22, case_23

xprop_exception_state_failure:
  case_24, case_25, case_26, case_27
```

### branch_ge_condition_failure

覆盖 case：

```text
case_1  riscv_debug_stress_test_0
case_2  riscv_dret_test_0
case_3  riscv_ebreak_test_0
case_4  riscv_hint_instr_test_0
case_5  riscv_invalid_csr_test_0
case_6  riscv_jump_stress_test_0
case_7  riscv_loop_test_0
case_8  riscv_machine_mode_rand_test_0
case_9  riscv_mmu_stress_test_0
case_10 riscv_perf_counter_test_0
case_11 riscv_rand_instr_test_0
case_12 riscv_rand_jump_test_0
case_13 riscv_rv32im_instr_test_0
case_14 riscv_unaligned_load_store_test_0
case_15 riscv_user_mode_rand_test_0
```

共同现象：

```text
regr.log 都是 Mismatch[1]
sim.log 没有 UVM_FATAL/UVM_ERROR/assertion
trace tail 多数进入固定 completion loop
first mismatch 附近均能看到应 taken 的 bge/bgeu 路径被 Ibex fall-through
```

典型证据：

```text
case_1:
  80001b34 bge x25,x12,80001b4e
  x25=0x000000c9, x12=0x00000051
  signed compare 应 taken，但 Ibex 继续执行 80001b38 mulhsu

case_2:
  80000c7a bge x7,x0,80000ca2
  x7=0x7ffdd1d0, x0=0
  应 taken，但 Ibex 继续执行 80000c7e srai

case_3:
  800009a6 bgeu x21,x22,800009b2
  x21=0xffffff61, x22=0x00000002
  unsigned compare 应 taken，但 Ibex 继续执行 800009aa c.xor

case_5/case_10:
  80000aac bge x27,x24,80000ab8
  x27=0x0000000b, x24=0xffffffbf
  signed compare 应 taken，但 Ibex 继续执行 80000ab2 remu

case_11/case_15:
  80000368 bge x7,x9,80000370
  x7=0x48e0f000, x9=0xffffffff
  signed compare 应 taken，但 Ibex 没有按 Spike 路径进入 80000370
```

判断：

这 15 个 case 虽然 test 名不同，first mismatch 指令也不同，但共同根因特征非常强：`bge` / `bgeu` 为 true 时，Ibex 没有跳到目标地址。更像是 branch greater/equal 条件 decode 或比较结果极性错误，而不是各自 first mismatch 指令本身的问题。

### debug_entry_timeout

覆盖 case：

```text
case_16, case_17
```

证据：

```text
case_16:
  riscv_debug_ebreak_test.0 failed
  UVM_FATAL Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period

case_17:
  riscv_debug_ebreakmu_test.0 failed
  UVM_FATAL Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period
```

判断：

这组和 benchmark1 的 `case_1/case_2` 同型，是 debug mode entry 没有发生或 core_status 没有按预期更新。

### memory_fault_status_mismatch

覆盖 case：

```text
case_18
```

证据：

```text
riscv_mem_error_test.0 failed
UVM_FATAL Check failed signature_data == core_status (10 [0xa] vs 9 [0x9])
Core did not register correct memory fault type
```

判断：

这是 memory fault 类型识别或状态编码错误。它不是普通 cosim mismatch，也不是 IRQ/debug timeout；fatal 已经明确指出 core 没登记正确的 memory fault type。

### irq_entry_timeout

覆盖 case：

```text
case_19, case_20
```

证据：

```text
case_19:
  riscv_multiple_interrupt_test.0 failed
  UVM_FATAL Did not receive core_status HANDLING_IRQ within 750 cycle timeout period

case_20:
  riscv_single_interrupt_test.0 failed
  UVM_FATAL Did not receive core_status HANDLING_IRQ within 750 cycle timeout period
```

判断：

这两个 case 是干净的 interrupt handling entry timeout，和 benchmark1 的 IRQ timeout 类一致。

### debug_dret_return_failure

覆盖 case：

```text
case_21, case_22, case_23
```

证据：

```text
case_21:
  riscv_debug_basic_test.0 failed
  UVM_FATAL No dret detected, or incorrect privilege mode switch in timeout period of 20000 cycles

case_22:
  riscv_debug_stress_test_0
  regr.log Mismatch[1]
  trace first mismatch 附近出现连续 dret:
    8000f8d8 dret
    8000f8dc dret
    8000f8e0 addi x5,x5,-124
  Spike 仍在正常/期望路径 800016da

case_23:
  riscv_dret_test_0
  regr.log Mismatch[1]
  trace first mismatch 前一条为:
    800005a4 dret
  Ibex 继续执行 800005a8/800005ae
  Spike 已到 80012080 附近处理路径
```

判断：

这组围绕 `dret`、debug return 和 privilege mode switch。case_21 是显式 fatal，case_22/case_23 虽然表现为 mismatch，但分叉点都紧邻 `dret`，不应和前面普通 branch mismatch 归为一类。

### xprop_exception_state_failure

覆盖 case：

```text
case_24, case_25, case_26, case_27
```

证据：

```text
case_24:
  riscv_csr_test.0 failed
  IbexBranchDecisionValid
  IbexDataOffsetKnown
  IbexWbStateKnown
  IbexCsrWdataIntKnown

case_25:
  riscv_interrupt_csr_test.0 failed
  IbexDataOffsetKnown
  IbexCsrWdataIntKnown

case_26:
  riscv_multiple_interrupt_test.0 failed
  IbexDataOffsetKnown
  IbexCsrWdataIntKnown
  later fatal: mcause.exception_code is encoding the wrong exception type

case_27:
  riscv_single_interrupt_test.0 failed
  IbexDataOffsetKnown
  IbexCsrWdataIntKnown
  later fatal: Did not receive core_status HANDLING_IRQ
```

判断：

case_24 到 case_27 都先出现 CSR/LSU 相关 unknown/assertion。case_26 和 case_27 后续分别表现为 `mcause` 错误和 IRQ timeout，但更早的共同证据是 `data_offset` / `csr_wdata_int` 的 X 传播。因此这里按根因优先级归入同一类，而不是按最后一个 fatal 文本拆分。

## Benchmark 2 输出建议

```csv
Case,bucket
1,branch_ge_condition_failure
2,branch_ge_condition_failure
3,branch_ge_condition_failure
4,branch_ge_condition_failure
5,branch_ge_condition_failure
6,branch_ge_condition_failure
7,branch_ge_condition_failure
8,branch_ge_condition_failure
9,branch_ge_condition_failure
10,branch_ge_condition_failure
11,branch_ge_condition_failure
12,branch_ge_condition_failure
13,branch_ge_condition_failure
14,branch_ge_condition_failure
15,branch_ge_condition_failure
16,debug_entry_timeout
17,debug_entry_timeout
18,memory_fault_status_mismatch
19,irq_entry_timeout
20,irq_entry_timeout
21,debug_dret_return_failure
22,debug_dret_return_failure
23,debug_dret_return_failure
24,xprop_exception_state_failure
25,xprop_exception_state_failure
26,xprop_exception_state_failure
27,xprop_exception_state_failure
```

## 关键注意点

### 不要只看 first mismatch 指令

benchmark2 的 `case_1` 到 `case_15` 里，first mismatch 指令表面上有 `mulhsu`、`srai`、`c.xor`、`remu`、`c.or`、`sra`、`ori` 等很多类型。如果只按 first mismatch instruction 聚类，会错误拆成 ALU/multdiv/shift/logic 多类。

真正共同点在 first mismatch 之前的 branch：

```text
bge/bgeu 条件应 taken
Ibex fall-through
Spike 进入 branch target
```

### 优先使用最早、最结构化的故障信号

例如 `case_27` 最后 fatal 是 `HANDLING_IRQ` timeout，但更早已有大量：

```text
IbexDataOffsetKnown
IbexCsrWdataIntKnown
```

这类 case 更适合归入 X-prop/state corruption，而不是纯 IRQ entry timeout。

### trace tail loop 只能作为辅助证据

许多 mismatch case 的 tail 都进入类似：

```text
auipc
sw
c.j
```

这说明程序已经进入 signature/completion loop，但不能单独作为 root cause。真正的分组依据仍然是 first divergence 附近的控制流或 sim assertion。

## 总结

benchmark1 推荐 3 个自然子类：

```text
debug_or_ebreak_trap_entry_failure
irq_entry_timeout
xprop_exception_state_failure
```

benchmark2 推荐 6 个自然子类：

```text
branch_ge_condition_failure
debug_entry_timeout
memory_fault_status_mismatch
irq_entry_timeout
debug_dret_return_failure
xprop_exception_state_failure
```

这套划分比简单按 test name 或 first mismatch 指令更稳，能解释同一根因在不同 directed/random tests 中产生的不同表面症状。
