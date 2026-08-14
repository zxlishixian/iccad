# 可能的 bug 类型全景（给队友造数据的指导）

> 目的：官方 hidden 集大概率**不只** benchmark_set_2 那两类（4 个 bug 太小，不能代表全集）。本文综合 Ibex RTL 结构 + RV32IMCB 指令集 + UVM 测试覆盖 + 官方 public 数据 + EnCorpus，**尽量全面地枚举**可能的 bug 类型，供队友造数据时覆盖。

## 官方测试覆盖的功能域（来自 Ibex 验证文档）

官方 Ibex 的 UVM 测试台覆盖：**RV32IMCB 指令 + 特权规范（CSR）+ 异常/中断 + Debug Mode + 内存访问/内存错误**。

- RV32**I** = 基础整数（算术/逻辑/移位/分支/访存/跳转）
- RV32**M** = 乘除
- RV32**C** = 压缩指令
- RV32**B** = 位操作（bitmanip）

## 两大类症状（已从官方 public 数据实锤）

| 类别 | 日志症状 | 区分信号 |
|---|---|---|
| **功能单元缺陷** | mismatch（Ibex vs Spike 指令/寄存器分歧）| 分歧指令的**功能单元家族** |
| **测试断言失败** | test_fail（无 mismatch，测试直接 FAILED）| **测试名/功能域** |

---

## A. mismatch 类（功能单元缺陷）

### A1. ALU 运算错误（RV32I 算术/逻辑/移位）
- 加法/减法进位错误、移位量错误、比较（slt/sltu）错误、逻辑运算（and/or/xor）错误
- **位宽/符号扩展错误**（截断、零扩展 vs 符号扩展）
- 症状：对应指令（add/addi/slt/srai 等）处 mismatch
- 官方对应：bug_234（addi/sltu）

### A2. MDU 乘除错误（RV32M）
- 乘法（mul/mulh/mulhsu）、除法（div/divu）、余数（rem/remu）错误
- 症状：对应指令处 mismatch
- 官方对应：bug_107（remu/mulhsu/rem）

### A3. 分支/跳转错误（控制流）
- 分支条件判断错误（beq/bne/blt）、跳转目标地址计算错误（jal/jalr）、PC 计算错误
- 症状：mismatch（PC 分歧，Ibex 和 Spike 执行不同指令）
- ⚠️ 我们的 fake benchmark6 那种「从 cycle 0 就 cascade」很可能就是这类（取指/PC 级错误）

### A4. 译码错误
- 指令译码错误（opcode/funct 判断错）、立即数扩展错误、寄存器读选择错误
- 症状：mismatch

### A5. 访存错误（LSU）
- 地址计算错误、读写选择错误、对齐错误、字节使能（byte enable）错误
- 症状：mismatch 或 mem_error

### A6. 压缩指令错误（RV32C）
- 压缩指令译码/扩展成 32 位指令的错误
- 症状：mismatch
- 官方对应：bug_304（c.addi）

### A7. 位操作错误（RV32B）
- 位操作指令（clz/ctz/ror/andn 等）错误
- 症状：mismatch

### A8. 流水线/微架构控制错误
- 数据转发（forwarding）错误、停顿（stall）错误、冲刷（flush）错误、冒险（hazard）处理错误
- 症状：mismatch（时序相关，可能只在特定指令序列才显形）

---

## B. test_fail 类（测试断言失败，无 mismatch）

### B1. CSR 错误（特权规范）
- CSR 读写错误、权限检查错误（用户态访问特权 CSR）、mstatus/mepc/mcause/mtval 更新错误
- 症状：test_fail（CSR 测试失败）
- 官方对应：bug_7023、bug_7021

### B2. 中断/异常错误
- 中断优先级错误、异常向量（mtvec）错误、进入/退出 handler 错误、中断使能/屏蔽错误
- 症状：test_fail（interrupt 测试失败，如 `Did not receive core_status HANDLING_IRQ`）
- 官方对应：bug_7023、bug_7021、bug_2014

### B3. 调试错误（Debug Mode）
- ebreak/单步（single step）/dret/调试寄存器（dscratch/dcsr）错误、进入 Debug Mode 错误
- 症状：test_fail（debug 测试失败）
- 官方对应：bug_304（debug_ebreak）

### B4. 内存错误（memory fault）
- 访问未映射地址、对齐错误、权限错误、访问错误类型（load/store 到非法地址）
- 症状：test_fail（mem_error 测试失败）
- 官方对应：bug_2014

---

## C. 通用 RTL 修改模式（EnCorpus 的参考，覆盖上面各类）

| 模式 | 含义 | 对应 |
|---|---|---|
| **Signal Mix-Ups**（信号交换）| 把两个信号接反 | A1~A8 的各种功能错误 |
| **Broken Conditionals**（条件破坏）| 把 if/条件判断改错 | A3（分支）、B2（中断条件）|

---

## 给队友的覆盖建议

1. **两类症状都要覆盖**：mismatch 类（A1~A8）和 test_fail 类（B1~B4），比例可参考官方（public 里两类都有）
2. **mismatch 类 bug 要注入到具体功能单元**（ALU/MDU/LSU/分支/译码），让分歧在**对应指令**处显形，不要都注入到取指/PC（那是我们 fake 的问题）
3. **test_fail 类 bug 要注入到 CSR/中断/调试/内存逻辑**，让对应测试 FAILED
4. **每个 bug 症状稳定可区分**（同 bug 的 case 指令家族/测试域一致），且**不同 bug 症状不同**（消歧）
5. **指令集覆盖**：RV32I（必）、RV32M（必，官方有 bug_107）、RV32C（必，官方有 bug_304）、RV32B（可选）
6. **mismatch_print_limit=1**（单 Mismatch 块）

## 参考

- 官方 bug 分布：`official_bug_distribution.md`
- 官方注入方法：`official_bug_injection.md`
- Ibex 验证文档：https://ibex-core.readthedocs.io/en/latest/03_reference/verification.html
- EnCorpus bug 注入（Signal Mix-Ups / Broken Conditionals）：https://github.com/comsec-group/encarsia
