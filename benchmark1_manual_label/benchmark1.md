总体结论
benchmark_set_1 一共有 9 个失败 case。先划分为 5个 bucket：

Bucket	Cases	说明
debug_entry_timeout	case_1, case_2	debug sequence 后没有进入 IN_DEBUG_MODE
ebreak_exception_not_taken	case_3	c.ebreak 后 Ibex 顺序执行，Spike 已进入 trap handler
irq_handling_timeout	case_4, case_5, case_7, case_9	IRQ 已注入，但没有进入 HANDLING_IRQ
csr_wdata_xprop	case_6	CSR 测试中 csr_wdata_int 为 X，并伴随多种 X-prop assertion
lsu_data_offset_xprop	case_8	IRQ handler 路径中 data_offset 为 X，并伴随 csr_wdata_int 为 X

## benchmark_set_1

### case_1

- 建议 bucket：`debug_entry_timeout`，置信度：高
- 判断理由：等待 core_status=IN_DEBUG_MODE 超时；debug 入口相关。
- UVM test：`core_ibex_debug_ebreak_test`，seed：`7`
- sim 首个报错：`UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 4065000: uvm_test_top [uvm_test_top] Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period`
- regr 摘要：`riscv_debug_ebreak_test.0 : [FAILED]`
- sim 相关上下文：
  - `1: Command: /workspace/ibex/dv/uvm/out/rtl_sim/vcs_simv +vcs+lic+wait +signature_addr=0x8ffffffc -ucli -do /workspace/ibex/dv/uvm/vcs.tcl +ntb_random_seed=7 +require_signature_addr=1 +enable_debug_seq=1 +UVM_TESTNAME=core_ibex_debug_ebreak_test +bin=/workspace/ibex/dv/uvm/out/instr_gen/asm_tests/riscv_debug_ebreak_test_0.bin -l sim.log +UVM_VERBOSITY=UVM_MEDIUM`
  - `65: UVM_INFO /workspace/ibex/dv/uvm/tests/core_ibex_seq_lib.sv(39) @ 1965000: reporter@@debug_seq_single_h [debug_seq_single_h] Starting sequence...`
  - `66: UVM_INFO /workspace/ibex/dv/uvm/tests/core_ibex_seq_lib.sv(51) @ 2065000: reporter@@debug_seq_single_h [debug_seq_single_h] Exiting sequence`
  - `67: UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 4065000: uvm_test_top [uvm_test_top] Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period`
  - `94: [debug_seq_single_h] 2`
- trace 末尾观察：末尾 PC 有重复 800017cc, 800017ce, 800017d2, 800017d4；末尾指令序列: lui -> mulh -> c.lui -> blt -> mulhsu -> c.addi4spn -> andi -> c.addi -> sra -> xor

### case_2

- 建议 bucket：`debug_entry_timeout`，置信度：高
- 判断理由：等待 core_status=IN_DEBUG_MODE 超时；debug 入口相关。
- UVM test：`core_ibex_debug_ebreakmu_test`，seed：`7`
- sim 首个报错：`UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 4065000: uvm_test_top [uvm_test_top] Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period`
- regr 摘要：`riscv_debug_ebreakmu_test.0 : [FAILED]`
- sim 相关上下文：
  - `1: Command: /workspace/ibex/dv/uvm/out/rtl_sim/vcs_simv +vcs+lic+wait +signature_addr=0x8ffffffc -ucli -do /workspace/ibex/dv/uvm/vcs.tcl +ntb_random_seed=7 +require_signature_addr=1 +enable_debug_seq=1 +UVM_TESTNAME=core_ibex_debug_ebreakmu_test +bin=/workspace/ibex/dv/uvm/out/instr_gen/asm_tests/riscv_debug_ebreakmu_test_0.bin -l sim.log +UVM_VERBOSITY=UVM_MEDIUM`
  - `65: UVM_INFO /workspace/ibex/dv/uvm/tests/core_ibex_seq_lib.sv(39) @ 1965000: reporter@@debug_seq_single_h [debug_seq_single_h] Starting sequence...`
  - `66: UVM_INFO /workspace/ibex/dv/uvm/tests/core_ibex_seq_lib.sv(51) @ 2065000: reporter@@debug_seq_single_h [debug_seq_single_h] Exiting sequence`
  - `67: UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 4065000: uvm_test_top [uvm_test_top] Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period`
  - `94: [debug_seq_single_h] 2`
- trace 末尾观察：末尾指令序列: c.beqz -> lui -> jal -> jal -> c.jal -> c.j -> c.jal -> c.jal -> jal -> c.jal

### case_3

- 建议 bucket：`ebreak_exception_not_taken`，置信度：高
- 判断理由：riscv_ebreak_test 中 c.ebreak 后 Ibex 继续顺序执行，Spike 已进入 mtvec handler；说明 ebreak exception/trap 未按预期进入。
- UVM test：`core_ibex_base_test`，seed：`7`
- sim 首个报错：`UVM_ERROR : 0`
- regr 第一处 mismatch：Ibex `8000019c c.addi x29,6: t4:00000000`；Spike `ffffffff80011080 addi t0, t0, -124: t0:80021da0`；matched=64，mismatch=3458
- trace 中 Ibex mismatch PC 附近：722@8000019a:c.andi x13,14; 727@8000019c:c.addi x29,6; 728@8000019e:c.sub x11,x9; 734@800001a0:blt x29,x30,8000018e; 761@8000018e:or x8,x16,x24; 771@80000192:slti x24,x3,664; 772@80000196:c.ebreak; 791@80000198:c.addi x0,0
- trace 末尾观察：末尾 PC 有重复 8000f9ec, 8000f9f0, 8000f9f4；末尾指令序列: c.j -> auipc -> sw -> c.j -> auipc -> sw -> c.j -> auipc -> sw -> c.j
regr.log 第 1-4 行：

Mismatch[1]:
ibex[74] : pc[8000019c] c.addi x29,6: t4:00000000
spike[74] : pc[ffffffff80011080] addi t0, t0, -124: t0:80021da0
[FAILED]: 64 matched, 3458 mismatch
这说明第一个观测到的不一致是：

Ibex: 还在 0x8000019c 执行普通指令 c.addi
Spike: 已经到了 0x80011080 执行 trap handler 里的 addi
再看 trace.log.gz，前面有设置 trap vector：

80000134 addi x22,x22,-304   x22=0x80011000
80000138 ori  x22,x22,1      x22=0x80011001
8000013c csrrw x0,mtvec,x22
也就是 mtvec 被设置到 0x80011000 附近。

然后在 trace 第 85 行附近：

80000196 c.ebreak
80000198 c.addi x0,0
8000019a c.andi x13,14
8000019c c.addi x29,6
这里很关键：执行 c.ebreak 之后，Ibex 没有跳到 mtvec handler，而是继续执行了 80000198、8000019a、8000019c。

而 Spike 期望路径已经进入 handler：

80011080 addi x5,x5,-124
所以 regr.log 里才会看到：

Ibex pc = 8000019c
Spike pc = 80011080
另外，sim.log.gz 比较特殊，它没有报 UVM fatal，甚至显示：

--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
所以这个 case 不是 testbench 自己检测出的 fatal，而是Ibex trace 和 Spike ISS 对比失败。

结论：case3 应该不是简单的 compressed instruction mismatch，真正根因更像是：

ebreak exception/trap 没有被 Ibex 正确处理
### case_4

- 建议 bucket：`irq_handling_timeout`，置信度：高
- 判断理由：等待 core_status=HANDLING_IRQ 超时；中断响应/进入 handler 相关。
- UVM test：`core_ibex_debug_intr_basic_test`，seed：`7`
- sim 首个报错：`UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3965000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- regr 摘要：`riscv_multiple_interrupt_test.0 : [FAILED]`
- sim 相关上下文：
  - `68: UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3965000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- trace 末尾观察：末尾 PC 有重复 800001ae, 800001b0, 800001b4；末尾指令序列: divu -> blt -> sub -> add -> c.addi -> c.addi -> remu -> c.beqz -> c.srli -> jal
证据链如下。

regr.log 第 1 行：

riscv_multiple_interrupt_test.0 : [FAILED]
这个说明失败的是 multiple interrupt 测试。

关键在 sim.log.gz。

第 1 行显示这个测试打开了 multiple IRQ sequence：

+enable_irq_multiple_seq=1
+UVM_TESTNAME=core_ibex_debug_intr_basic_test
+bin=.../riscv_multiple_interrupt_test_0.bin
第 65-67 行显示 IRQ sequence 已经真的发起了：

65 ... irq_raise_seq_h ... Starting sequence...
66 ... irq_raise_seq_h ... Exiting sequence
67 ... irq: 0b11101110001100000000000010001000
然后第 68 行报 fatal：

UVM_FATAL ... Did not receive core_status HANDLING_IRQ within 750 cycle timeout period
所以 testbench 的判断是：

我已经 raise IRQ 了，但 750 cycle 内没有看到 core_status = HANDLING_IRQ
trace 也支持这个判断。IRQ 发起大约在 2463000 到 2467000 时间点附近，trace 附近是：

2393000  PC 800001a2  srl
2405000  PC 800001a6  auipc
2421000  PC 800001aa  or
2423000  PC 800001ae  c.addi
2497000  PC 800001b0  divu
2501000  PC 800001b4  blt
也就是说 IRQ 发起后，Ibex 仍在普通指令流里跑，没有立刻进入 interrupt handler。

再往后直到 fatal 前，它仍然在类似循环里执行：

8000019e ...
800001b4 blt ...
8000019e ...
800001b4 blt ...
所以 case4 的理解是：

multiple interrupt 被 testbench 注入后，Ibex 没有及时响应中断，没有进入 HANDLING_IRQ 状态。
这和 case1/case2 的 debug-entry timeout 不同；case4 是 IRQ/interrupt handling 路径的问题。

### case_5

- 建议 bucket：`irq_handling_timeout`，置信度：高
- 判断理由：等待 core_status=HANDLING_IRQ 超时；中断响应/进入 handler 相关。
- UVM test：`core_ibex_debug_intr_basic_test`，seed：`7`
- sim 首个报错：`UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3943000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- regr 摘要：`riscv_single_interrupt_test.0 : [FAILED]`
- sim 相关上下文：
  - `68: UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3943000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- trace 末尾观察：末尾 PC 有重复 800001aa, 800001ae, 800001b0, 800001b4；末尾指令序列: c.addi -> divu -> blt -> sub -> add -> c.addi -> c.addi -> remu -> c.beqz -> c.srli

regr.log 第 1 行：

riscv_single_interrupt_test.0 : [FAILED]
这说明 case5 是 single interrupt 测试失败。case4 是 multiple interrupt，case5 是 single interrupt。

关键仍然在 sim.log.gz：

第 1 行：

+enable_irq_single_seq=1
+UVM_TESTNAME=core_ibex_debug_intr_basic_test
+bin=.../riscv_single_interrupt_test_0.bin
第 65-67 行：

65 ... irq_single_seq_h ... Starting sequence...
66 ... irq_single_seq_h ... Exiting sequence
67 ... irq: 0b10000000000000000000
说明 testbench 已经发起了单个 IRQ。

然后第 68 行：

UVM_FATAL ... Did not receive core_status HANDLING_IRQ within 750 cycle timeout period
也就是说：

IRQ 已经 raise，但 750 cycle 内没有看到 core_status = HANDLING_IRQ
trace 也支持这个判断。IRQ 发起在大约 2441000 到 2445000 时间点，trace 附近还在普通循环中：

2377000  8000019e  or
2393000  800001a2  srl
2405000  800001a6  auipc
2421000  800001aa  or
2423000  800001ae  c.addi
2497000  800001b0  divu
2501000  800001b4  blt
没有看到进入 interrupt handler 的迹象。

和 case4 的关系：

case4: multiple interrupt
case5: single interrupt
但失败模式完全一致：

IRQ sequence 发起 -> 打印 irq bitmask -> 等 HANDLING_IRQ 超时
所以 case4 和 case5 可以放在同一个 bucket：

irq_handling_timeout
如果后续要更细分，也可以拆成：

irq_handling_timeout_multiple
irq_handling_timeout_single
但从“回归失败 bucketing”的角度，它们很可能是同一个根因：core 对外部 IRQ 没有及时进入中断处理状态。


### case_6

- 建议 bucket：`csr_wdata_xprop`，置信度：中
- 判断理由：首要断言是 branch_decision_i unknown，后续可能伴随 LSU/CSR X 传播。
- UVM test：`core_ibex_csr_test`，seed：`7`
- sim 首个报错：`UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 1321000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.id_stage_i.IbexBranchDecisionValid] IbexBranchDecisionValid: branch_in_dec |-> !$isunknown(branch_decision_i) (/workspace/ibex/rtl/ibex_id_stage.sv:636)`
- assertion 统计：IbexCsrWdataIntKnown=131, IbexBranchDecisionValid=18, IbexDataOffsetKnown=18, IbexWbStateKnown=18
- regr 摘要：`riscv_csr_test.0 : [FAILED]`
- sim 相关上下文：
  - `67: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 1321000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.id_stage_i.IbexBranchDecisionValid] IbexBranchDecisionValid: branch_in_dec |-> !$isunknown(branch_decision_i) (/workspace/ibex/rtl/ibex_id_stage.sv:636)`
  - `70: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 1321000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.load_store_unit_i.IbexDataOffsetKnown] IbexDataOffsetKnown: !$isunknown(data_offset) (/workspace/ibex/rtl/ibex_load_store_unit.sv:487)`
  - `73: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 1323000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.id_stage_i.IbexWbStateKnown] IbexWbStateKnown: !$isunknown(id_wb_fsm_cs) (/workspace/ibex/rtl/ibex_id_stage.sv:633)`
  - `76: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 1393000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.id_stage_i.IbexBranchDecisionValid] IbexBranchDecisionValid: branch_in_dec |-> !$isunknown(branch_decision_i) (/workspace/ibex/rtl/ibex_id_stage.sv:636)`
  - `79: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 1393000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.load_store_unit_i.IbexDataOffsetKnown] IbexDataOffsetKnown: !$isunknown(data_offset) (/workspace/ibex/rtl/ibex_load_store_unit.sv:487)`
- trace 末尾观察：末尾指令序列: addi -> bne -> csrrs -> addi -> bne -> addi -> slli -> addi -> lui -> addi

regr.log 第 1 行：

riscv_csr_test.0 : [FAILED]
说明这是 CSR 测试失败。

sim.log.gz 第 60-61 行：

Running test core_ibex_csr_test
Running test : .../riscv_csr_test_0.bin
真正的错误从第 65 行开始。

首个 assertion 是第 65-67 行：

IbexBranchDecisionValid: started at 13210000ps failed at 13210000ps
Offending '(!$isunknown(branch_decision_i))'

UVM_ERROR ... [ASSERT FAILED] ... IbexBranchDecisionValid:
branch_in_dec |-> !$isunknown(branch_decision_i)
也就是：

branch_decision_i 出现 X
但是紧接着同一时间还有：

IbexDataOffsetKnown
第 68-70 行：

IbexDataOffsetKnown: !$isunknown(data_offset)
然后第 71-73 行：

IbexWbStateKnown: !$isunknown(id_wb_fsm_cs)
更关键的是后面大量出现：

IbexCsrWdataIntKnown: !$isunknown(csr_wdata_int)
我统计了一下 assertion 数量：

IbexCsrWdataIntKnown      131 次
IbexBranchDecisionValid    18 次
IbexDataOffsetKnown        18 次
IbexWbStateKnown           18 次
所以虽然第一个报错是 IbexBranchDecisionValid，但整个 case 的主导特征其实是 CSR 写数据内部信号 csr_wdata_int 大量变成 X。

trace 末尾也支持这是 CSR 测试路径，最后大量在操作 mtvec：

80000928 csrrc  x2,mtvec,x7
8000093c csrrc  x2,mtvec,x7
80000954 csrrc  x2,mtvec,x7
80000960 csrrwi x2,mtvec,5
8000096c csrrwi x2,mtvec,26
80000978 csrrwi x2,mtvec,8
...
所以 case6 的理解是：

CSR 测试中，CSR 写路径/相关控制路径产生 X，导致 branch_decision、data_offset、WB


### case_7

- 建议 bucket：`irq_handling_timeout`，置信度：高
- 判断理由：等待 core_status=HANDLING_IRQ 超时；中断响应/进入 handler 相关。
- UVM test：`core_ibex_irq_csr_test`，seed：`7`
- sim 首个报错：`UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3453000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- regr 摘要：`riscv_interrupt_csr_test.0 : [FAILED]`
- sim 相关上下文：
  - `68: UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3453000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- trace 末尾观察：末尾 PC 有重复 800001b8, 800001bc, 800001c0, 800001c2；末尾指令序列: csrrc -> c.addi -> c.li -> blt -> or -> auipc -> slt -> csrrc -> c.addi -> c.li
regr.log 第 1 行：

riscv_interrupt_csr_test.0 : [FAILED]
看名字确实带 CSR，所以我们需要小心确认它是不是和 case6 一类。但 sim.log.gz 的实际 fatal 是 IRQ 超时。

sim.log.gz 第 1 行：

+enable_irq_single_seq=1
+UVM_TESTNAME=core_ibex_irq_csr_test
+bin=.../riscv_interrupt_csr_test_0.bin
说明这是一个 IRQ + CSR 相关测试，并且启用了 single IRQ sequence。

第 65-67 行：

65 ... irq_single_seq_h ... Starting sequence...
66 ... irq_single_seq_h ... Exiting sequence
67 ... irq: 0b10000000000000000000
testbench 已经注入了单个 IRQ。

第 68 行是核心报错：

UVM_FATAL ... Did not receive core_status HANDLING_IRQ within 750 cycle timeout period
也就是：

IRQ 已经 raise，但 750 cycle 内没有看到 core_status = HANDLING_IRQ
trace 里确实能看到它在执行一些 CSR 指令，例如末尾附近有：

800001bc csrrc x14,mscratch,x30
但这不是 assertion 报错。sim.log.gz 里没有像 case6 那样的：

IbexCsrWdataIntKnown
IbexBranchDecisionValid
IbexDataOffsetKnown
case7 的失败机制还是 testbench 等中断处理状态超时。

所以 case7 和 case5 更接近：

case5: riscv_single_interrupt_test，single IRQ 后没进 HANDLING_IRQ
case7: riscv_interrupt_csr_test，single IRQ + CSR 程序后没进 HANDLING_IRQ
我会把 case7 和 case4、case5 放同一 bucket


### case_8

- 建议 bucket：`xprop_data_offset`，置信度：中
- 判断理由：首要断言是 data_offset unknown，偏访存地址/LSU X 传播。
- UVM test：`core_ibex_debug_intr_basic_test`，seed：`7`
- sim 首个报错：`UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 4165000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.load_store_unit_i.IbexDataOffsetKnown] IbexDataOffsetKnown: !$isunknown(data_offset) (/workspace/ibex/rtl/ibex_load_store_unit.sv:487)`
- assertion 统计：IbexDataOffsetKnown=15, IbexCsrWdataIntKnown=15
- regr 摘要：`riscv_multiple_interrupt_test.0 : [FAILED]`
- sim 相关上下文：
  - `70: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 4165000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.load_store_unit_i.IbexDataOffsetKnown] IbexDataOffsetKnown: !$isunknown(data_offset) (/workspace/ibex/rtl/ibex_load_store_unit.sv:487)`
  - `73: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 4165000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.cs_registers_i.IbexCsrWdataIntKnown] IbexCsrWdataIntKnown: !$isunknown(csr_wdata_int) (/workspace/ibex/rtl/ibex_cs_registers.sv:1040)`
  - `76: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 4167000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.load_store_unit_i.IbexDataOffsetKnown] IbexDataOffsetKnown: !$isunknown(data_offset) (/workspace/ibex/rtl/ibex_load_store_unit.sv:487)`
  - `79: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 4167000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.cs_registers_i.IbexCsrWdataIntKnown] IbexCsrWdataIntKnown: !$isunknown(csr_wdata_int) (/workspace/ibex/rtl/ibex_cs_registers.sv:1040)`
  - `82: UVM_ERROR /workspace/ibex/shared/rtl/prim_assert.sv(21) @ 4169000: reporter [ASSERT FAILED] [core_ibex_tb_top.dut.u_ibex_core.load_store_unit_i.IbexDataOffsetKnown] IbexDataOffsetKnown: !$isunknown(data_offset) (/workspace/ibex/rtl/ibex_load_store_unit.sv:487)`
- trace 末尾观察：末尾指令序列: sw -> csrrs -> sw -> lui -> addi -> addi -> c.slli -> c.addi -> sw -> csrrs

我建议暂定：

lsu_data_offset_xprop
证据如下。

regr.log 第 1 行：

riscv_multiple_interrupt_test.0 : [FAILED]
它和 case4 一样是 multiple interrupt 测试，但失败机制不同。

sim.log.gz 第 65-67 行：

65 ... irq_raise_seq_h ... Starting sequence...
66 ... irq_raise_seq_h ... Exiting sequence
67 ... irq: 0b11101110001100000000000010001000
说明 IRQ 已经注入。

但 case8 没有报：

Did not receive core_status HANDLING_IRQ
而是从第 68 行开始报 assertion。

第一个 assertion 是：

68 ... IbexDataOffsetKnown: started at 41650000ps failed at 41650000ps
69     Offending '(!$isunknown(data_offset))'
70 UVM_ERROR ... [ASSERT FAILED] ... IbexDataOffsetKnown:
   !$isunknown(data_offset)
同一时间还伴随：

IbexCsrWdataIntKnown: !$isunknown(csr_wdata_int)
我统计了一下 assertion：

IbexDataOffsetKnown      15 次
IbexCsrWdataIntKnown     15 次
所以它不是 IRQ 没响应，而是 IRQ 之后某条执行路径里 LSU/CSR 信号出现 X。

trace 也支持它已经进入 handler，而不是停在普通循环里。末尾能看到跳进 0x80010452 附近，执行一串保存寄存器的 store：

8001202c jal x0,80010452
80010452 addi x5,x5,-124
80010456 sw x1,4(x5)
8001045a sw x2,8(x5)
...
800104ce sw x31,124(x5)
然后继续读写异常/中断相关 CSR：

800104e6 csrrs x22,mcause,x0
80010506 csrrs x22,mstatus,x0
80010522 csrrs x22,mcause,x0
这和 case4 完全不同：

case4: IRQ 后没有进入 HANDLING_IRQ，直接 timeout
case8: IRQ 后进入 handler/保存上下文，但 LSU data_offset 和 CSR wdata 出现 X
所以 case8 建议单独归类为：

lsu_data_offset_xprop
可以备注为：

multiple interrupt handler path 中 data_offset / csr

### case_9

- 建议 bucket：`irq_handling_timeout`，置信度：高
- 判断理由：等待 core_status=HANDLING_IRQ 超时；中断响应/进入 handler 相关。
- UVM test：`core_ibex_debug_intr_basic_test`，seed：`7`
- sim 首个报错：`UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3943000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- regr 摘要：`riscv_single_interrupt_test.0 : [FAILED]`
- sim 相关上下文：
  - `68: UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3943000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- trace 末尾观察：末尾 PC 有重复 800001aa, 800001ae, 800001b0, 800001b4；末尾指令序列: c.addi -> divu -> blt -> sub -> add -> c.addi -> c.addi -> remu -> c.beqz -> c.srli
regr.log 第 1 行：

riscv_single_interrupt_test.0 : [FAILED]
sim.log.gz 第 1 行显示：

+enable_irq_single_seq=1
+UVM_TESTNAME=core_ibex_debug_intr_basic_test
+bin=.../riscv_single_interrupt_test_0.bin
第 65-67 行显示 IRQ sequence 已经执行：

65 ... irq_single_seq_h ... Starting sequence...
66 ... irq_single_seq_h ... Exiting sequence
67 ... irq: 0b10000000000000000000
第 68 行是核心失败：

UVM_FATAL ... Did not receive core_status HANDLING_IRQ within 750 cycle timeout period
所以含义是：

single IRQ 已经注入，但 core 没有在 750 cycle 内进入 HANDLING_IRQ 状态。
trace 末尾也显示它仍然在普通指令流里循环：

8000019e or
800001a2 srl
800001a6 auipc
800001aa or
800001ae c.addi
800001b0 divu
800001b4 blt
...
没有看到进入 interrupt handler 的迹象。