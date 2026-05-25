# benchmark_set_2

### case_1

branch_condition_wrong或者更细：bge_not_taken_control_flow_mismatch

它表面上 regr.log 显示的是 mulhsu mismatch，但仔细看 trace 后，真正问题不是 mulhsu 计算错，而是前一条分支 bge 没有按预期跳转，导致 Ibex 和 Spike 走了不同控制流。

证据如下。

regr.log 第 1-4 行：
```text
Mismatch[1]:
ibex[425] : pc[80001b38] mulhsu x30,x6,x11: t5:00000000
spike[425] : pc[ffffffff80001b52] add s11, s3, t0: s11:00049c58
[FAILED]: 360 matched, 35687 mismatch
```
第一眼会以为是 mulhsu 问题。但看 trace：
```text
80001b34 bge x25,x12,80001b4e   x25:0x000000c9 x12:0x00000051
80001b38 mulhsu x30,x6,x11
```
这里 x25 = 0xc9，x12 = 0x51。

按 signed 比较：

0xc9 >= 0x51
应该成立，所以 bge 应该跳到：

80001b4e
Spike 的路径也确实已经到了后面：
```text
80001b52 add x27,x19,x5
```
但 Ibex 却没有跳转，而是继续顺序执行：
```text
80001b38 mulhsu x30,x6,x11
80001b3c c.mv x21,x11
```
...
所以 regr.log 里看到的 mulhsu 只是“分叉之后 Ibex 执行到的第一条不同指令”，不是根因本身。

sim.log.gz 也没有 UVM fatal/error：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
说明这是 ISS compare mismatch，不是 testbench assertion/fatal。

另外这个 case 是 debug stress 测试：
```text
+enable_debug_seq=1
+UVM_TESTNAME=core_ibex_debug_intr_basic_test
+bin=.../riscv_debug_stress_test_0.bin
```
trace 里前面有大量 dret，说明 debug stress 在频繁进出 debug，但第一处分叉点明确落在 bge 分支决策。

所以 case1 当前建议 bucket：
```text
branch_condition_wrong
```
或更细：
```text
bge_not_taken_control_flow_mismatch
```

### case_2

benchmark_set_2/case_2 -> branch_condition_wrong
更细叫：
```text
bge_not_taken_control_flow_mismatch
```
它和 benchmark_set_2/case_1 很像：regr.log 表面显示 Ibex 停在一条普通指令上，但真正根因是前一条 bge 分支应该跳转却没有跳。

关键证据如下。

regr.log：
```text
Mismatch[1]:
ibex[937] : pc[80000c7e] srai x21,x28,0x1: s5:ffffff32
spike[937] : pc[ffffffff80000ca2] c.mv s0, sp: s0:00000338
[FAILED]: 883 matched, 64810 mismatch
```
trace 中 mismatch 前的关键窗口：
```text
80000c76 bge x28,x23,80000c9a   x28:0xfffffe64 x23:0x00000001
80000c7a bge x7,x0,80000ca2     x7:0x7ffdd1d0 x0:0x00000000
80000c7e srai x21,x28,0x1       x28:0xfffffe64 x21=0xffffff32
80000c82 or x21,x1,x9
```
重点是这一行：
```text
80000c7a bge x7,x0,80000ca2（1264行）
```
此时：

x7 = 0x7ffdd1d0
x0 = 0
按 signed 比较，x7 >= x0 成立，所以 bge 应该跳到：

80000ca2
Spike 的路径也确实到了：
```text
80000ca2 c.mv s0, sp
```
但 Ibex 没有跳，而是继续顺序执行：
```text
80000c7e srai x21,x28,0x1
```
所以 srai 不是根因，只是 Ibex 走错控制流后执行到的第一条不同指令。

sim.log.gz 显示这是：
```text
core_ibex_dret_test
riscv_dret_test_0.bin
```
并且有很多：
```text
Illegal instruction ... 0x7b200073
```
0x7b200073 是 dret，在这个测试里这类 illegal instruction 记录本身不等于最终 fatal；真正失败仍然来自 ISS trace compare mismatch。

因此 case2 当前建议：
```text
case_2 -> branch_condition_wrong
```

### case_3
```text
branch_condition_wrong
```
更具体：
```text
bgeu_not_taken_control_flow_mismatch
```
证据如下。

regr.log：
```text
Mismatch[1]:
ibex[1194] : pc[800009aa] c.xor x8,x12: s0:800142d3
spike[1194] : pc[ffffffff800009b6] mulh a6, sp, a3: a6:00000040
[FAILED]: 678 matched, 1696 mismatch
```
表面看是 Ibex 在 c.xor，Spike 在 mulh，但 trace 里前一条关键指令是：
```text
800009a6 bgeu x21,x22,800009b2   x21:0xffffff61 x22:0x00000002
800009aa c.xor x8,x12
800009ac c.and x15,x9
800009ae divu x13,x28,x5
800009b2 fence
800009b6 mulh x16,x2,x13
```
bgeu 是无符号比较。此时：

x21 = 0xffffff61
x22 = 0x00000002
无符号下：

0xffffff61 >= 0x00000002
显然成立，所以应该跳到：

800009b2
然后继续到：
```text
800009b6 mulh
```
Spike 的路径正是在 800009b6。但 Ibex 没有跳，而是顺序执行：
```text
800009aa c.xor
```
所以 c.xor 只是错误控制流下的第一条不同指令，不是根因。

sim.log.gz 也显示 UVM 层没有 fatal/error：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
这个 case 是 riscv_ebreak_test_0.bin，但这里第一处分叉点不是 ebreak，而是 bgeu 分支决策错误。

### case_4
```text
branch_condition_wrong
```
更具体：
```text
bge_not_taken_control_flow_mismatch
```
关键证据：

regr.log：
```text
Mismatch[1]:
ibex[633] : pc[80000aa0] remu x26,x17,x10: s10:0000005d
spike[633] : pc[ffffffff80000aa6] div ra, zero, a1: ra:00000000
[FAILED]: 602 matched, 16795 mismatch
```
trace 关键窗口：
```text
80000a90 ori  x24,x5,-65        x24=0xffffffbf
80000a94 c.add x7,x15
80000a96 add  x15,x1,x12
80000a9a bge  x27,x24,80000aa6  x27:0x0000000b x24:0xffffffbf
80000a9e c.addi x0,0
80000aa0 remu x26,x17,x10
80000aa4 c.and x9,x13
80000aa6 div  x1,x0,x11
```
重点是这一条：
```text
80000a9a bge x27,x24,80000aa6
```
这是有符号比较：

x27 = 0x0000000b  = 11
x24 = 0xffffffbf  = -65
所以：

11 >= -65
条件成立，应该跳到：

80000aa6
Spike 正是在 80000aa6 执行：

div x1,x0,x11
但 Ibex 没跳，继续顺序执行：
```text
80000a9e c.addi
80000aa0 remu
```
所以 remu 不是根因，它只是 Ibex 错误控制流上的第一条 mismatch 指令。

sim.log.gz 也显示 UVM 层没有 fatal/error：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
所以 case4 和 case1/2/3 一样，都是 ISS compare 发现的控制流分叉。

### case_5
```text
branch_condition_wrong
```
更具体：
```text
bge_not_taken_control_flow_mismatch
```
证据如下。

regr.log：
```text
Mismatch[1]:
ibex[653] : pc[80000ab2] remu x26,x17,x10: s10:0000005d
spike[653] : pc[ffffffff80000ab8] div ra, zero, a1: ra:00000000
[FAILED]: 647 matched, 34055 mismatch
```
表面看是：

Ibex 执行 remu
Spike 执行 div
但真正分叉点在前面的 bge：
```text
80000aac bge x27,x24,80000ab8   x27:0x0000000b x24:0xffffffbf
80000ab0 c.addi x0,0
80000ab2 remu x26,x17,x10
80000ab6 c.and x9,x13
80000ab8 div x1,x0,x11
```
bge 是 signed compare：

x27 = 0x0000000b = 11
x24 = 0xffffffbf = -65
所以：

11 >= -65
条件成立，应该跳到：

80000ab8
Spike 也确实到了 80000ab8 执行 div。
但 Ibex 没跳，继续顺序执行了：

80000ab0
```text
80000ab2 remu
```
所以 remu 不是根因，只是错误控制流上的第一条不同指令。

sim.log.gz 也没有 UVM fatal/error：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
这个 case 的测试是：
```text
core_ibex_invalid_csr_test
riscv_invalid_csr_test_0.bin
```
虽然测试名不同于 case4，但第一处分叉机制相同：bge 条件成立时 Ibex fall-through。

所以 case5 建议：
```text
case_5 -> branch_condition_wrong
```
与前面四个case的共同现象：条件分支应该 taken，但 Ibex 没跳，导致后续 ISS mismatch

### case_6
```text
bgeu_not_taken_control_flow_mismatch
```
也归入大 bucket：
```text
branch_condition_wrong
```
证据如下。

regr.log：
```text
Mismatch[1]:
ibex[231] : pc[80000474] c.or x14,x8: a4:00ef9e4f
spike[231] : pc[ffffffff80000464] c.lui t3, 0x9: t3:00009000
[FAILED]: 377 matched, 11187 mismatch
```
表面看 Ibex 在 c.or，Spike 在 c.lui，但真正分叉点在前面的 bgeu：
```text
80000456 bgeu x29,x24,80000464   x29:0x00002324 x24:0x00000008
8000045a bne  x22,x9,80000470
80000470 bltu x15,x14,80000486
80000474 c.or x14,x8
```
bgeu 是无符号比较：

x29 = 0x00002324
x24 = 0x00000008
显然：

x29 >= x24
条件成立，所以应该跳到：

80000464
Spike 正是在 80000464 执行：

c.lui t3,0x9
但 Ibex 没有跳，而是继续往后走，最终到了：
```text
80000474 c.or x14,x8
```
所以 c.or 不是根因，根因还是分支 taken 失败。

sim.log.gz 也没有 UVM 层错误：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
测试是：
```text
riscv_jump_stress_test_0.bin
```
这个名字也和结论吻合：jump/branch stress 下出现控制流错误。

当前结论：
```text
case_6 -> branch_condition_wrong
```
细分标签：
```text
bgeu_not_taken_control_flow_mismatch
```

### case_7
```text
branch_condition_wrong
```
更具体：
```text
bge_not_taken_control_flow_mismatch
```
证据如下。

regr.log：
```text
Mismatch[1]:
ibex[186] : pc[800002c4] rem x30,x5,x4: t5:8001de1c
spike[186] : pc[ffffffff800002be] c.addi tp, -2: tp:fffffffe
[FAILED]: 228 matched, 7623 mismatch
```
表面看 Ibex 在 rem，Spike 在 c.addi。但看 trace，关键窗口是：
```text
800002b4 auipc x8,0x91abc
800002b8 c.addi x0,0
800002ba sltu x31,x19,x5
800002be c.addi x4,-2
800002c0 bge x4,x24,800002b4   x4:0x00000000 x24:0xfffffff8
800002c4 rem x30,x5,x4
```
注意 bge 是 signed compare：

x4  = 0x00000000 = 0
x24 = 0xfffffff8 = -8
所以：

0 >= -8
条件成立，应该跳回：

800002b4
这个是一个 loop。Spike 还在 loop body 内的：
```text
800002be c.addi x4,-2
```
但 Ibex 没有跳回，而是 fall-through 到：
```text
800002c4 rem x30,x5,x4
```
所以 rem 不是根因，真正根因是 bge 分支未按条件跳转，loop 提前退出/控制流跑偏。

sim.log.gz 同样没有 UVM fatal/error：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
测试是：
```text
riscv_loop_test_0.bin
```
这也和结论吻合：loop 测试中分支回跳失败。

结论：
```text
case_7 -> branch_condition_wrong
```
细分：

bge loop-back branch should be taken but Ibex fell throug

### case_8

还是同一大类：
```text
branch_condition_wrong
```
细分是：
```text
bgeu_not_taken_control_flow_mismatch
```
证据如下。

regr.log：
```text
Mismatch[1]:
ibex[79] : pc[8000021e] c.slli x20,0x4: s4:4b94f690
spike[79] : pc[ffffffff80000240] c.addi s9, 6: s9:00002006
[FAILED]: 77 matched, 4262 mismatch
```
trace 关键窗口：
```text
80000216 slt  x13,x22,x22
8000021a bgeu x12,x28,8000023c   x12:0xfcb0de05 x28:0x00000000
8000021e c.slli x20,0x4
```
...
```text
8000023c xor x13,x22,x24
80000240 c.addi x25,6
```
bgeu 是无符号比较：

x12 = 0xfcb0de05
x28 = 0x00000000
无符号下：

0xfcb0de05 >= 0
必然成立，所以应该跳到：

8000023c
然后继续到：
```text
80000240 c.addi
```
Spike 正是在 80000240。
但 Ibex 没有跳，继续顺序执行：
```text
8000021e c.slli
```
所以 c.slli 不是根因，只是分支 fall-through 后的第一条不同指令。

sim.log.gz 没有 UVM fatal/error：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
测试是：
```text
riscv_machine_mode_rand_test_0.bin
```
结论：
```text
case_8 -> branch_condition_wrong
```

### case_9

branch_condition_wrong。

关键证据在 benchmark_set_2/case_9/regr.log 和 trace.log.gz：
```text
Mismatch[1]:
ibex[31157]  : pc[8002c47c] ori x16,x3,903
spike[31157] : pc[ffffffff8002c484] add s10,s8,a5
```
也就是说分歧点是：Spike 已经跳到了 8002c484，但 Ibex 还在顺序执行 8002c47c。

对应 trace：
```text
8002c478 bge x10,x24,8002c484
```
x10 = 0x00007c2e
x24 = 0xb38dc825
```text
8002c47c ori x16,x3,903
8002c480 add x17,x26,x26
8002c484 add x26,x24,x15
```
这里是 bge，也就是 signed compare。
按有符号数解释：

x10 = 0x00007c2e = 正数 31790
x24 = 0xb38dc825 = 负数
所以条件 x10 >= x24 一定成立，正确行为应该跳转到 8002c484。Spike 的行为是正确的；Ibex 没跳，继续执行了 8002c47c，因此失败原因仍然是分支条件/分支决策错误。

另外 sim.log.gz 里 UVM 本身是通过的：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
所以这不是 timeout、exception、interrupt 或 testbench fatal，而是标准的 ISS mismatch。

结论：
```text
benchmark_set_2,case_9,branch_condition_wrong
```

### case_10

case10 结论：还是归到 branch_condition_wrong，而且它和 benchmark2 的 case5 非常像，几乎是同一个分支失败模式。

首个 mismatch 在 regr.log：
```text
Mismatch[1]:
ibex[653]  : pc[80000ab2] remu x26,x17,x10
spike[653] : pc[ffffffff80000ab8] div ra,zero,a1
```
也就是 Spike 已经到了 80000ab8，Ibex 却还在执行 80000ab2。看 trace 的关键位置：
```text
80000aac  bge x27,x24,80000ab8   x27:0x0000000b x24:0xffffffbf
80000ab0  c.addi x0,0
80000ab2  remu x26,x17,x10
80000ab6  c.and x9,x13
80000ab8  div x1,x0,x11
```
这里是 bge，有符号比较：

x27 = 0x0000000b = 11
x24 = 0xffffffbf = -65
所以 x27 >= x24 成立，正确行为应该跳转到 80000ab8。Spike 的行为正确；Ibex 没跳，继续执行了 80000ab0 / 80000ab2，于是和 Spike 分叉。

sim.log.gz 也显示 UVM 本身通过，没有 timeout/fatal：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
所以 case10 不是仿真环境失败，而是分支执行结果和 ISS 不一致。

最终分类：
```text
benchmark_set_2,case_10,branch_condition_wrong
```

### case_11

case11 结论：继续归到 branch_condition_wrong。这次分叉点稍微有意思：Ibex 是前一条 bge 没有正确跳转，随后又被下一条 beq 带到了另一个位置。

regr.log 的首个 mismatch：
```text
Mismatch[1]:
ibex[167]  : pc[80000378] mulhsu x10,x13,x11
spike[167] : pc[ffffffff80000370] ori s7,zero,190
```
也就是说正确路径应该在 80000370，但 Ibex 跑到了 80000378。

trace 关键段：
```text
80000368  bge x7,x9,80000370    x7:0x48e0f000  x9:0xffffffff
8000036c  beq x10,x31,80000378  x10:0x00000000 x31:0x00000000
80000378  mulhsu x10,x13,x11
```
这里第一条是 bge，有符号比较：

x7 = 0x48e0f000 = 正数
x9 = 0xffffffff = -1
所以 x7 >= x9 为真，应该跳到 80000370。Spike 正是这么做的，所以它停在：
```text
80000370  ori s7,zero,190
```
但 Ibex 没有执行这个跳转，而是错误地 fall-through 到 8000036c。接着 beq x10,x31,80000378 因为 0 == 0 成立，又跳到了 80000378，于是和 Spike 产生首个 mismatch。

sim.log.gz 没有 UVM fatal/error：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
所以 case11 仍然是 ISS mismatch，根因是分支判断错误。

最终分类：
```text
benchmark_set_2,case_11,branch_condition_wrong
```

### case_12

仍然归到 branch_condition_wrong，根因是 bge 条件成立但 Ibex 没有跳转。

regr.log 首个 mismatch：
```text
Mismatch[1]:
ibex[151]  : pc[800109e8] sra x22,x17,x13
spike[151] : pc[ffffffff80010a1c] c.addi16sp sp,400
```
也就是说 Spike 已经走到了 80010a1c，而 Ibex 还停在 fall-through 路径上的 800109e8。

关键 trace：
```text
800109dc  or   x31,x18,x26
800109e0  bge  x17,x6,80010a18   x17:0x00000001 x6:0x00000000
800109e4  bgeu x8,x27,80010a26   x8:0xd70c6000 x27:0xffffffce
800109e8  sra  x22,x17,x13
```
重点是这一行：
```text
800109e0  bge x17,x6,80010a18
```
bge 是有符号比较：

x17 = 0x00000001 = 1
x6  = 0x00000000 = 0
所以 x17 >= x6 明显成立，正确行为应该跳到 80010a18。Spike 走了跳转后的路径，所以后面出现在 80010a1c；Ibex 没跳，继续执行 800109e4、800109e8，于是产生 mismatch。

sim.log.gz 也显示 UVM 本身通过：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
所以 case12 不是 testbench timeout/fatal，而是 ISS 对比失败。分类：
```text
benchmark_set_2,case_12,branch_condition_wrong
```

### case_13

case13 仔细看下来，结论还是：branch_condition_wrong。这不是新的 bucket，仍然是 bge 有符号分支条件成立，但 Ibex 没有跳转。

regr.log 首个 mismatch 是：
```text
Mismatch[1]:
ibex[727]  : pc[80000b98] remu x28,x20,x30: t3:00000fff
spike[727] : pc[ffffffff80000bd0] andi t3,s6,291: t3:00000021
```
也就是说同一条 retire 序号上，Spike 认为下一条应该在 80000bd0，而 Ibex 实际还在 80000b98。这通常意味着前面有一条控制流指令，Spike 跳了，Ibex 没跳。

关键 trace 在这里：
```text
80000b68  add  x10,x7,x17       x7:0x00000000 x17:0x80000936 x10=0x80000936
80000b6c  add  x9,x30,x29       x30:0x0000001c x29:0x0000009e x9=0x000000ba
80000b70  srli x26,x21,0x5      x21:0x4c366000 x26=0x0261b300
80000b74  rem  x24,x13,x16      x13:0x00000092 x16:0xffffffda x24=0x00000020
80000b78  sltu x2,x5,x10        x5:0x80023e1c x10:0x80000936 x2=0x00000000
80000b7c  beq  x0,x12,80000b80  x0:0x00000000 x12:0x00000000
80000b80  sltu x24,x26,x27      x26:0x0261b300 x27:0x000003c3 x24=0x00000000
80000b84  andi x18,x14,863      x14:0x0000001e x18=0x0000001e
80000b88  srai x30,x10,0x13     x10:0x80000936 x30=0xfffff000
80000b8c  slt  x11,x17,x17      x17:0x80000936 x17:0x80000936 x11=0x00000000
80000b90  remu x12,x14,x6       x14:0x0000001e x6:0x00000000 x12=0x0000001e
80000b94  bge  x7,x10,80000bd0  x7:0x00000000 x10:0x80000936
80000b98  remu x28,x20,x30      x20:0xffffffff x30:0xfffff000 x28=0x00000fff
```
真正的分叉点是：
```text
80000b94  bge x7,x10,80000bd0
```
这里一定要按 有符号数 理解，因为是 bge，不是 bgeu。

x7  = 0x00000000 = 0
x10 = 0x80000936 = signed 负数
所以判断条件是：

0 >= 负数
这个条件成立。因此正确行为应该跳到：

80000bd0
Spike 正是这么走的，所以 regr.log 里 Spike 的 PC 是：

pc[ffffffff80000bd0] andi t3,s6,291
但 Ibex 没跳，继续顺序执行了 fall-through 路径：
```text
80000b98 remu x28,x20,x30
```
因此首个 mismatch 出现在 80000b98 vs 80000bd0。

这里我还检查了 sim.log.gz，UVM 没有报 fatal/error：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
测试类型是：
```text
core_ibex_base_test
riscv_rv32im_instr_test_0.bin
```
所以它不是 timeout、exception、UVM failure，也不是压缩指令或 jump 本身的问题。case13 的根因非常清楚：bge 分支判断错误，Ibex 对一个应该 taken 的 signed branch 执行成了 not taken。

最终分类：
```text
benchmark_set_2,case_13,branch_condition_wrong
```

### case_14

也就是大类仍然是 branch_condition_wrong，细分 bucket 我建议叫：
```text
bgeu_unsigned_taken_as_not_taken
```
含义：bgeu 是无符号大于等于分支，条件本应成立、应该跳转，但 Ibex 没有跳。

关键 mismatch 在 regr.log：
```text
Mismatch[1]:
ibex[11266]  : pc[80011bf6] c.li x7,-1
spike[11266] : pc[ffffffff80011c28] srl s7,s1,a3
```
所以 Spike 已经跳到了 80011c28，但 Ibex 还在顺序执行 80011bf6。

对应 trace 的分叉点是：
```text
80011bee  fence ...
80011bf2  bgeu x1,x27,80011c28   x1:0xffff9293 x27:0x800232b3
80011bf6  c.li x7,-1
80011bf8  c.sub x15,x9
80011bfa  divu x30,x21,x5
```
重点是这条：
```text
80011bf2  bgeu x1,x27,80011c28
```
bgeu 是 unsigned compare。所以这里不能按有符号数看，要直接比较 32-bit 无符号值：

x1  = 0xffff9293
x27 = 0x800232b3
无符号下：

0xffff9293 > 0x800232b3
因此 x1 >= x27 成立，正确行为应该跳到 80011c28。Spike 的 PC 正是 80011c28，说明 Spike 走了正确路径。Ibex 却继续执行了 fall-through 的：
```text
80011bf6  c.li x7,-1
```
所以这是非常明确的：bgeu 条件成立但 Ibex 没跳。

sim.log.gz 也显示仿真本身通过：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
测试是：
```text
riscv_unaligned_load_store_test_0.bin
```
虽然测试名字和 unaligned load/store 有关，但首个可解释分歧不是 load/store 数据错，而是控制流在 bgeu 这里走错。
最终建议记录为：

顶层 bucket: branch_condition_wrong
细分 bucket: bgeu_unsigned_taken_as_not_taken

### case_15

case15 结论：
```text
benchmark_set_2,case_15,branch_condition_wrong,bge_signed_taken_as_not_taken
```
它和 case11 是同一个细分模式，几乎是同一段分叉路径。

regr.log 首个 mismatch：
```text
Mismatch[1]:
ibex[167]  : pc[80000378] mulhsu x10,x13,x11
spike[167] : pc[ffffffff80000370] ori s7,zero,190
```
也就是说正确路径应该到 80000370，但 Ibex 实际去了 80000378。

关键 trace：
```text
80000364  div  x31,x24,x7       x24:0x2121e18c x7:0x48e0f000 x31=0x00000000
80000368  bge  x7,x9,80000370   x7:0x48e0f000  x9:0xffffffff
8000036c  beq  x10,x31,80000378 x10:0x00000000 x31:0x00000000
80000378  mulhsu x10,x13,x11
```
真正的错误点是：
```text
80000368  bge x7,x9,80000370
```
bge 是 signed compare：

x7 = 0x48e0f000 = 正数
x9 = 0xffffffff = -1
所以：

x7 >= x9
条件成立，应该跳到 80000370。Spike 正确跳到了 80000370，而 Ibex 没有跳，继续执行下一条：
```text
8000036c  beq x10,x31,80000378
```
因为 x10 == x31 == 0，Ibex 又被这条 beq 带到了 80000378，最终和 Spike 分叉。

sim.log.gz 没有 UVM 错误：
```text
--- RISC-V UVM TEST PASSED ---
UVM_ERROR : 0
UVM_FATAL : 0
```
测试是：
```text
riscv_user_mode_rand_test_0.bin
```
所以 case15 不是 user mode/exception 本身失败，而是 ISS 对比发现控制流错误。细分 bucket 推荐：
```text
bge_signed_taken_as_not_taken
```

### case_16

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
```text
benchmark_set_2,case_16,debug_entry_timeout,debug_req_not_enter_debug_mode
```
证据主要在 sim.log.gz，而不是 regr.log。regr.log 只有：
```text
riscv_debug_ebreak_test.0 : [FAILED]
```
没有 ISS mismatch 的 PC 对比。真正的失败信息是：

UVM_TESTNAME=core_ibex_debug_ebreak_test
```text
+enable_debug_seq=1
riscv_debug_ebreak_test_0.bin
```
然后：

UVM_INFO ... debug_seq_single_h [debug_seq_single_h] Starting sequence...
UVM_INFO ... debug_seq_single_h [debug_seq_single_h] Exiting sequence
UVM_FATAL ... Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period
这说明 testbench 发起了 debug sequence，等待 core 进入 IN_DEBUG_MODE，但 1000 cycle 内没有收到这个状态，于是 UVM fatal。这个和 benchmark1 的 case1/case2 是同一种 debug entry timeout 模式。

trace 也能配合说明：程序还在正常执行普通指令流，没有出现进入 debug ROM/handler 的明显控制流切换。比如 debug sequence 开始/退出大概在 1.965us/2.065us，而 trace 里这段之后仍在普通测试代码附近继续执行：
```text
80000188  jalr x2,118(x26)
800016d2  div x26,x19,x26
800016d6  bgeu x18,x29,800016da
```
...
800017xx  普通 load/store/ALU 指令流
所以根因不是某条 branch 比较错，而是外部 debug request/ebreak debug 测试期望 core 进入 debug mode，但 core 没有在规定时间内进入。

最终我建议：

顶层 bucket: debug_entry_timeout
细分 bucket: debug_req_not_enter_debug_mode

### case_17

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

建议分类：
```text
benchmark_set_2,case_17,debug_entry_timeout,debug_req_not_enter_debug_mode
```
regr.log 只有测试失败：
```text
riscv_debug_ebreakmu_test.0 : [FAILED]
```
关键失败在 sim.log.gz：

UVM_TESTNAME=core_ibex_debug_ebreakmu_test
```text
+enable_debug_seq=1
```
然后 testbench 发起 debug sequence：
```text
debug_seq_single_h ... Starting sequence...
debug_seq_single_h ... Exiting sequence
```
随后 fatal：
```text
Did not receive core_status IN_DEBUG_MODE within 1000 cycle timeout period
```
这说明它等待 core 进入 IN_DEBUG_MODE，但 1000 cycle 内没有等到。这个和 case16 完全同型，只是测试名从 riscv_debug_ebreak_test 变成了 riscv_debug_ebreakmu_test。

trace 里也能看到 debug sequence 之后，core 还在正常执行普通代码流，没有进入 debug mode。例如 sequence 时间大约在 1965000 到 2065000，trace 对应附近仍然是普通指令：
```text
8000016c addi x10,x0,-9
80000170 mulhu x22,x11,x13
80000174 c.addi x0,0
80000176 andi x27,x5,-24
```
...
```text
800001b6 blt x29,x30,800001a0
```
所以 case17 的根因不是某条分支比较错，而是 debug 请求/ebreakmu debug 测试没有让 core 按预期进入 debug mode。

最终建议：

顶层 bucket: debug_entry_timeout
细分 bucket: debug_req_not_enter_debug_mode

### case_18

建议分类：
```text
benchmark_set_2,case_18,mem_fault_type_mismatch,imem_fault_reported_wrong_core_status
```
regr.log 只有测试失败：
```text
riscv_mem_error_test.0 : [FAILED]
```
真正的失败在 sim.log.gz：

UVM_TESTNAME=core_ibex_mem_error_test
```text
riscv_mem_error_test_0.bin
```
先注入 dmem error：
```text
Injected dmem error
```
exiting mem fault checker
然后开始注入 imem fault：
```text
Injecting imem fault
Injecting imem fault
Injecting imem fault
latched_imem_err: 0x1
```
最后 fatal：
```text
Check failed signature_data == core_status (10 [0xa] vs 9 [0x9])
Core did not register correct memory fault type
```
这句话非常关键。testbench 期望 core 写出的状态码是 10 / 0xa，但实际 core_status 是 9 / 0x9。也就是说 core 捕捉到了某种 fault，但把 memory fault 类型登记错了。

trace 能看到 fault handler 的行为：先进入 trap handler，并读取 mcause。例如第一次 dmem fault 附近：
```text
80000206  lw x0,-12(x2)
80012000  jal x0,80012080
```
...
```text
800126ca  csrrs x22,mcause,x0   x22=0x00000005
```
mcause=5 对应 load access fault，这和 dmem/load fault 是合理的。后面 imem fault 被注入后，进入 handler：
```text
80000322  auipc x30,0x16
80012000  jal x0,80012080
```
...
```text
80012114  csrrs x22,mepc,x0     x22=0x8000032a
80012118  csrrs x22,mcause,x0   x22=0x00000002
```
这里 mcause=2 是 illegal instruction，而不是 instruction access fault。结合 sim log 的 latched_imem_err: 0x1 和最终状态码 0xa vs 0x9，更像是 imem fault 已经被 testbench latch 到，但 core 最后登记/上报的 fault 类型不符合预期。

所以 case18 的根因不是指令结果 mismatch，而是 memory fault 分类/状态码错误。建议命名：

顶层 bucket: mem_fault_type_mismatch
细分 bucket: imem_fault_reported_wrong_core_status

### case_19

- 建议 bucket：`irq_handling_timeout`，置信度：高
- 判断理由：等待 core_status=HANDLING_IRQ 超时；中断响应/进入 handler 相关。
- UVM test：`core_ibex_debug_intr_basic_test`，seed：`7`
- sim 首个报错：`UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3965000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- regr 摘要：`riscv_multiple_interrupt_test.0 : [FAILED]`
- sim 相关上下文：
  - `68: UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3965000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- trace 末尾观察：末尾 PC 有重复 800001ae, 800001b0, 800001b4；末尾指令序列: divu -> blt -> sub -> add -> c.addi -> c.addi -> remu -> c.beqz -> c.srli -> jal

case19 结论：这是 IRQ 相关 timeout，不是 branch_condition_wrong。

建议分类：
```text
benchmark_set_2,case_19,irq_handling_timeout,multiple_irq_not_enter_handling_irq
```
regr.log 只有：
```text
riscv_multiple_interrupt_test.0 : [FAILED]
```
关键失败在 sim.log.gz：
```text
+enable_irq_multiple_seq=1
```
UVM_TESTNAME=core_ibex_debug_intr_basic_test
```text
riscv_multiple_interrupt_test_0.bin
```
testbench 发起中断 sequence：
```text
irq_raise_seq_h ... Starting sequence...
irq_raise_seq_h ... Exiting sequence
```
irq: 0b11101110001100000000000010001000
随后 fatal：
```text
Did not receive core_status HANDLING_IRQ within 750 cycle timeout period
```
所以它等待 core 进入 HANDLING_IRQ 状态，但 750 cycle 内没等到。

trace 也支持这个判断。IRQ 发起大约在 2463000 到 2467000，对应 trace 附近仍在普通程序循环/ALU 指令里，例如：
```text
800001a0  or x8,x16,x24
800001a2  srl x24,x3,x24
800001a6  auipc x4,0x86f7d
800001aa  or x14,x30,x27
800001ae  c.addi x29,6
800001b0  divu x14,x24,x21
800001b4  blt x29,x30,8000019e
```
没有看到进入 mtvec/IRQ handler 的控制流，也没有 HANDLING_IRQ 状态被 testbench 捕获。

因此 case19 和 benchmark1 里的 IRQ timeout 类 case 是同一种大类。细分上，因为它是 multiple interrupt 测试，我建议记为：

顶层 bucket: irq_handling_timeout
细分 bucket: multiple_irq_not_enter_handling_irq

### case_20

- 建议 bucket：`irq_handling_timeout`，置信度：高
- 判断理由：等待 core_status=HANDLING_IRQ 超时；中断响应/进入 handler 相关。
- UVM test：`core_ibex_debug_intr_basic_test`，seed：`7`
- sim 首个报错：`UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3943000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- regr 摘要：`riscv_single_interrupt_test.0 : [FAILED]`
- sim 相关上下文：
  - `68: UVM_FATAL /workspace/ibex/dv/uvm/tests/core_ibex_base_test.sv(194) @ 3943000: uvm_test_top [uvm_test_top] Did not receive core_status HANDLING_IRQ within 750 cycle timeout period`
- trace 末尾观察：末尾 PC 有重复 800001aa, 800001ae, 800001b0, 800001b4；末尾指令序列: c.addi -> divu -> blt -> sub -> add -> c.addi -> c.addi -> remu -> c.beqz -> c.srli

case20 结论：和 case19 同一顶层 bucket，都是 IRQ 没有被 core 及时进入处理状态；细分上 case20 是 single interrupt。

建议分类：
```text
benchmark_set_2,case_20,irq_handling_timeout,single_irq_not_enter_handling_irq
```
regr.log 只有：
```text
riscv_single_interrupt_test.0 : [FAILED]
```
关键失败在 sim.log.gz：
```text
+enable_irq_single_seq=1
```
UVM_TESTNAME=core_ibex_debug_intr_basic_test
```text
riscv_single_interrupt_test_0.bin
```
testbench 发起单个中断：
```text
irq_single_seq_h ... Starting sequence...
irq_single_seq_h ... Exiting sequence
```
irq: 0b10000000000000000000
随后 fatal：
```text
Did not receive core_status HANDLING_IRQ within 750 cycle timeout period
```
trace 也能配合说明。IRQ 发起在大约 2441000 到 2445000，但这之后 core 仍在普通程序流中循环执行，没有看到跳入 mtvec/IRQ handler：
```text
8000019e  or x8,x16,x24
800001a2  srl x24,x3,x24
800001a6  auipc x4,0x86f7d
800001aa  or x14,x30,x27
800001ae  c.addi x29,6
800001b0  divu x14,x24,x21
800001b4  blt x29,x30,8000019e
```
所以它不是 branch 条件错，而是中断来了以后 core 没有在 750 cycle 内进入 HANDLING_IRQ。

最终建议：

顶层 bucket: irq_handling_timeout
细分 bucket: single_irq_not_enter_handling_irq

### case_21

case21 是 debug 类，但和 case16/17 不完全一样。它不是“没有进入 debug mode”，而是 debug stress 过程中 dret/特权级切换检查失败。

建议分类：
```text
benchmark_set_2,case_21,debug_dret_privilege_timeout,dret_or_privilege_switch_not_observed
```
regr.log：
```text
riscv_debug_basic_test.0 : [FAILED]
```
关键在 sim.log.gz：
```text
+enable_debug_seq=1
```
UVM_TESTNAME=core_ibex_debug_intr_basic_test
```text
riscv_debug_basic_test_0.bin
```
debug stress sequence 被启动、停止、退出：
```text
debug_seq_stress_h ... Starting sequence...
debug_seq_stress_h ... Stopping sequence
debug_seq_stress_h ... Exiting sequence
```
ECALL instruction is detected, test done
但最后 UVM fatal：
```text
No dret detected, or incorrect privilege mode switch in timeout period of 20000 cycles
```
这里和 case16/17 的区别很大：

case16/17 是：
```text
Did not receive core_status IN_DEBUG_MODE
```
case21 是：
```text
No dret detected, or incorrect privilege mode switch
```
trace 里其实能看到很多 dret：
```text
8000519a  7b200073  dret
8000519e  7b200073  dret
```
...
并且后面周期性出现：
```text
8000519e  7b200073  dret
80005236  csrrs x22,mcause,x0
```
这说明问题未必是“完全没有执行到 dret 指令”，而更像是 testbench 期待的 dret 后 privilege mode / core status 切换没有被正确观察到。UVM fatal 文案本身也给了两个可能条件：“No dret detected, or incorrect privilege mode switch”。

因此我建议不要把它和 debug_entry_timeout 混在一起，而是单独作为 debug 返回路径相关 bucket：

顶层 bucket: debug_dret_privilege_timeout
细分 bucket: dret_or_privilege_switch_not_observed

### case_22

具体原因要落在 trace.log.gz，不是 regr.log。

regr.log 只是报出第一个不一致点：

[regr.log](/Users/luopo/竞赛相关/2026ICCAD contest/数据集/problem/benchmark_set_2/case_22/regr.log:1)
```text
ibex[88] : pc[8000f8e0] addi x5,x5,-124
spike[88] : pc[ffffffff800016da] addi t0,t0,-64
```
也就是说：Ibex 在执行 0x8000f8e0，Spike 认为应该执行 0x800016da。

真正解释“为什么 Ibex 跑到 0x8000f8e0”的是 trace.log.gz 解压后的第 101-106 行：
```text
101  2021000   910   80000188   jalr x2,118(x26)      x26:0x8000165c  x2=0x8000018c
102  2117000   958   800016d2   div x26,x19,x26       x26=0x00000000
103  2143000   971   80000000   jal x0,8000f8d8
104  2197000   998   8000f8d8   dret
105  2199000   999   8000f8dc   dret
106  2237000   1018  8000f8e0   addi x5,x5,-124       x5=0x80022da0
```
这里最关键的是两处：

第一，trace.log.gz 第 103 行：
```text
80000000 jal x0,8000f8d8
```
Ibex 从正常程序流 800016d2 附近突然进入了 0x80000000，再跳到 0x8000f8d8。这就是和 Spike 分叉的开始。

第二，trace.log.gz 第 104-106 行：
```text
8000f8d8 dret
8000f8dc dret
8000f8e0 addi x5,x5,-124
```
这里非常可疑：dret 是 debug return，按直觉它应该从 debug mode 返回到被打断的正常程序 PC，而不是继续顺序执行到下一条 8000f8dc，更不应该一路落到 8000f8e0 做保存上下文/handler 代码。
所以更细的根因我会改成：
```text
debug_dret_fallthrough_or_bad_dpc_restore
```
再看 sim.log.gz，它解释为什么 debug 事件会在这个窗口发生。解压后第 65 行：
```text
@ 2049000: debug_seq_stress_h Starting sequence...
```
而 Ibex 在 trace.log.gz 第 102-103 行分叉的时间是：

2117000  pc=800016d2
2143000  pc=80000000
时间上正好落在 debug stress sequence 启动之后。因此完整链条是：

sim.log.gz 第 65 行：debug stress sequence 启动。
trace.log.gz 第 102 行：Ibex 正在正常程序 800016d2。
trace.log.gz 第 103 行：Ibex 被带到 80000000 -> 8000f8d8。
trace.log.gz 第 104-106 行：执行 dret 后没有回到 Spike 期望的正常路径，而是落到 8000f8e0。
regr.log 第 2-3 行：ISS compare 报 Ibex 8000f8e0 vs Spike 800016da。
所以 case22 不是 branch 条件错。更准确地说，它是 debug stress 触发后，DRET/Debug return 没有恢复到正确 PC，导致 Ibex 进入 debug handler/保存上下文路径，与 Spike 正常路径分叉。

我建议 bucket 用：
```text
debug_dret_restore_wrong
```
细分：
```text
dret_falls_through_to_debug_handler
```

### case_23

case23 的具体根因在 trace.log.gz 第 457 行：
```text
457  10515000  5157  800005a4  7b200073  dret
458  10517000  5158  800005a8  01a393b3  sll x7,x7,x26
459  10519000  5159  800005ac  0001      c.addi x0,0
460  10537000  5168  800005ae  01ab3eb3  sltu x29,x22,x26
```
也就是：Ibex 在普通程序流里执行了 dret，但没有进入 trap/handler，而是直接顺序执行了下一条 800005a8。

这和 regr.log 的首个 mismatch 正好对上：

[regr.log](/Users/luopo/竞赛相关/2026ICCAD contest/数据集/problem/benchmark_set_2/case_23/regr.log:1)
```text
ibex[371] : pc[800005ae] sltu x29,x22,x26: t4:00000001
spike[371] : pc[ffffffff80012080] addi t0,t0,-124: t0:80022da0
```
Spike 认为执行到这里时应该已经进 0x80012080 的 handler 了；Ibex 却还在 0x800005ae 正常往下跑。

为什么 Spike 期望去 0x80012080？前面 trace.log.gz 第 53-55 行设置过 trap vector：
```text
53  80000134  addi x22,x22,-304   x22=0x80012000
54  80000138  ori x22,x22,1        x22=0x80012001
55  8000013c  csrrw x0,mtvec,x22
```
而后面正常 trap handler 的入口形态也能看到，比如最终 ecall 时：
```text
55060  800013da  ecall
55061  80012000  jal x0,80012080
55062  80012080  addi x5,x5,-124
```
所以 0x80012080 是这个测试的异常处理入口之一。

结论：case23 的问题是 dret 在非 debug mode / 普通执行流中没有按预期触发异常，Ibex 错误地把它当成可继续执行的指令，导致继续 fall-through 到 800005a8 -> 800005ac -> 800005ae。

建议 bucket：
```text
debug_dret_illegal_not_trapped
```
细分：
```text
dret_in_normal_mode_falls_through
```
它和 case22 有关联，都是 dret 相关；但 case22 是 debug stress 后 dret 恢复 PC/返回路径异常，case23 更明确：普通流里遇到 dret，应该 trap，Ibex 没 trap。

### case_24

可以比较明确地归到：
```text
csr_mip_xprop
```
细分：
```text
mip_read_returns_unknown_bit
```
原因在 trace.log.gz 和 sim.log.gz 都很直接。

regr.log 只有一句：

[regr.log](/Users/luopo/竞赛相关/2026ICCAD contest/数据集/problem/benchmark_set_2/case_24/regr.log:1)
```text
riscv_csr_test.0 : [FAILED]
```
真正的失败原因在 sim.log.gz。这是 CSR 测试：
```text
core_ibex_csr_test
riscv_csr_test_0.bin
```
最早的 assertion 出现在 sim.log.gz 解压后第 65-73 行：
```text
@ 1321000: IbexBranchDecisionValid failed
@ 1321000: IbexDataOffsetKnown failed
@ 1323000: IbexWbStateKnown failed
```
但这些更像是 X 已经扩散之后的连锁症状。更关键的是后面大量集中出现：

IbexCsrWdataIntKnown: !$isunknown(csr_wdata_int)
统计一下 assertion 名称，数量是：
```text
357  IbexCsrWdataIntKnown
54   IbexWbStateKnown
54   IbexDataOffsetKnown
54   IbexBranchDecisionValid
```
所以主症状是 CSR 写数据内部信号 csr_wdata_int 变成 X。

再看 trace.log.gz，第一次明显的 X 来源在 mip CSR 读写附近。比如第 66 行：
```text
80000184  csrrw x1,mip,x13   x13:0x5a5a5a5a  x1=0x00000X00
```
后面不断重复：
```text
80000198  csrrw x1,mip,x13   x1=0x00000X00
800001ac  csrrs x1,mip,x13   x1=0x00000X00
800001c0  csrrs x1,mip,x13   x1=0x00000X00
800001e8  csrrc x1,mip,x13   x1=0x00000X00
8000021c  csrrwi x1,mip,5    x1=0x00000X00
```
这里的关键点是：读 mip 时返回值里有未知位 X，表现为 x1=0x00000X00。
这个 X 进入 CSR 后续读改写路径，导致 csr_wdata_int 也变成 unknown，于是触发大量 IbexCsrWdataIntKnown assertion。

所以 case24 不属于 branch，也不属于 debug/dret。它和 benchmark1 的 case6 比较接近，都是 CSR X-prop；但这里更细地说是 mip CSR 返回 unknown bit，污染 CSR write-data path。

建议 bucket：
```text
csr_mip_xprop
```
更粗一点也可以并到：
```text
csr_wdata_xprop
```

### case_25

case25 也是 CSR/X-prop 类，但比 case24 多了一个中断上下文。建议归到：
```text
csr_mip_xprop
```
细分：
```text
irq_handler_mip_read_returns_unknown_bit
```
regr.log 只有测试失败名：

[regr.log](/Users/luopo/竞赛相关/2026ICCAD contest/数据集/problem/benchmark_set_2/case_25/regr.log:1)
```text
riscv_interrupt_csr_test.0 : [FAILED]
```
sim.log.gz 说明这是 interrupt CSR 测试，而且开了单中断序列：
```text
+enable_irq_single_seq=1
+UVM_TESTNAME=core_ibex_irq_csr_test
+riscv_interrupt_csr_test_0.bin
```
第 65-67 行中断被拉起：
```text
@ 1951000: irq_single_seq_h Starting sequence...
@ 1953000: irq_single_seq_h Exiting sequence
@ 1955000: irq: 0b10000000000000000000
```
之后进入 interrupt handler，trace.log.gz 的关键位置是第 166 行：
```text
166  3561000  1680  8001116e  csrrs x22,mip,x0   x22=0x00080X00
```
这里读 mip 出来了未知位 X。紧接着第 167 行把这个带 X 的值写到签名地址：
```text
167  3581000  1690  80011172  sw x22,0(x8)  x22=0x00080X00
```
然后第 169 行附近继续用这个 X 值清 mip：
```text
80013008  csrrs x22,mip,x0     x22=0x00080X00
8001300c  csrrc x22,mip,x22    x22=0x00080X00
```
这就解释了 sim.log.gz 的 assertion。最早 assertion 是：
```text
@ 3697000: IbexDataOffsetKnown failed
@ 3697000: IbexCsrWdataIntKnown failed
```
统计 assertion 名称：
```text
96 IbexDataOffsetKnown
96 IbexCsrWdataIntKnown
```
所以 case25 的链条是：

单中断被注入。
Ibex 进入 interrupt handler。
handler 读 mip。
mip 返回 0x00080X00，出现未知位。
未知位进入 CSR read/modify/write 路径，触发 IbexCsrWdataIntKnown。
同时 X 还污染 LSU 地址/offset 相关路径，触发 IbexDataOffsetKnown。
它和 case24 是同一大类：mip CSR X-prop。区别是 case24 是普通 CSR test 中直接读写 mip，case25 是 interrupt handler 里读/清 mip 时出现 X。

### case_26
```text
csr_mip_xprop
```
细分：
```text
multiple_irq_handler_mip_read_returns_unknown_bit
```
它和 case25 很像，但 case25 是 single interrupt，case26 是 multiple interrupt。

regr.log 只有测试失败名：

[regr.log](/Users/luopo/竞赛相关/2026ICCAD contest/数据集/problem/benchmark_set_2/case_26/regr.log:1)
```text
riscv_multiple_interrupt_test.0 : [FAILED]
```
sim.log.gz 开头说明这是多中断测试：
```text
+enable_irq_multiple_seq=1
+UVM_TESTNAME=core_ibex_debug_intr_basic_test
+riscv_multiple_interrupt_test_0.bin
```
第一次中断拉起在 sim.log.gz 解压后第 65-67 行：
```text
@ 2463000: irq_raise_seq_h Starting sequence...
@ 2465000: irq_raise_seq_h Exiting sequence
@ 2467000: irq: 0b11101110001100000000000010001000
```
真正的根因在 trace.log.gz 里。第一次进入 interrupt handler 后，读取 mip 时出现未知位：
```text
80011b3a  csrrs x22,mip,x0   x22=0x40300X08
80011b3e  sw x22,0(x8)       x22=0x40300X08
```
随后 handler 跳到 80013000，再次读 mip，还是带 X：
```text
80013008  csrrs x22,mip,x0     x22=0x40300X08
8001300c  csrrc x22,mip,x22    x22=0x40300X08
```
这和 sim.log.gz 的最早 assertion 完全对上：
```text
@ 4165000: IbexDataOffsetKnown failed
@ 4165000: IbexCsrWdataIntKnown failed
```
也就是说：多中断进入 handler 后，mip 读值出现 X，然后这个 X 被用于 csrrc mip,x22，污染 CSR 写数据路径，触发 IbexCsrWdataIntKnown；同时也污染 LSU 相关路径，触发 IbexDataOffsetKnown。

后面还有一个 fatal：

mcause.exception_code is encoding the wrong exception type
3 [0x3] vs 11 [0xb]
但这个发生在非常后面，前面已经有大量 X-prop assertion。它更像是 mip X 扩散后的连锁后果，不是第一根因。

所以 case26 的 bucket 我会定为：
```text
csr_mip_xprop
```
更细：
```text
multiple_irq_handler_mip_read_returns_unknown_bit
```
和 case25 可以放同一个大 bucket；如果你要更细分，case25 是 single_irq_handler_mip_read_returns_unknown_bit，case26 是 multiple_irq_handler_mip_read_returns_unknown_bit

### case_27
```text
csr_mip_xprop
```
细分：
```text
single_irq_handler_mip_read_returns_unknown_bit
```
它和 case25 基本是同一个 bug 形态：单中断进入 handler 后，读 mip 得到带 X 的值，然后污染 CSR 写数据路径和 LSU 路径。

regr.log 只有：

[regr.log](/Users/luopo/竞赛相关/2026ICCAD contest/数据集/problem/benchmark_set_2/case_27/regr.log:1)
```text
riscv_single_interrupt_test.0 : [FAILED]
```
sim.log.gz 第 65-67 行显示单中断被注入：
```text
@ 2441000: irq_single_seq_h Starting sequence...
@ 2443000: irq_single_seq_h Exiting sequence
@ 2445000: irq: 0b10000000000000000000
```
真正的根因在 trace.log.gz 第 196-202 行：
```text
196  80010e1a  csrrs x22,mip,x0      x22=0x00080X00
197  80010e1e  sw x22,0(x8)          x22=0x00080X00
198  80010e22  jal x0,80013000
```
```text
201  80013008  csrrs x22,mip,x0      x22=0x00080X00
202  8001300c  csrrc x22,mip,x22     x22=0x00080X00
```
这里非常明确：mip 读出来的值含有未知位 X，即 0x00080X00。
之后这个值又被拿去做 csrrc mip,x22，所以 X 进入 CSR 写路径。

这正好对应 sim.log.gz 第 68-73 行的最早 assertion：
```text
@ 4165000: IbexDataOffsetKnown failed
@ 4165000: IbexCsrWdataIntKnown failed
```
统计 assertion：
```text
561 IbexDataOffsetKnown
561 IbexCsrWdataIntKnown
```
最后 sim.log.gz 第 1362 行还有一个 fatal：
```text
Did not receive core_status HANDLING_IRQ within 750 cycle timeout period
```
但这个是后续结果。因为在它之前很早就已经有 mip X-prop 和 1000 多条 assertion 了，所以我不把 case27 归到普通 irq_handling_timeout，而是归到 csr_mip_xprop。如果需要粗 bucket，它和 case25 可以合并；如果需要细分，case25/case27 都是 single_irq_handler_mip_read_returns_unknown_bit
