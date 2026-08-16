# Ibex RTL bug 注入类型全清单（造数据按批次执行）

> 给造数据的队友。依据：官方 QA（A2/A3/A5/A7：任意 RTL 修改、Ibex commit `8ce399d…`、消歧、
> RTL 源码注入）、RV32IMCB 指令集、Ibex 微架构、EnCorpus 的 bug 注入模式（Signal Mix-Ups /
> Broken Conditionals）、以及通用硬件 bug 分类学。
>
> 用法：按 §2 的功能单元分组，**逐个单元铺开**，每批造一个单元里的若干 bug。总原则见
> `DATA_GENERATION_ROADMAP.md`：不合并、不卡 k、类型越广越好、严格按官方流程（不写自定义断言）。

## 1. 通用注入模式（HOW：任何单元都可用）

这 12 种是「怎么改 RTL」，可作用在下面任意功能单元的信号/逻辑上：

| 模式 | 含义 | 典型效果 |
|---|---|---|
| **信号交换** | 两个信号/接线接反 | 功能错乱（EnCorpus Signal Mix-Ups）|
| **运算符替换** | `+`↔`-`、`&`↔`\|`、`<<`↔`>>`、`==`↔`!=`、`<`↔`>=` | 算错/比错 |
| **有符号↔无符号** | 比较/除法/移位的有符号性改错 | slt↔sltu、div↔divu、sra↔srl |
| **常量错误** | 常量/立即数/偏移量写错（如 `4`→`8`）| 地址/计数/对齐错 |
| **条件破坏** | `if`/`else`/`case` 分支条件改错、default 删掉 | 控制流错（EnCorpus Broken Conditionals）|
| **位宽/截断** | 信号位宽声明错、高位截断 | 数据被截断/溢出 |
| **符号/零扩展** | sign-extend ↔ zero-extend 用错 | 负数变正数 |
| **缺失赋值** | 漏掉 default/某个 case 的赋值 | 偶发错误/锁存 |
| **复位值错误** | 寄存器复位初值写错 | 启动即错（注意别退化成 cascade）|
| **优先级错误** | FSM/case 的优先级排反 | 多条件冲突时选错 |
| **使能/握手错误** | 使能信号漏接、握手时序错 | 时序/停顿错 |
| **索引/偏移错** | 数组索引、位选择 off-by-one | 取错寄存器/位 |

## 2. 按功能单元的具体 bug 清单（WHERE）

每个 bug 标注症状类型与区分信号：
- **[M]** = mismatch 类（Ibex vs Spike 指令分歧，区分靠首个 mismatch 的 opcode/寄存器）
- **[T]** = test_fail 类（测试断言失败，区分靠测试名/标准 UVM 断言）

### 2.1 取指 / PC / 分支预测
| bug | 改法 | 症状 |
|---|---|---|
| PC+4 计算错 | next PC 常量/加法改错 | [M] 连续取指错位 |
| 分支目标地址算错 | branch target = PC+imm 的 imm 用错 | [M] 分支处 PC 分歧 |
| 分支条件判断错 | beq/bne/blt/bge/bltu/bgeu 的比运算符改错 | [M] 分支处分歧 |
| 有符号/无符号分支比较错 | blt↔bltu、bge↔bgeu | [M] |
| 静态分支预测方向错 | 预测 taken/not-taken 反 | [M] 偶发取指错 |
| 跳转 jal/jalr 目标算错 | jalr 的 base+imm 或 &~1 对齐错 | [M] |
| 指令地址对齐错误 | 取指地址未 4/2 字节对齐 | [T] 非法指令/取指异常 |
| 预取缓冲指针错 | prefetch FIFO 读写指针 off-by-one | [M] 偶发 |

### 2.2 译码（decoder + compressed decoder）
| bug | 改法 | 症状 |
|---|---|---|
| 32 位 opcode 译码错 | 把某 opcode 判成另一条指令 | [M] 该指令处分歧 |
| funct3/funct7 译码错 | add↔sub、sll↔srl 等 | [M] |
| I 型立即数扩展错 | 12 位立即数符号扩展错 | [M] addi/ori 等 |
| S/B 型立即数拼接错 | store/branch 的 imm 字段位序错 | [M] 访存/分支地址错 |
| U/J 型立即数错 | lui/auipc/jal 的高位立即数错 | [M] |
| 寄存器选择错 | rs1/rs2/rd 译码错位 | [M] 用错寄存器 |
| 压缩指令扩展错 | RV32C → 32 位展开时 opcode/立即数错 | [M] c.* 处分歧 |
| 压缩象限判断错 | c0/c1/c2 象限误判 | [M] c.* 处分歧 |
| 非法指令检测漏/错 | 合法判非法或反之 | [T] illegal_instr 测试 |

### 2.3 ALU
| bug | 改法 | 症状 |
|---|---|---|
| 加减法进位错 | 加法进位链/减法借位改错 | [M] add/sub |
| 逻辑运算替换 | and↔or↔xor 接反 | [M] 对应指令 |
| 移位量错 | 移位量的低 5 位取错 | [M] sll/srl/sra |
| 移位方向反 | 左移↔右移 | [M] |
| 算术/逻辑移位错 | sra↔srl（符号扩展 vs 补零）| [M] |
| 比较符号错 | slt↔sltu | [M] |
| 比较运算符替换 | < ↔ >=、== ↔ != | [M] |
| 位宽截断错 | 结果高位截掉 | [M] 高位数据错 |

### 2.4 乘除（MDU）
| bug | 改法 | 症状 |
|---|---|---|
| mul 高低位错 | mul↔mulh 结果选错 | [M] mul/mulh |
| mulhsu 符号错 | 有符号×无符号 符号处理错 | [M] mulhsu |
| mulhu 符号错 | 无符号×无符号 高位零扩展错 | [M] mulhu |
| div/divu 符号错 | 有符号↔无符号除法 | [M] div/divu |
| rem/remu 符号错 | 余数符号错 | [M] rem/remu |
| 除零处理错 | 除零时结果/异常错 | [M]/[T] |
| 乘法器流水错 | 乘法器的 stage 使能/锁存错 | [M] 偶发 |

### 2.5 LSU（load/store）
| bug | 改法 | 症状 |
|---|---|---|
| load 地址算错 | base+imm 的 imm 编码错 | [M] lw/lb 等 |
| store 地址算错 | S 型立即数用成 I 型 | [M] sw/sb 等 |
| 符号/零扩展选择错 | lb↔lbu、lh↔lhu | [M] 符号位错 |
| byte enable 错 | 部分写时 byte 使能位错 | [M] sb/sh 部分写错 |
| 对齐检测错 | 非对齐访存未报/误报 | [T] mem_error |
| 数据宽度选择错 | word/half/byte 选错 | [M] 数据错位 |
| 读写选择错 | load↔store 数据通道接反 | [M] |

### 2.6 CSR
| bug | 改法 | 症状 |
|---|---|---|
| CSR 地址译码错 | 某 CSR 号判成另一个 | [T] CSR 测试 |
| CSR 权限错 | 用户态访问特权 CSR 未报异常 | [T] 越权 |
| CSR 写屏蔽错 | WPRM 位未生效，写了不该写的位 | [T] |
| mstatus.MIE 更新错 | 中断使能位写错 | [T] interrupt 测试 |
| mstatus.MPIE/MPP 错 | 进入/退出 handler 状态错 | [T] |
| mepc 写入错 | 异常 PC 保存错 | [T] 异常返回错 |
| mcause 编码错 | 异常/中断原因码错 | [T] 对应测试 |
| mtvec 基址/模式错 | 向量基址或 vectored/direct 错 | [T] 中断入口错 |
| mtval 值错 | 故障地址/值写错 | [T] |
| mcycle/minstret 计数错 | 计数器递增错 | [T] |
| CSR 复位值错 | CSR 初值错 | [T]/[M] 启动即错 |

### 2.7 中断 / 异常（controller）
| bug | 改法 | 症状 |
|---|---|---|
| 异常向量跳转错 | mtvec 计算/跳转错 | [T] |
| 中断优先级错 | 多个中断同时来，选错 | [T] 多中断测试 |
| 中断使能检查错 | 漏查 MIE/全局使能 | [T] |
| 进入 handler 状态保存错 | mstatus/mepc/mcause 保存错 | [T] |
| mret 状态恢复错 | MIE←MPIE 等恢复错 | [T] |
| 中断源映射错 | mip/mie 位 ↔ 中断源接反 | [T] |
| 非法指令异常漏报 | 非法指令未触发异常 | [T] illegal_instr |
| ecall/ebreak 异常错 | 系统调用/断点异常类型错 | [T] |

### 2.8 Debug
| bug | 改法 | 症状 |
|---|---|---|
| debug 入口条件错 | 何时进 debug mode 判错 | [T] debug 测试 |
| dret 返回错 | 从 debug 返回 PC/特权错 | [T] |
| dpc 保存/恢复错 | debug PC 错 | [T] |
| dcsr 字段错 | debug 状态位更新错 | [T] |
| ebreak 进入 debug 错 | ebreak 未进 debug | [T] debug_ebreak |
| 单步逻辑错 | single-step 使能/触发错 | [T] |
| dscratch 读写错 | 调试暂存寄存器错 | [T] |

### 2.9 PMP（物理内存保护）
| bug | 改法 | 症状 |
|---|---|---|
| PMP 地址匹配错 | 地址区间匹配（TOR/NAPOT）错 | [T] PMP 测试 |
| PMP 权限错 | R/W/X 权限检查错 | [T] |
| pmpcfg 字段错 | 配置寄存器位（A/R/W/X/L）错 | [T] |
| pmpaddr 编码错 | 地址寄存器编码错 | [T] |
| PMP 优先级错 | 多 region 匹配时选错 | [T] |

### 2.10 总线接口（AHB-Lite）
| bug | 改法 | 症状 |
|---|---|---|
| 握手时序错 | HREADY/HRESP 时序错 | [M]/[T] 偶发 |
| 突发传输错 | burst 地址/长度错 | [M] |
| 读数据通道错 | HRDATA 对齐/位序错 | [M] |
| 写数据/使能错 | HWDATA/HWSTRB 错 | [M] |
| 传输类型错 | idle/busy/seq/nonseq 状态机错 | [M]/[T] |

### 2.11 寄存器堆 / 写回（register file / writeback）
| bug | 改法 | 症状 |
|---|---|---|
| 写回使能错 | 该写不写/不该写写 | [M] 偶发 |
| 写回数据源选错 | ALU/LSU/CSR/PC 结果选错 | [M] |
| 写地址错 | rd 译码/传递错 | [M] |
| 读数据错 | rs1/rs2 读出口接反 | [M] |
| x0 保护错 | x0 被覆盖（未强制为 0）| [M] |

### 2.12 复位 / 初始化
| bug | 改法 | 症状 |
|---|---|---|
| 复位值错 | 某寄存器复位初值错 | [M]/[T] 启动即错 |
| 初始化状态错 | FSM 初始态错 | [M]/[T] |
| 复位释放时序错 | 复位释放过早/过晚 | [M]/[T] 偶发 |

## 3. 症状 → 区分信号对照（造完自检用）

| bug 类别 | 症状 | 模型可用的区分信号 |
|---|---|---|
| mismatch 类 | Ibex vs Spike 指令/寄存器分歧 | 首个 mismatch 的 **opcode 家族**（+寄存器+PC 区域）|
| test_fail 类（功能域）| 测试 FAILED，无 mismatch | **测试名（功能域）** + 标准 UVM 断言文本 |
| test_fail 类（内存）| mem_error | 标准 mem_model 断言（`read to uninitialized addr`）|

消歧要求：**不同 bug 至少在一个区分信号维度上不同**（opcode 家族 / 测试名 / 断言文本 / trace 分歧模式）。同家族多个 bug 要确保它们的首 opcode 或断言可区分，否则二选一剔除。

## 4. 给队友的批次划分建议

按功能单元逐批铺，每批 ~10~12 个**新** bug（不重复已造类型）：

1. **批 4**：分支 beq/bne/blt/bge/bltu/bgeu（§2.1）+ 非法指令检测（§2.2）——分支核心三兄弟还没覆盖
2. **批 5**：CSR 全空间 mepc/mcause/mtvec/mtval/mie/mip（§2.6）+ 中断优先级（§2.7）
3. **批 6**：PMP（§2.9）+ WFI 唤醒 + 总线接口（§2.10）
4. **批 7**：LSU byte enable/对齐（§2.5）+ 寄存器堆写回（§2.11）+ 复位（§2.12）
5. 之后：把 ALU/MDU/译码里还没用到的「运算符替换 / 符号错误」变体继续铺开（上百种的空间）

每批仍满足硬门槛：late first-mismatch、消歧自检、`mismatch_print_limit=1`、无泄漏、每 bug ≥2 测试且无 1:1 捷径。
