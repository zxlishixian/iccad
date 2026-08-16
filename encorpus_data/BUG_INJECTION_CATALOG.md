# Ibex RTL bug 注入类型全清单（造数据按批次执行）

> 给造数据的队友。依据：官方 QA（A2/A3/A5/A7：任意 RTL 修改、Ibex commit `8ce399d…`、消歧、
> RTL 源码注入）、RV32IMCB 指令集、Ibex 微架构、EnCorpus 的 Signal Mix-Ups / Broken
> Conditionals。只列 **bug 类型**，具体改哪个信号/文件由你按 checkout 的 RTL 自行定位。

## 1. 通用注入模式（HOW：任何单元都能套）

信号交换 · 运算符替换（+↔-、&↔|、<<↔>>、==↔!=、<↔>=）· 有符号↔无符号 · 常量错误 ·
条件破坏（if/else/case 改错）· 位宽/截断 · 符号/零扩展用反 · 缺失赋值 · 复位值错误 ·
优先级错误 · 使能/握手错误 · 索引/位选择 off-by-one。

## 2. bug 类型清单（按功能单元，只列类型）

### 取指 / PC / 分支
PC+4 计算错 · 分支目标地址算错 · 分支条件判断错（beq/bne/blt/bge/bltu/bgeu）·
有符号/无符号分支比较错 · 静态分支预测方向错 · jal 目标算错 · jalr 目标算错 ·
指令地址对齐错 · 预取缓冲指针错

### 译码（decoder + compressed decoder）
opcode 译码错 · funct3/funct7 译码错 · I 型立即数扩展错 · S 型立即数拼接错 ·
B 型立即数拼接错 · U 型立即数错 · J 型立即数错 · rs1/rs2/rd 选择错 ·
RV32C 展开错 · RV32C 象限判断错 · 非法指令检测错

### ALU
加法进位错 · 减法借位错 · and/or/xor 替换 · 移位量错 · 移位方向反 ·
算术/逻辑移位错（sra↔srl）· 比较符号错（slt↔sltu）· 比较运算符替换 · 位宽截断错

### 乘除（MDU）
mul 高低位错 · mulhsu 符号错 · mulhu 符号错 · div 符号错 · divu 符号错 ·
rem 符号错 · remu 符号错 · 除零处理错 · 乘法器流水错

### LSU（load/store）
load 地址算错 · store 地址算错 · lb/lbu 符号扩展错 · lh/lhu 符号扩展错 ·
byte enable 错 · 对齐检测错 · 数据宽度选择错 · 读写通道接反

### CSR
CSR 地址译码错 · CSR 权限错 · 写屏蔽错 · mstatus.MIE 错 · mstatus.MPIE/MPP 错 ·
mepc 错 · mcause 错 · mtvec 错 · mtval 错 · mcycle/minstret 计数错 · CSR 复位值错

### 中断 / 异常（controller）
异常向量跳转错 · 中断优先级错 · 中断使能检查错 · handler 状态保存错 ·
mret 状态恢复错 · 中断源映射错 · 非法指令异常漏报 · ecall/ebreak 异常错

### Debug
debug 入口条件错 · dret 返回错 · dpc 错 · dcsr 错 · ebreak 进入 debug 错 ·
单步逻辑错 · dscratch 错

### PMP（物理内存保护）
地址匹配错 · 权限错 · pmpcfg 字段错 · pmpaddr 编码错 · 优先级错

### 总线接口（AHB-Lite）
握手时序错 · 突发传输错 · 读数据通道错 · 写数据/使能错 · 传输类型错

### 寄存器堆 / 写回
写回使能错 · 写回数据源选错 · 写地址错 · 读数据错 · x0 保护错

### 复位 / 初始化
复位值错 · 初始化状态错 · 复位释放时序错

## 3. 粒度维度（粗 vs 细，不同批次要有区分）

官方 hidden 集 k 从 2 到 64：**桶少时 bug 划分粗、桶多时划分细**。所以造数据要两种粒度都覆盖：

- **粗粒度批次**：每个 bug = 一个**独立功能单元**（ALU / MDU / LSU / CSR / 中断 / debug 各一个），
  约 8~12 个 bug。对应小-k 场景（k=2/4/8，官方大概率按"大功能域"划桶）。
- **细粒度批次**：每个 bug = **同一单元里的不同指令/寄存器**（如 ALU 内 add/sub/sll/slt/and/or/xor
  各一个，MDU 内 mul/mulh/mulhsu/mulhu/div/divu/rem/remu 各一个，CSR 内 mepc/mcause/mtvec/mtval/
  mie/mip 各一个）。对应大-k 场景（k=32/64，官方需要在同一单元内拆出多个 bug）。

已造的三批可对照：benchmark8 偏粗（5 个 mismatch bug 各落一个单元），k32_new12 偏细
（多个 bug 挤在 RV32C/MDU 内）——这个混合方向是对的，继续保持。

## 4. 批次划分建议

按单元逐批铺，每批 ~10~12 个**新** bug（不重复已造类型），并有意交替粗/细粒度：

1. **批 4（粗粒度）**：分支 beq/bne/blt/bge/bltu/bgeu + 跳转 jal/jalr + 非法指令检测——补分支核心
2. **批 5（细粒度，CSR）**：mepc/mcause/mtvec/mtval/mie/mip 读写与权限 + 中断优先级 + 中断源映射
3. **批 6（粗粒度，边缘模块）**：PMP + WFI + 总线接口 + 复位初始化
4. **批 7（细粒度，LSU/写回）**：lb/lbu/lh/lhu 扩展选择 + byte enable + 对齐 + 写回数据源 + x0 保护
5. 之后继续把 ALU/MDU/译码里还没用到的「运算符替换 / 符号错误」变体铺开（上百种的空间）

每批硬门槛照旧：late first-mismatch、消歧自检、`mismatch_print_limit=1`、无泄漏、
每 bug ≥2 测试且无 1:1 捷径。
