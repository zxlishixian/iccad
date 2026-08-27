# 造数据路线图：下一批该造什么、怎么造

> 给数据生成的同学。基于：官方 Q&A（B_QA_20260612/20260727）、`official_bug_injection.md`、
> `official_bug_distribution.md`、`bug_type_taxonomy.md`、`fake_vs_official_gap.md`，以及我们
> 对 benchmark5_500cases / benchmark8_500cases 两批数据的独立质量审计与模型实测结论。
> 已有两批数据的反馈见 `NEXT_BATCH_FEEDBACK.md`；本文聚焦**下一批的选题与工程细节**。

## 总原则（先读这条，以下所有建议都在这个边界内）

**严格按照官方公开的造数据标准流程来造数据，禁止造数据流程越界。** 官方已披露的信息就是边界：

- bug 注入 = 官方 Ibex commit（`8ce399dbe678f0a66856ac302ec7609ba366d8fd`）之上的**任意 RTL 修改**，
  VCS + UVM testbench 是 oracle —— **不写自定义断言、不改 testbench、不人为加工日志格式**。
- `mismatch_print_limit=1`、**消歧**（剔除 diff-bugs-same-syndrome）。
- 凡官方没公开的、或会改变日志分布的手段（自定义断言、精确测试名→bug 映射、人为改写日志），一律不做。

在此边界内，**设计注入 bug 时范围种类越多越好，目标覆盖官方 hidden 数据集的分布**。

## 0. 背景：官方 hidden 集到底长什么样（决定造什么）

官方 10 个 benchmark（B_20260601 §3.2）：N=2~3000、k=2~64，其中 k 序列为 **2/4/8/16/32/64**，
且最大两个 benchmark（N=3000）是 **k=64**。QA A11 明确 buckets 数 = bug 数，QA A12 明确 --k 按表原值传入。

我们当前**干净**的 fake 数据覆盖：k=3（official_vcs）、k=4（stable）、k=10（directed_cross_v4 /
benchmark5_500 / benchmark8_500）、k=12（k32_new12），共 **32 个 distinct 干净 bug**（bug_037~127，
分散在 3 个新集里）。旧 benchmark6_final（64 bug）和 benchmark5_final（32 bug）是 lui 级联的
**歧义数据**，不能当高-k 预演。

**结论：我们缺的不是某个固定 k，而是「更广的 bug 类型覆盖」——官方 hidden 集 bug 类型空间远超 64，我们 32 个还远远不够。**

## 1. P0（最高优先级）：持续造新数据集，bug 类型/数量/表现越广越好

**核心原则：不合并现有数据集、也不卡某个 k 值。** 官方 A2 说 bug 是「任意 RTL 修改」、无固定
故障分类法；而 Ibex 的 RV32IMC ~80 条指令 + ~20 个 CSR + ~20 个中断源 + debug/PMP/总线等，
**每条指令/每个寄存器的实现都可能出错，distinct bug 类型可达上百种**。所以目标就是让 fake 数据的
bug 类型、case 数量、日志表现**越广越好**，尽可能覆盖 hidden 集的分布。

**进度**：已积累 **32 个 distinct 干净 bug**（benchmark5=10 + benchmark8=10 + k32_new12=12）。
接下来继续造**全新的独立数据集**（每批 ~10~12 个新 bug，别重复已造类型），按 §2 缺口清单逐个补
功能单元，直到把上百种类型尽量铺开。

- 硬要求（每批都要满足，和 benchmark5/8 一样）：
  1. **late first-mismatch**（首个 mismatch index > 32，杜绝 cycle-0 lui 级联）
  2. **每 bug 症状可区分**（见 §3 消歧）
  3. `mismatch_print_limit=1`，regr.log 只有单个 `Mismatch[1]`
  4. 无 bug label / 真实绝对路径泄漏
- 为什么：bug 类型越广，模型见过的症状空间越接近 hidden 集，泛化越稳。

## 2. P1：bug 类型覆盖越广越好（目标覆盖 hidden 分布）

**目标：注入 bug 的范围种类越多越好，尽量贴近官方 hidden 数据集的 bug 类型分布。** 官方 A2 明确
「无预定义故障分类法、bug 是任意 RTL 修改」，所以我们应把 Ibex 各功能单元都覆盖到，而不是只盯
几个已知官方 bug 号。

已覆盖（三批共 32 bug）：ALU 算数/逻辑/移位、MDU 的 mul/mulhsu/**div/rem**、LSU 符号扩展与
store 地址、分支比较器与立即数编码、AUIPC、CSR 的 MIE/mstatus、中断向量与原因、debug ebreak/入口、
**RV32C 压缩指令译码（c.addi/c.li/c.sub/c.slli/c.mv/c.lwsp）**。

**下一批优先补这些（官方有对应证据的标了官方 bug 号；MDU 除余、RV32C 已在 k32_new12 补上）：**

| 缺口 | 具体 bug 建议 | 症状 |
|---|---|---|
| **分支 beq/bne/blt** | 等号/不等号/小于条件错（目前只有 bgeu + 立即数编码，beq/bne/blt 这核心三兄弟还没覆盖）| mismatch（PC 分歧）|
| **CSR 全空间** | mepc/mtvec/mtval/mcause/mie/mip 读写错、用户态访问特权 CSR 越权（目前只碰了 MIE/mstatus）| test_fail |
| **LSU byte enable/对齐** | 部分写使能错、非对齐访存、符号/零扩展选择 | mismatch 或 mem_error |
| **PMP（物理内存保护）** | 权限检查错、地址匹配错 | test_fail |
| **WFI/睡眠** | WFI 唤醒逻辑错（中断未唤醒/错误唤醒）| test_fail |
| **总线接口（AHB-Lite）** | 握手/突发传输错 | mismatch / 时序错 |
| **复位/初始化** | 复位值错、初始化状态错 | 早期 mismatch（注意别退化成 cascade）|

参考 `bug_type_taxonomy.md` 的 A1~A8 / B1~B4 全景表。

## 3. P2：消歧铁律系统化（每加一个 bug 都要跑）

官方 Q5 明确剔除了 diff-bugs-same-syndrome。造数据必须保证**每个 bug 在至少一个可观测维度上有
唯一、稳定的症状签名**：

- mismatch 类：首个 mismatch 的 **opcode 家族 + 操作数模式 + PC 区域**。
- test_fail 类：**标准 testbench 断言 + file:line + 测试名（功能域）**。

**两条硬规则：**
1. 每个 bug 造完后，和已有所有 bug 的签名比对；若有 bug 签名完全重叠 → 删掉其中一个（消歧）。
2. **不要写自定义断言**。官方流程是「注入 RTL bug → 标准 testbench 跑 → 标准断言自然触发」
   （testbench 是 oracle，A7），官方不写自定义断言。`read to uninitialized addr`
   （mem_model.sv:28）是**标准的 mem_model 断言**，官方 bug_2014（`mem_error` 测试）也会触发它
   ——这是合法的官方式症状，**不是数据质量问题**。所谓"消歧"发生在**选 bug 层面**：选那些触发
   **不同标准症状**（不同断言 / 不同 opcode / 不同 trace 分歧模式）的 bug。若两个 bug 连标准
   症状都完全一样（官方 Q5 说的"几乎一样、只差时间戳"），才二选一剔除；地址不同（0x13 vs 0x3）
   配合 trace 已经算可区分，官方同样靠这些细节区分，**不需要人为造更具体的断言**。

## 4. P3：首个 mismatch 位置要多样化（不要全 late）

benchmark5/8 的首 mismatch 全是 late（index 32+）。但官方分布里，**译码类 bug 在特定指令即显形
（较浅）**，而**条件位翻转/计数器触发类 bug 要深执行才显形（很深）**。下一批建议有意混合：
- 一部分 bug 设计成"特定指令/特定 CSR 即触发"（浅，index 小）
- 一部分 bug 设计成"执行 N 条指令后才触发"（深，index 大）

这样首 mismatch index 的分布更贴近真实，也顺便给模型提供"分歧深度"这个判别维度。

## 5. P4：测试多样性与种子多样性

- 用更多 Ibex riscv-dv 测试（官方 testlist 有 ~50 个），别只停留在 12~16 个。
- 每个 bug 覆盖 ≥2 个测试、每个测试覆盖 ≥2 个 bug（无 1:1 捷径，保持）。
- 每个 bug 用多个 `+ntb_random_seed` / 不同 bin 文件，产生**多样但功能域稳定**的症状
  （同 bug 的 50 个 case 应落在同一功能域，如都 MDU 或都 CSR——这是模型能学到稳定签名的前提）。

## 6. P5：验证门槛（造完自检，缺一不可）

1. 歧义自检：任意两个 bug 无完全相同的症状签名（§3）。
2. 测试名自检：无 test→bug 的 1:1 映射。
3. 三日志逐字节相同但跨 bug 的 case = 0（用户定义的硬标准）。
4. late first-mismatch 抽查：mismatch case 首个 index > 32。
5. 泄漏检查：日志 0 个 bug label / 真实绝对路径。
6. 规模检查：单日志行数 < 10M，k 与 N 符合目标 benchmark 规格。

## 7. 一句话总结

**严格按照官方公开的造数据标准流程的信息来造数据，禁止造数据流程越界。继续生成一批和官方隐藏
数据集规格相符合的数据，同时设计注入 bug 时，尽量使范围种类越多越好，目标就是能够使得覆盖官方
隐藏数据集的分布。**

## 8. 当前验证状态（2026-08-26）

当前已形成两个彼此独立、可复核的官方布局数据包：

- `coverage64_release_20260826_v4`：999 cases、68 个 root，已通过官方 validator 和完整质量审计。
- `batch11_k32_new12_380cases_official`：380 cases、12 个细粒度 root，覆盖 MDU、RV32C、LSU、ALU/Decode、Branch；已通过 validator、深度、泄漏、单 mismatch block 和签名消歧审计。

当前新增验证的 `bug_104`、`bug_119` 已在 batch11 中稳定覆盖两个标准 test；`bug_057` 已在 v4 中覆盖三个 interrupt test。`bug_150/151/154`、`bug_205/206/207` 等仅单测试或无 mismatch 的候选不纳入正式包。

PMP 在官方配置中由 `PMPEnable=0` 禁用，当前 ISA 为 `RV32IMC`、没有 RV32B 指令，源码没有 AHB-Lite 接口；因此这些域不能通过自定义 testbench、断言或手工 failure marker 伪造。独立流水线/hazard 根因同样暂未找到稳定的标准 oracle，继续保持为未观测状态。
## 9. 新增寄存器堆/流水线补充（2026-08-27）

独立包 `register_pipeline_supplement_20260827.tar.gz` 已完成并回传，包含
20 个官方布局 case、3 个新 RTL 根因：

- `bug_218`：MUL 多周期等待阶段漏 stall，9 cases。
- `bug_222`：寄存器堆写入 x27 时翻转 bit 0，8 cases。
- `bug_223`：寄存器堆写入 x20 时翻转 bit 0，3 cases。

该包通过官方 validator（0 errors/0 warnings），每个 bug 至少覆盖 2 个
测试，每个测试至少覆盖 2 个 bug；详细 mismatch 均为单个 `Mismatch[1]`，
首分歧为 39--233，无泄漏、无重复三日志组合。

`bug_217`（完全取消 mult/div stall）、`bug_220`（branch stall）和 x27
写入抑制探针因 timeout 或全 clean 被淘汰；`bug_223` 的两个首分歧为 28
的 case 也未进入独立包。上述淘汰遵循宁缺毋滥原则。

## 10. 寄存器堆/流水线补充 v2（2026-08-27）

在不改动旧包的前提下，新增独立包
`register_pipeline_supplement_20260827_v2.tar.gz`，共 26 个 case、4 个
根因：`bug_218` 9 个、`bug_222` 8 个、`bug_223` 3 个、`bug_224` 6 个。
其中 `bug_224` 为 x21 写回 bit 0 翻转。该包保留了 bug/test 交叉覆盖，
所有 case 的首分歧均大于 32（39--233），通过 validator，且无日志泄漏、
精确重复三日志组合或环境失败。`bug_224` 的首分歧为 29/31 的 4 个
case 被排除，没有为均衡而放松消歧门槛。

## 11. MDU 除法/余数补充（2026-08-27）

独立包 `mdu_supplement_20260827.tar.gz` 新增 18 个 case、两个细粒度
MDU 根因：`bug_225` 为 REMU 除零结果 bit 0 翻转，`bug_226` 为 signed
REM 的符号源错误。两者各 9 个 case，均覆盖
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`。所有 case 首分歧为 73--1036，validator 和
独立签名审计均通过，旧包未被改写。

## 12. Branch/Jump 补充（2026-08-27）

独立包 `branch_supplement_20260827.tar.gz` 新增 16 个 case、两个细粒度
控制流根因：`bug_228` 为条件 BGE comparator inversion，`bug_229` 为
条件 BGEU comparator inversion。两个 bug 各 8 个 case，共享
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`。所有保留 case 首分歧为 96--2809，validator
和签名审计通过；一个 clean run 和一个 timeout run 被排除。
## 13. CSR/中断候选复核（2026-08-27）

`bug_230`（压缩非法指令 `mtval` bit 翻转）和 `bug_232`/`bug_233`
（多源中断优先级条件错误）均完成 9 个标准流程运行，但没有产生可收录
的 failure-only case；`bug_231`（`mtvec` MODE 读回错误）仅有少量标准
`CSR TEST_FAILED`，覆盖不足且未满足交叉测试门槛。因此这些候选均保持
probe-only，不进入任何数据包，也不添加人工 failure marker。
## 14. Decode/Branch 条件触发补充（2026-08-27）

新增独立包 `decode_branch_supplement_20260827.tar.gz`，共 11 cases：
`bug_228` 8 例、`bug_234` 3 例。三种测试均同时覆盖两个 bug，避免 test 名
一对一捷径；所有保留 case 的首个 mismatch index 为 50--1418，validator
为 0 errors / 0 warnings。`bug_235` 全部 clean，bug_234 的 6 个早期分叉
case 已剔除，均未放宽消歧和官方日志流程。
## 15. ALU/Decode 条件触发补充（2026-08-27）

新增独立包 `alu_decode_supplement_20260827.tar.gz`，共 16 cases：
`bug_239` 4 例、`bug_240` 6 例、`bug_241` 6 例。三个 bug 均覆盖
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，validator 为 0 errors / 0 warnings；首个
mismatch index 为 60--4589，无日志泄漏或完整日志重复。bug236--238
探针全部 clean，因此未纳入正式包。
## 16. MDU/Decode 条件触发补充（2026-08-27）

新增独立包 `mdu_decode_supplement_20260827.tar.gz`，共 10 cases：
`bug_242` 4 例、`bug_243` 6 例。两者均覆盖三种标准 riscv-dv test，
validator 为 0 errors / 0 warnings；首个 mismatch index 为 170--2119，
无日志泄漏或完整日志重复。bug242 的两个 clean run 未收录。
## 17. BEQ/BNE/BLT 分支译码补充（2026-08-27）

新增独立包 `branch_decode_supplement_20260827.tar.gz`，共 16 cases：
`bug_244` 5 例、`bug_245` 6 例、`bug_246` 5 例。三个 bug 均覆盖三种
标准 riscv-dv test，validator 为 0 errors / 0 warnings；首个 mismatch
index 为 107--6077，无日志泄漏或完整日志重复。一个 clean run 和一个
timeout run 已排除。
## 18. LSU 条件触发补充（2026-08-27）

新增独立包 `lsu_supplement_20260827.tar.gz`，共 7 cases：`bug_247` 3 例、
`bug_249` 4 例。两个 bug 均覆盖 `riscv_unaligned_load_store_test` 和
`riscv_mmu_stress_test`，validator 为 0 errors / 0 warnings；首个 mismatch
index 为 59--5255，无日志泄漏或完整日志重复。bug248 因只在单一测试触发
而淘汰。

## 19. Interrupt state 补充（2026-08-27）

新增独立包 `interrupt_state_supplement_20260827.tar.gz`，共 20 cases：
`bug_253` 10 例、`bug_254` 10 例。两者均覆盖
`riscv_single_interrupt_test` 和 `riscv_multiple_interrupt_test`，两个测试
也各覆盖两个 bug，避免测试名一对一捷径。`bug_253` 的标准 UVM 失败为
`mstatus.mpie was not set to 1'b1 after entering handler`，`bug_254` 的
标准 UVM 失败为等待 `csr 0x300` 写回超时。

该包由干净 base commit 之上的 RTL 注入生成，未修改 testbench、未添加自定义
断言、未人工改写 failure marker。20/20 case 通过标准失败门槛；validator 为
0 errors / 0 warnings；无路径或 bug 标签泄漏、无重复三日志组合。对应根因和
审计分别见 `interrupt_state_supplement_root_causes.csv` 与
`interrupt_state_supplement_20260827_audit.json`。

## 20. ALU/Decode 条件触发补充 v2（2026-08-27）

新增独立包 `alu_decode_supplement_20260827_v2.tar.gz`，共 46 cases：
`bug_255` 16 例、`bug_256` 16 例、`bug_257` 14 例。三个细粒度根因分别
覆盖特定位模式下的 `andi` AND->OR、`slli` SLL->SRL 和 `sltiu` SLTU->SLT
译码错误。三者均覆盖 `riscv_machine_mode_rand_test`、
`riscv_rand_instr_test`、`riscv_rv32im_instr_test`，每个测试也覆盖全部三个
bug，避免测试名捷径。

该包首个 mismatch index 为 103--6882，0 个 <=32；46/46 case 有标准
`[FAILED]` marker，validator 为 0 errors / 0 warnings；无路径或 bug 标签
泄漏、无重复三日志组合。clean run 已排除。根因和审计见
`alu_decode_supplement_20260827_v2_root_causes.csv` 与
`alu_decode_supplement_20260827_v2_audit.json`。

## 21. MDU 条件触发补充 v3（2026-08-27）

新增独立包 `mdu_supplement_20260827_v3.tar.gz`，共 16 cases：
`bug_261` 5 例、`bug_262` 6 例、`bug_263` 5 例。三个 MDU 根因分别覆盖
条件触发的 `mulh/mulhsu`、`div/divu` 和 `rem/remu` signed-mode 译码路径；
每个 bug 均覆盖 `riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，避免测试名一对一捷径。

该包使用标准 VCS/UVM 与 riscv-dv 流程重新生成，修复了比较器尾部路径，
因此每个有效 `regr.log` 只保留一个 `Mismatch[1]`。首个 mismatch index
为 116--7006，0 个 <=32；validator 为 0 errors / 0 warnings；无日志泄漏、
无重复三日志组合。根因和审计见
`mdu_supplement_20260827_v3_root_causes.csv` 与
`mdu_supplement_20260827_v3_audit.json`。

## 22. RV32C 条件触发补充（2026-08-27）

新增独立包 `rvc_supplement_20260827.tar.gz`，共 10 cases：
`bug_264` 2 例、`bug_265` 4 例、`bug_266` 4 例。三个 RV32C 根因分别为
条件触发的 `c.addi` 立即数符号扩展、`c.andi` 运算译码和 `c.and` 运算译码
错误；每个 bug 均覆盖 `riscv_machine_mode_rand_test` 与
`riscv_rand_instr_test`，两个 test 也都覆盖三个 bug。

该包使用标准 VCS/UVM 与 riscv-dv 流程生成；18 个源 run 中 2 个重型
`riscv_rv32im_instr_test` 超时并排除。保留 case 的每个 `regr.log` 只有一个
`Mismatch[1]`，首个 mismatch index 为 164--486，0 个 <=32；validator 为
0 errors / 0 warnings；无日志泄漏、无重复三日志组合。根因和审计见
`rvc_supplement_20260827_root_causes.csv` 与
`rvc_supplement_20260827_audit.json`。

## 23. 流水线/多周期控制补充（2026-08-27）

新增独立包 `pipeline_supplement_20260827.tar.gz`，共 10 cases：
`bug_268` 4 例、`bug_269` 6 例。两个可观测根因分别为条件触发的 LSU
stall 缺失和 mult/div stall 缺失；两个 bug 均覆盖
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，三个测试也都覆盖两个 bug。

该包严格使用标准 VCS/UVM 与 riscv-dv 流程。4 个 case 由标准 mem model
`UVM_ERROR` 暴露，6 个 case 为 trace mismatch；mismatch case 均只有一个
`Mismatch[1]`，首个 mismatch index 为 206--1094，0 个 <=32；validator 为
0 errors / 0 warnings；无日志泄漏、无重复三日志组合。branch stall 候选
`bug_267` 的 6 个 run 全部超时，已淘汰。根因和审计见
`pipeline_supplement_20260827_root_causes.csv` 与
`pipeline_supplement_20260827_audit.json`。

## 24. CSR 计数器候选复核（2026-08-27）

`bug_270`（minstret 压缩指令计数）、`bug_271`（条件 mcycle 增量）和
`bug_272`（mcountinhibit 位映射）均完成 3 个标准 CSR/中断相关 test 的
探针运行，9/9 为 clean。当前官方 testbench 未观测到这些计数器差异，因而
不进入正式数据包，也不添加人工 failure marker。

## 25. Debug 条件触发补充（2026-08-27）

新增独立包 `debug_supplement_20260827.tar.gz`，共 6 cases：
`bug_273` 3 例、`bug_276` 3 例。两个 debug 根因分别为外部 debug request
保存错误的 `dcsr.cause`，以及 debug 进入时保存错误的 `dcsr.prv`。
两者均覆盖 `riscv_debug_csr_entry_test`、`riscv_debug_ebreak_test` 和
`riscv_debug_ebreakmu_test`，每个 test 也同时覆盖两个 bug，避免测试名捷径。

该包严格使用标准 VCS/UVM 与 riscv-dv 流程；6/6 case 均由标准 UVM
`UVM_FATAL` 触发，validator 为 0 errors / 0 warnings，无重复三日志组合，
无日志泄漏。此前 `bug_274`（ebreakm 入口）与 `bug_275`（depc 偏移）在
探针 test 中均为 clean，未纳入正式包。根因和审计见
`debug_supplement_20260827_root_causes.csv` 与
`debug_supplement_20260827_audit.json`。

## 28. Decode 条件触发补充 v2（2026-08-27）

新增独立包 `decode_supplement_20260827_v2.tar.gz`，共 6 cases：
`bug_284` 3 例、`bug_285` 3 例。两个译码根因分别为指定 PC 区域内
XOR->OR 和 OR->XOR 的 ALU 控制选择错误；两者均覆盖
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，每个 test 也覆盖两个 bug。

该包严格使用标准 VCS/UVM 与 riscv-dv 流程。6/6 case 均为单个
`Mismatch[1]`，首个 mismatch index 为 212--4068，0 个 <=32；validator
为 0 errors / 0 warnings，无重复三日志组合或日志泄漏。首轮缺失 test
的 clean 结果未纳入；补跑后才满足双向 test 覆盖门槛。根因和审计见
`decode_supplement_20260827_v2_root_causes.csv` 与
`decode_supplement_20260827_v2_audit.json`。

## 27. MDU 除法/余数条件触发补充 v4（2026-08-27）

新增独立包 `mdu_supplement_20260827_v4.tar.gz`，共 6 cases：
`bug_282` 3 例、`bug_283` 3 例。两个 MDU 根因分别为指定 PC 区域内
DIV->REM 和 REM->DIV 的操作码译码错误；两者均覆盖
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，每个 test 也覆盖两个 bug。

该包严格使用标准 VCS/UVM 与 riscv-dv 流程。6/6 case 均为单个
`Mismatch[1]`，首个 mismatch index 为 264--2103，0 个 <=32；validator
为 0 errors / 0 warnings，无重复三日志组合或日志泄漏。根因和审计见
`mdu_supplement_20260827_v4_root_causes.csv` 与
`mdu_supplement_20260827_v4_audit.json`。

## 26. ALU 条件触发补充（2026-08-27）

新增独立包 `alu_supplement_20260827.tar.gz`，共 6 cases：
`bug_280` 3 例、`bug_281` 3 例。两个 ALU 根因分别为 PC 区域内 SRA
移位操作数错误和 SLTU 操作数错误；两者均覆盖
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，每个 test 也覆盖两个 bug。

该包严格使用标准 VCS/UVM 与 riscv-dv 流程。6/6 case 均为单个
`Mismatch[1]`，首个 mismatch index 为 195--1359，0 个 <=32；validator
为 0 errors / 0 warnings，无重复三日志组合或日志泄漏。操作数模式候选
`bug_277/278` 为 clean，PC 区域 ADD 候选 `bug_279` 产生未初始化内存连锁
并被淘汰。根因和审计见 `alu_supplement_20260827_root_causes.csv` 与
`alu_supplement_20260827_audit.json`。

## 29. Coverage64 官方候选 v5（2026-08-27）

新增独立交付包 `coverage64_release_20260827_v5.tar.gz`，包含 931 个
official-layout case 与 64 个 bucket。该版本从已审计的 coverage64 v4
中移除会造成单测试/单 bug 捷径的根因与测试，再加入 12 个已独立验证的
寄存器/写回、流水线、多源中断、debug、ALU、MDU 根因；旧日志只复制，未
改写 `regr.log`、`sim.log.gz` 或 `trace.log.gz`。

全量门禁结果：官方 validator 为 0 errors / 0 warnings；931/931 case
均有标准 failure marker；769 个 mismatch case 均只有一个 `Mismatch[1]`，
首个 mismatch index 为 37--6820 且没有 <=32；其余 162 个为标准 UVM-only
failure；无 bug 标签或宿主机路径泄漏，无跨 bug 三日志重复；每个 bug 至少
覆盖两个 test、每个 test 至少覆盖两个 bug；最长日志 4,179,017 行，低于
10M 限制。

配套文件为 `coverage64_release_20260827_v5_audit.json` 与
`coverage64_release_20260827_v5_root_cause_map.csv`。PMP、AHB-Lite、RV32B
仍未人为填充，因为当前配置和标准 testbench 没有稳定合法 oracle。

## 30. Interrupt/CSR 独立补充 v2（2026-08-27）

新增独立交付包 `interrupt_csr_supplement_20260827_v2.tar.gz`，共 6 个
official-layout case、2 个 bucket：`bug_289` 和 `bug_290` 各 3 例。
`bug_289` 为异常入口未清除 `mstatus.MIE`，`bug_290` 为异常入口反转
`mcause` 低位 cause bit。两个 bug 均覆盖
`riscv_interrupt_csr_test`、`riscv_multiple_interrupt_test` 和
`riscv_single_interrupt_test`，避免 test-to-bug 一对一捷径。

6/6 case 均由标准 UVM failure 触发，不含 mismatch block；官方布局验证器
为 0 errors / 0 warnings，最长日志 250 行，无 bug 标签、worker 临时路径
泄漏或跨 bucket 三日志重复。根因、审计和校验值分别见
`interrupt_csr_supplement_20260827_v2_root_causes.csv`、
`interrupt_csr_supplement_20260827_v2_audit.json` 和
`interrupt_csr_supplement_20260827_v2.tar.gz.sha256`。

本轮同时修复了 collector 对 `/home/<user>/iccad/worker_*/ibex` 临时
工作树路径的规范化规则；规范化后重新收集并验证，未改写 simulator 原始
日志，也未添加自定义断言或人工 failure marker。

## 31. Branch/Jump 独立补充 v1（2026-08-27）

新增独立交付包 `branch_supplement_20260827_v1.tar.gz`，共 6 个
official-layout case、2 个 bucket：`bug_291` 和 `bug_292` 各 3 例。
两个根因分别是在选定分支立即数模式下将 BLT 选择为 BGE、将 BLTU
选择为 BGEU。两个 bug 均覆盖 `riscv_rand_jump_test`、
`riscv_jump_stress_test` 和 `riscv_machine_mode_rand_test`，避免
test-to-bug 一对一捷径。

6/6 case 均为标准单个 `Mismatch[1]`；首个 mismatch index 为 232--1521，
没有 <=32 的启动级分歧。官方布局验证器为 0 errors / 0 warnings，最长日志
8595 行，无 bug 标签、worker 临时路径泄漏或跨 bucket 三日志重复。根因、
审计和校验值分别见 `branch_supplement_20260827_v1_root_causes.csv`、
`branch_supplement_20260827_v1_audit.json` 和
`branch_supplement_20260827_v1.tar.gz.sha256`。

## 32. CSR 读回候选复核（2026-08-27）

`bug_293/294`（MPRV/TW 读回交叉）在 6 个标准运行中全部 clean，未收录。
`bug_295`（MPIE 读回使用 MIE）与历史 `bug_193` 根因和标准症状重复，
`bug_296`（MPP 读回固定为 U-mode）与历史 MPP 读回候选重叠，均淘汰。
`bug_297/298`（MSCRATCH/MTVAL 读回交叉）初始 seed 产生了不稳定结果，
额外 seed 后分别出现 clean；其中 CSR directed 的 `CSR TEST_FAILED!`
文本也不足以形成稳定的跨 bug 消歧签名，因此不进入正式数据包。

这些探针均未修改 testbench、未添加自定义断言，也未改写 simulator 日志；
淘汰遵循 clean、不稳定或重复 syndrome 不收录的门槛。

## 33. WFI/TW 特权探针复核（2026-08-27）

`bug_299`（移除 U-mode WFI 的 `mstatus.TW` 陷阱条件）和 `bug_300`
（将该条件反转为 `~TW`）分别运行
`riscv_umode_tw_test`、`riscv_interrupt_wfi_test` 和
`riscv_debug_wfi_test`，共 6 个标准 VCS/UVM oracle 运行，结果全部为
clean：`ok=6`、`run_failed=0`、`patch_failed=0`。对应 `regr.log` 均为
`[PASSED]`，`sim.log` 均为标准 `RISC-V UVM TEST PASSED`，没有可观测
failure marker，因此不打包、不进入正式 bug 集。

本轮仅验证了官方 RTL 条件，不改 testbench、不添加自定义断言、不加工日志；
该缺口在当前 test/config 下不可稳定观测，按门槛淘汰。

## 34. Register-file late writeback supplement v1（2026-08-27）

新增独立包 `register_file_supplement_20260827_v1.tar.gz`，共 6 个
official-layout case、2 个根因：`bug_304` 为后段 PC 区域内 x15 写回
bit 0 翻转，`bug_305` 为同一区域内 x11 写回 bit 0 翻转。两个 bug
均覆盖 `riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，每个 test 也覆盖两个 bug。

6/6 case 均为标准单个 `Mismatch[1]`；首个 mismatch index 为 114--1720，
没有 <=32 的启动级分歧。官方布局验证器为 0 errors / 0 warnings；无缺失
日志、bug 标签或 worker 路径泄漏、跨 bug 三日志重复，最长日志 20,258 行。
根因、审计和校验值分别见 `register_file_supplement_20260827_v1_root_causes.csv`、
`register_file_supplement_20260827_v1_audit.json` 和
`register_file_supplement_20260827_v1.tar.gz.sha256`。

同轮的 x10 变体（`bug_301`、`bug_303`）分别因首分歧 <=32 或三个 test
全部 clean 而淘汰；未将其混入正式包。

## 35. LSU byte sign-extension supplement v1（2026-08-27）

新增独立包 `lsu_supplement_20260827_v1.tar.gz`，共 4 个 official-layout
case、2 个根因：`bug_306` 为 offset 1 的 signed byte load 使用错误符号位，
`bug_307` 为 offset 2 的 signed byte load 使用错误符号位。两个 bug 均覆盖
`riscv_mmu_stress_test` 和 `riscv_unaligned_load_store_test`，两个 test 也
各覆盖两个 bug。

4/4 case 均为标准单个 `Mismatch[1]`；首个 mismatch index 为 71--185，
没有 <=32 的启动级分歧。官方布局验证器为 0 errors / 0 warnings；无缺失
日志、bug 标签或 worker 路径泄漏、跨 bug 三日志重复，最长日志 49,514 行。
根因、审计和校验值分别见 `lsu_supplement_20260827_v1_root_causes.csv`、
`lsu_supplement_20260827_v1_audit.json` 和
`lsu_supplement_20260827_v1.tar.gz.sha256`。

同轮 `riscv_machine_mode_rand_test` 的两个 clean 运行未收录；没有为凑数量
添加人工 failure marker 或修改 testbench。

## 36. LSU byte sign-extension supplement v2（2026-08-27）

新增独立包 `lsu_supplement_20260827_v2.tar.gz`，共 4 个 official-layout
case、2 个根因：`bug_308` 为 offset 0 的 signed byte load 使用错误符号位，
`bug_309` 为 offset 3 的 signed byte load 使用错误符号位。两个 bug 均覆盖
`riscv_mmu_stress_test` 和 `riscv_unaligned_load_store_test`，两个 test 也
各覆盖两个 bug。

4/4 case 均为标准单个 `Mismatch[1]`；首个 mismatch index 为 106--6527，
没有 <=32 的启动级分歧。官方布局验证器为 0 errors / 0 warnings；无缺失
日志、bug 标签或 worker 路径泄漏、跨 bug 三日志重复，最长日志 99,709 行。
根因、审计和校验值分别见 `lsu_supplement_20260827_v2_root_causes.csv`、
`lsu_supplement_20260827_v2_audit.json` 和
`lsu_supplement_20260827_v2.tar.gz.sha256`。

初次 100 秒 probe 中两个 unaligned test 超时；将探针 timeout 提高到 180
秒后完整成功，最终包只使用完整成功的标准 oracle 结果。

## 37. MDU divide-by-zero supplement v1（2026-08-27）

新增独立包 `mdu_zero_supplement_20260827_v1.tar.gz`，共 6 个
official-layout case、2 个根因：`bug_310` 为 DIV 除零结果错误返回 0，
`bug_311` 为 REM 除零结果错误返回 0。两个 bug 均覆盖
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，每个 test 也覆盖两个 bug。

6/6 case 均为标准单个 `Mismatch[1]`；首个 mismatch index 为 61--291，
没有 <=32 的启动级分歧。官方布局验证器为 0 errors / 0 warnings；无缺失
日志、bug 标签或 worker 路径泄漏、跨 bug 三日志重复，最长日志 8,464 行。
根因、审计和校验值分别见 `mdu_zero_supplement_20260827_v1_root_causes.csv`、
`mdu_zero_supplement_20260827_v1_audit.json` 和
`mdu_zero_supplement_20260827_v1.tar.gz.sha256`。

## 38. RV32C compressed-decode supplement v1（2026-08-27）

新增独立包 `rvc_supplement_20260827_v1.tar.gz`，共 4 个 official-layout
case、2 个根因：`bug_312` 为 c.slli/x5 解压后的 shamt bit 0 翻转，
`bug_313` 为 c.srli/x15 解压后的 shamt bit 0 翻转。两个 bug 均覆盖
`riscv_machine_mode_rand_test` 和 `riscv_rand_instr_test`，每个 test 也
覆盖两个 bug。

4/4 case 均为标准单个 `Mismatch[1]`；首个 mismatch index 为 128--1111，
没有 <=32 的启动级分歧。官方布局验证器为 0 errors / 0 warnings；无缺失
日志、bug 标签或 worker 路径泄漏、跨 bug 三日志重复，最长日志 2,208 行。
根因、审计和校验值分别见 `rvc_supplement_20260827_v1_root_causes.csv`、
`rvc_supplement_20260827_v1_audit.json` 和
`rvc_supplement_20260827_v1.tar.gz.sha256`。

初始 c.slli/x10 变体因两个 test 均 clean 而淘汰，未混入正式包。

## 39. Branch/Jump target supplement v1（2026-08-27）

新增独立包 `jump_target_supplement_20260827_v1.tar.gz`，共 4 个
official-layout case、2 个根因：`bug_315` 在选定的后段跳转目标区域翻转
PC target bit 1，`bug_316` 在相同区域翻转 bit 2。两个 bug 均覆盖
`riscv_rand_jump_test` 和 `riscv_machine_mode_rand_test`，两个 test 也均覆盖
两个 bug。

4/4 case 均为标准单个 `Mismatch[1]`；首个 mismatch index 为 243--1571，
没有 <=32 的启动级分歧。官方布局验证器为 0 errors / 0 warnings；无缺失
日志、bug 标签或 worker 路径泄漏、跨 bug 三日志重复，最长日志 9,474 行。

`bug_315` 的 machine-mode 补跑已成功；初始 jump probe 中的 timeout/clean
运行没有收录。根因、审计和校验值分别见
`jump_target_supplement_20260827_v1_root_causes.csv`、
`jump_target_supplement_20260827_v1_audit.json` 和
`jump_target_supplement_20260827_v1.tar.gz.sha256`。

## 40. Coverage64 v5 current-state reconciliation（2026-08-27）

核对 `coverage64_release_20260827_v5.tar.gz` 后，当前包实际包含 931 个
case、64 个 distinct bug；旧的 `COVERAGE64_RELEASE_README.md` 描述的是
早期 v2（981 case / 65 bug），不能作为 v5 的说明。

v5 审计结果为：官方布局验证 0 errors / 0 warnings，769 个 mismatch case、
162 个标准 UVM failure case，首个 mismatch index 为 37--6820，<=32 的
case 为 0；无缺失 failure marker、bug 标签或宿主机路径泄漏，无跨 bug
三日志重复，最长日志 4,179,017 行。64 个 bug 均至少覆盖两个 test，且
每个选用 test 至少覆盖两个 bug。

当前已验证的可观测域包括 ALU、Branch/Jump、LSU、MDU、RV32C/decode、
decode、register-file/writeback、CSR、中断/异常、Debug 和
pipeline/control。PMP、AHB-Lite、RV32B 在当前配置中没有稳定、无冲突的
标准 oracle 结果，因此继续不人工伪造这些域。

## 41. Low-count ALU/MDU balance probes（2026-08-27）

针对 v5 中样本较少的 `bug_280/281` 和 `bug_282/283`，各使用一个新 seed
运行三个共享的标准测试。两组均为 `6/6` runner `ok`，没有环境、授权或
patch 失败；但 failure-only collector 分别只保留了 ALU 的 4 个 case
（`bug_280` 仅 1 个有效测试）和 MDU 的 2 个 case（每个 bug 仅 1 个有效
测试）。

这些 case 虽然各自通过 official-layout validator，且没有泄漏、重复或
启动级分歧，但不满足独立补充集的交叉测试门槛，故全部作为探针淘汰，未
打包、未回传，也未混入 `coverage64_release_20260827_v5`。这验证了低计数
bug 不应为了均衡而放松消歧和交叉覆盖要求。

## 42. ALU conditional operand supplement v3（2026-08-27）

新增独立包 `alu_supplement_20260827_v3.tar.gz`，共 7 个
official-layout case、2 个根因：`bug_317` 在 PC region 4 的选定 XOR
操作使用错误的 operand-B 位模式，`bug_318` 在同一区域的选定 OR 操作
使用错误的 operand-B 位模式。两个 bug 均覆盖
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，三个 test 也均覆盖两个 bug。

7/7 case 均为标准单个 `Mismatch[1]`；首个 mismatch index 为 158--4378，
没有 <=32 的启动级分歧。官方布局验证器为 0 errors / 0 warnings；无缺失
日志、bug 标签或 worker 路径泄漏、跨 bug 三日志重复，最长日志 15,620 行。

首轮 license daemon 退出导致的 6 个 run_failed 已删除并不计入统计；恢复
合法 license 服务后重跑得到最终结果。根因、审计和校验值分别见
`alu_supplement_20260827_v3_root_causes.csv`、
`alu_supplement_20260827_v3_audit.json` 和
`alu_supplement_20260827_v3.tar.gz.sha256`。

## 43. MDU conditional signed-mode supplement v6（2026-08-27）

新增独立包 `mdu_supplement_20260827_v6.tar.gz`，共 7 个 official-layout
case、2 个根因：`bug_319` 在 PC region 6 的选定 `mulhsu` 操作使用错误的
signed-mode，`bug_320` 在 PC region 7 的选定 `mulhu` 操作使用错误的
signed-mode。两个 bug 均覆盖 `riscv_machine_mode_rand_test`、
`riscv_rand_instr_test` 和 `riscv_rv32im_instr_test`，三个 test 也均覆盖
两个 bug。

7/7 case 均为标准单个 `Mismatch[1]`；首个 mismatch index 为 211--4967，
没有 <=32 的启动级分歧。官方布局验证器为 0 errors / 0 warnings；无缺失
日志、bug 标签或 worker 路径泄漏、跨 bug 三日志重复，最长日志 14,682 行。

根因、审计和校验值分别见 `mdu_supplement_20260827_v6_root_causes.csv`、
`mdu_supplement_20260827_v6_audit.json` 和
`mdu_supplement_20260827_v6.tar.gz.sha256`。

## 44. Remaining-domain observability check（2026-08-27）

进一步核对 official Ibex/UVM 工作树：`rtl/ibex_core.sv` 的默认
`PMPEnable` 为 `1'b0`，UVM 回归路径没有覆盖该参数；testlist 中也没有
PMP、RV32B/bitmanip 或 AHB-Lite 测试入口。因而在不改 testbench、不改
配置、不人为加工日志的约束下，这三个域没有可交付的标准 oracle 观察面。

这不是生成失败，而是流程边界：不能为了声称 12 域覆盖而启用非官方配置
或添加自定义断言。当前有效交付继续以 64-bug 主候选和独立的 ALU、MDU、
Branch/Jump、LSU、RVC、寄存器堆等补充包为准。

## 45. Debug cause probes（2026-08-27）

复核了三种标准 Debug cause 注入：`bug_321` 将 debug request cause 改为
`EBREAK`，`bug_322` 改为 `TRIGGER`，`bug_323` 改为 `STEP`。所有 runner
均无环境或 patch 失败，但 failure-only collector 结果分别只覆盖部分
Debug 测试：`bug_322` 的 basic test 两个补跑均 clean，`bug_323` 仅
`riscv_debug_ebreak_test` 触发。

由于不能满足每个 bug 至少两个测试、每个测试至少两个 bug 的交叉覆盖，
且没有理由人工添加 failure marker，这些 probe 全部淘汰，不打包、不混入
主集。已有稳定 Debug 根因继续使用 v5 和既有独立补充包中的结果。

## 46. LSU balance supplement (2026-08-27)

针对主候选中样本偏少的 `bug_051/052`，按官方标准 VCS/UVM/Spike 流程补跑
了新 seed。首轮 12 个组合得到 10 个 `ok`，其中收集到 `bug_051` 2 例、
`bug_052` 3 例；第二轮 12 个组合得到 11 个 `ok`，新增 `bug_051` 4 例，
其余 `machine_mode` 结果为 clean。失败运行和 clean 运行均未进入数据集。

将两轮真实 failure 合并后，形成独立包
`p0_lsu_balance_supplement_20260827_v2.tar.gz`，共 9 例：`bug_051=6`、
`bug_052=3`。两个 bug 均覆盖 `riscv_unaligned_load_store_test` 和
`riscv_mmu_stress_test`，两个 test 也均覆盖两个 bug。

该包通过 official-layout validator（0 errors / 0 warnings）和完整审计：
9/9 为单个 `Mismatch[1]`，首个 mismatch index 为 66--610，均大于 32；
无缺失 failure marker、路径或 bug 标签泄漏、坏 case、三日志重复。压缩包
包内同时包含根因清单、完整审计和 signature audit；v2 SHA256 为
`e8f3712111ae8a784bf92abaaaa5b3e51bd6ce3eef561d2a2abeadbc94ba7a79`。

## 47. ALU/Branch balance supplement (2026-08-27)

针对主候选中样本偏少的 `bug_045/047`，首轮 12 个组合因 SCL daemon
重启后未恢复而全部在 VCS 编译阶段失败；该批结果确认是环境失败，未计入
数据。恢复合法 license 服务后，使用相同清单重跑得到 12/12 `ok`。

收集到 9 个真实 failure：`bug_045=4`、`bug_047=5`。其中
`riscv_arithmetic_basic_test` 只触发 `bug_045`，`riscv_jump_stress_test`
只触发 `bug_047`，为避免 test-name shortcut 将其剔除，最终独立包
`p0_alu_branch_balance_supplement_20260827.tar.gz` 保留 6 例：
`bug_045=2`、`bug_047=4`。两个 bug 均覆盖
`riscv_machine_mode_rand_test` 和 `riscv_rand_instr_test`，两个 test 也
均覆盖两个 bug。

该包通过 official-layout validator（0 errors / 0 warnings）、完整审计和
signature audit：6/6 为单个 `Mismatch[1]`，首个 mismatch index 为
173--3073，均大于 32；无缺失 failure marker、坏 case、泄漏、三日志重复，
跨 bug signature collision 为 0。最终包 SHA256 为
`a036e176de8fedfd697a07bdc07d004c1496e04d62f045bbfdf189d10b2c9c7a`。

## 48. ALU balance supplement 048/049 (2026-08-27)

继续对主候选中的低样本 ALU 根因 `bug_048/049` 做补跑。两个根因各使用
`riscv_machine_mode_rand_test` 和 `riscv_rand_instr_test`，共 12 个组合，
12/12 `ok`，无 patch 或环境失败。

failure-only collector 保留 8 例：`bug_048=3`、`bug_049=5`。两个 bug
与两个 test 均双向覆盖，因此没有 test-name shortcut。最终独立包
`p0_alu_balance_048_049_20260827.tar.gz` 通过 official-layout validator、
完整审计和 signature audit：8/8 为单个 `Mismatch[1]`，首个 mismatch
index 为 208--1786，均大于 32；无 failure marker 缺失、坏 case、泄漏、
三日志重复，跨 bug signature collision 为 0。包 SHA256 为
`0a82c463e905f31309a67cc60b01ae23d6836a427ef046b9daa0fc3af845b3ab`。

## 49. Register-file balance supplement 223/224 (2026-08-27)

针对 v5 中样本偏少的寄存器堆根因 `bug_223/224`，按官方标准 VCS/UVM/Spike
流程补跑 18 个组合，18/18 runner `ok`。完整收集结果中有 9 个 case 的首个
mismatch 位于 27、30 或 32；按当前 late-mismatch 硬门槛将其排除，未人为
修改日志或 failure marker。

最终独立包 `p0_regfile_balance_late_20260827.tar.gz` 保留 9 例：
`bug_223=6`、`bug_224=3`。两个 bug 均覆盖
`riscv_machine_mode_rand_test`、`riscv_rand_instr_test` 和
`riscv_rv32im_instr_test`，三个 test 也均覆盖两个 bug。

该包通过 official-layout validator（0 errors / 0 warnings）、完整审计和
signature audit：9/9 为单个 `Mismatch[1]`，首个 mismatch index 为 68--176，
均大于 32；无缺失 failure marker、坏 case、泄漏、三日志重复或跨 bug
signature collision。压缩包 SHA256 为
`c423b8f4332b861c8106bbc887dfde9479634542a85cb44b3de3496cdbfe1b9`。

## 50. Debug balance supplement 273/276 (2026-08-27)

针对 v5 中样本偏少的 Debug 根因 `bug_273/276`，按官方标准 VCS/UVM/Spike
流程补跑 12 个组合，12/12 runner `ok`。最终独立包
`p0_debug_balance_273_276_20260827.tar.gz` 共 12 例：`bug_273=6`、
`bug_276=6`。

两个 bug 均覆盖 `riscv_debug_csr_entry_test`、`riscv_debug_ebreak_test` 和
`riscv_debug_ebreakmu_test`，三个 test 也均覆盖两个 bug。12/12 是标准 UVM
failure-only case（无 mismatch marker），验证器为 0 errors / 0 warnings，
无缺失 failure marker、坏 case、泄漏、三日志重复或跨 bug signature collision。
根因、审计和校验材料随包提供；压缩包 SHA256 为
`02107034099321b305825a23d7897583b2b7955dd821d2a78c7b33b9c7ff37c4`。
