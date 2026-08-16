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
