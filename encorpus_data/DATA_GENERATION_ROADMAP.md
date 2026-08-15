# 造数据路线图：下一批该造什么、怎么造

> 给数据生成的同学。基于：官方 Q&A（B_QA_20260612/20260727）、`official_bug_injection.md`、
> `official_bug_distribution.md`、`bug_type_taxonomy.md`、`fake_vs_official_gap.md`，以及我们
> 对 benchmark5_500cases / benchmark8_500cases 两批数据的独立质量审计与模型实测结论。
> 已有两批数据的反馈见 `NEXT_BATCH_FEEDBACK.md`；本文聚焦**下一批的选题与工程细节**。

## 0. 背景：官方 hidden 集到底长什么样（决定造什么）

官方 10 个 benchmark（B_20260601 §3.2）：N=2~3000、k=2~64，其中 k 序列为 **2/4/8/16/32/64**，
且最大两个 benchmark（N=3000）是 **k=64**。QA A11 明确 buckets 数 = bug 数，QA A12 明确 --k 按表原值传入。

我们当前**干净**的 fake 数据覆盖：k=3（official_vcs）、k=4（stable）、k=10（directed_cross_v4 +
benchmark5_500 + benchmark8_500，共 20 个 distinct bug）。旧 benchmark6_final（64 bug）和
benchmark5_final（32 bug）是 lui 级联的**歧义数据**，不能当高-k 预演。

**结论：我们缺一个「干净的高-k 数据集」来预演 hidden 集的 k=32/64 场景。**

## 1. P0（最高优先级）：造一个干净的 k=32 或 k=64 数据集

- 目标规格：k=32（benchmark5 规格）起步，最好 k=64（benchmark6 规格），N 按 1000~3000。
- 硬要求（和 benchmark5/8 一样）：
  1. **late first-mismatch**（首个 mismatch index > 32，杜绝 cycle-0 lui 级联）
  2. **每 bug 症状可区分**（见 §3 消歧）
  3. `mismatch_print_limit=1`，regr.log 只有单个 `Mismatch[1]`
  4. 无 bug label / 真实绝对路径泄漏
- 为什么重要：64-bug 分簇（TPR）是我们模型当前唯一没验证过的规模，也是 hidden 集里最难的
  benchmark。没有干净高-k 数据，就无法知道模型在 64 类上会不会崩。

## 2. P1：补全 bug 类型覆盖（按缺口排序）

已覆盖（benchmark5+8 共 20 bug）：ALU 算数/逻辑/移位、MDU 的 mul/mulhsu、LSU 符号扩展与
store 地址、分支比较器与立即数编码、AUIPC、CSR 的 MIE、中断向量与原因、debug ebreak/入口。

**下一批优先补这些（官方有对应证据的标了官方 bug 号）：**

| 缺口 | 具体 bug 建议 | 症状 |
|---|---|---|
| **MDU 除/余** | div/divu/rem/remu 错误、除零处理 | mismatch（对应指令）——官方 bug_107 = remu/mulhsu/rem |
| **RV32C 译码** | 压缩指令扩展成 32 位错误、c.* 立即数/寄存器字段错 | mismatch——官方 bug_304 = c.addi |
| **分支 beq/bne/blt** | 等号/不等号条件错、无符号比较错 | mismatch（PC 分歧）|
| **LSU byte enable/对齐** | 部分写使能错、非对齐访存、符号/零扩展选择 | mismatch 或 mem_error |
| **CSR 读/写/权限** | 具体 CSR（mepc/mtvec/mtval/mie/mip…）读写错、用户态访问特权 CSR 越权 | test_fail |
| **PMP（物理内存保护）** | 权限检查错、地址匹配错 | test_fail |
| **WFI/睡眠** | WFI 唤醒逻辑错（中断未唤醒/错误唤醒）| test_fail |
| **总线接口（AHB-Lite）** | 握手/突发传输错 | mismatch / 时序错 |
| **复位/初始化** | 复位值错、初始化状态错 | 早期 mismatch（注意别退化成 cascade）|

参考 `bug_type_taxonomy.md` 的 A1~A8 / B1~B4 全景表。

## 3. P2：消歧铁律系统化（每加一个 bug 都要跑）

官方 Q5 明确剔除了 diff-bugs-same-syndrome。造数据必须保证**每个 bug 在至少一个可观测维度上有
唯一、稳定的症状签名**：

- mismatch 类：首个 mismatch 的 **opcode 家族 + 操作数模式 + PC 区域**。
- test_fail 类：**UVM 断言的文本 + file:line + 检查的具体值**。

**两条硬规则：**
1. 每个 bug 造完后，和已有所有 bug 的签名比对；若有 bug 签名完全重叠 → 删掉其中一个（消歧）。
2. **test_fail 类别再用通用断言**。benchmark8 的 bug_083/085/102 三个 bug 全报同一个
   `read to uninitialized addr`（mem_model.sv:28），只差地址——这是软性同症状。改用**具体断言**
   （例如 `Check failed mcause[...] == X`、`Check failed xN == expected`），让不同 bug 的 fatal
   消息本身就可区分。

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

**下一批最重要的不是再多几个 k=10 的小集，而是冲一个「干净的 k=32/64」高-k 集**，把 MDU 除余、
RV32C 译码、分支、PMP 这几块缺口补上，同时把"消歧 + 具体断言"做成生成脚本里的硬门槛。
