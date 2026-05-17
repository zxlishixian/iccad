# Ibex 数据集生成交接手册

本文档用于交接 Ibex 回归失败聚类数据集的制作流程。目标是让接手同学在同一台虚拟机上，从开机、检查环境、运行造数脚本，到把数据集打包传回物理机，都能按步骤完成。

## 1. 当前目标

我们要为 ICCAD 回归失败聚类任务生成数据。一次标准造数流程包括：

- 对 Ibex 源码应用一个 bug patch。
- 使用 Ibex DV 和 Xcelium 运行随机测试。
- 生成每个 case 的失败日志，包括 `sim.log`、`regr.log`、`trace.log`。
- 汇总为标准数据集目录，包括 `input.csv`、`gold.csv`、`meta.csv` 和 `cases/case_xxxxxx/`。
- 将最终数据集压缩包通过 VMware 共享文件夹传回物理机。

当前已经验证过的 smoke 数据集是：

```text
~/iccad/dataset/bug001_smoke_dataset
```

对应物理机上已经检查过的压缩包是：

```text
D:\ICCAD\bug001_smoke_dataset.tar.gz
```

## 2. 虚拟机和工具版本

建议优先使用已经装好的 VMware 虚拟机，不建议回到旧服务器的 Xcelium 18.03 环境。旧环境主要卡在 DPI C++、Spike、Boost 和 libstdc++ 兼容性上，继续修会反复踩坑。

当前推荐环境如下：

```text
系统用户: meow
Ibex 仓库: ~/iccad/repos/ibex
造数脚本: ~/iccad/ibex_dataset_tools
patch 目录: ~/iccad/bug_patches
数据输出: ~/iccad/dataset
Xcelium: /opt/cadence/XCELIUM2309/tools/bin/xrun
Xcelium 版本: 23.09-s003
Spike: ~/iccad/tools/riscv-isa-sim/bin/spike
Conda 环境: ibex
```

推荐虚拟机资源：

```text
最低可用: 8 核 CPU，16 GB 内存，100 GB 可用磁盘
推荐配置: 12 核 CPU，32 GB 内存，200 GB 可用磁盘
更舒服配置: 16 核 CPU，48-64 GB 内存，300-500 GB 可用磁盘
```

优先级是磁盘大于内存大于 CPU。造数据会产生大量 `out/`、日志和波形相关中间文件，磁盘不够比 CPU 少更容易卡住。

## 3. 启动虚拟机

启动 VMware 虚拟机后，登录用户：

```text
用户名: meow
```

进入终端后先确认当前机器状态：

```bash
whoami
nproc
free -h
df -h
```

确认网络是否正常：

```bash
ping -c 2 8.8.8.8
ping -c 2 github.com
ping -c 2 mirrors.aliyun.com
```

如果 `8.8.8.8` 不通，通常是网卡没有连上。执行：

```bash
nmcli device status
sudo nmcli device connect ens33
```

如果 IP 能 ping 通，但域名不能解析，临时写入 DNS：

```bash
printf "nameserver 223.5.5.5\nnameserver 114.114.114.114\nnameserver 8.8.8.8\n" | sudo tee /etc/resolv.conf
```

再次检查：

```bash
ping -c 2 github.com
ping -c 2 mirrors.aliyun.com
```

## 4. 检查 Cadence 和许可证

确认 Xcelium 能被找到：

```bash
which xrun
xrun -version
echo $CDS_LIC_FILE
echo $LM_LICENSE_FILE
```

预期结果类似：

```text
/opt/cadence/XCELIUM2309/tools/bin/xrun
TOOL: xrun(64) 23.09-s003
/opt/cadence/IC231/share/license/license.dat
/opt/cadence/IC231/share/license/license.dat
```

如果 `xrun` 不存在，说明 Cadence 环境变量没有加载。如果许可证变量为空，需要先检查虚拟机自带的 Cadence 初始化脚本或 `.bashrc`。

## 5. 加载 Ibex 运行环境

每次打开新终端，都先执行：

```bash
source ~/iccad/env/ibex_run_env.sh
```

这个脚本很关键。它会加载 conda、RISC-V 工具链、Spike 和 Xcelium，同时清理会干扰 Xcelium DPI 编译的变量，例如 `CC`、`CXX`、`CFLAGS`、`CXXFLAGS`、`CPPFLAGS`、`LDFLAGS`。

执行后检查工具：

```bash
which python
python --version
which xrun
xrun -version
which riscv32-unknown-elf-gcc
which spike
spike --help | head -20
which fusesoc
fusesoc --version
python -c "import yaml, hjson, pandas; print('python deps ok')"
```

如果看到 `python deps ok`，说明 Python 依赖基本正常。

## 6. 一次最小 smoke 测试

进入 Ibex DV 目录：

```bash
cd ~/iccad/repos/ibex/dv/uvm/core_ibex
```

建议先清理旧输出：

```bash
rm -rf out
```

运行 1 个 seed：

```bash
make --keep-going \
  IBEX_CONFIG=opentitan \
  SIMULATOR=xlm \
  ISS=spike \
  ITERATIONS=1 \
  SEED=1 \
  TEST=riscv_machine_mode_rand_test \
  WAVES=0 \
  COV=0
```

如果 make 返回非零，不要立刻判断失败。先看日志是否完整生成：

```bash
find out -type f | sort | grep -E 'compile_tb|rtl_sim|regr|trace|log' | head -120
grep -RniE "error|fatal|failed|undefined symbol|NOFDPI|license|cannot|not found|segmentation|core dumped" out --include="*.log" --include="*.err" | tail -160
```

一个可采集的失败 case 通常需要至少有：

```text
rtl_sim.log
regr.log
trace_core_00000000.log
```

如果只有编译失败日志，没有 trace，那么这个 case 不能作为正常失败样本。

## 7. 当前已验证的 bug_001 patch

当前 smoke patch 位于：

```text
/home/meow/iccad/bug_patches/bug_001.patch
```

该 patch 修改 `rtl/ibex_alu.sv` 中的 ALU 逻辑：

```diff
-      ALU_MAX,  ALU_MAXU: adder_op_b_negate = 1'b1;
+      ALU_MAX,  ALU_MAXU: adder_op_b_negate = 1'b0;
```

这只是 smoke 级 bug，用来验证造数链路是否跑通。它不是最终精细设计过的完整 bug family。

应用前检查：

```bash
cd ~/iccad/repos/ibex
git status --short
git apply --check /home/meow/iccad/bug_patches/bug_001.patch
```

如果 `git apply --check` 没有输出，说明 patch 可以应用。

如果出现：

```text
fatal: unrecognized input
```

优先检查 patch 是否为空：

```bash
file /home/meow/iccad/bug_patches/bug_001.patch
wc -l /home/meow/iccad/bug_patches/bug_001.patch
sed -n '1,80p' /home/meow/iccad/bug_patches/bug_001.patch
```

如果显示 `empty` 或 `0` 行，说明 patch 文件生成失败，需要重新制作。

## 8. 运行造数脚本

进入造数工具目录：

```bash
cd ~/iccad/ibex_dataset_tools
source ~/iccad/env/ibex_run_env.sh
```

运行 bug_001 smoke 数据生成：

```bash
python scripts/run_matrix.py \
  --ibex-root ~/iccad/repos/ibex \
  --manifest ~/iccad/ibex_dataset_tools/bug_families.json \
  --raw-root ~/iccad/dataset/raw_runs_bug001_smoke \
  --bug bug_001 \
  --test riscv_machine_mode_rand_test
```

运行结束后检查 meta：

```bash
find ~/iccad/dataset/raw_runs_bug001_smoke -name meta.json -print
python -m json.tool ~/iccad/dataset/raw_runs_bug001_smoke/bug_001/riscv_machine_mode_rand_test/seed_000101/meta.json
```

一个可接受的 meta 可能长这样：

```json
{
  "status": "ok",
  "return_code": 2,
  "note": "make returned non-zero, but complete logs were generated; treating this as a collected failure case."
}
```

这里 `return_code: 2` 不一定是坏事。因为我们故意注入 bug，仿真失败是预期行为。关键是日志是否完整。

## 9. 汇总成标准数据集

执行：

```bash
python scripts/collect_dataset.py \
  --raw-root ~/iccad/dataset/raw_runs_bug001_smoke \
  --dataset-root ~/iccad/dataset/bug001_smoke_dataset
```

检查输出：

```bash
find ~/iccad/dataset/bug001_smoke_dataset -maxdepth 3 -type f | sort | head -80
head -20 ~/iccad/dataset/bug001_smoke_dataset/input.csv
head -20 ~/iccad/dataset/bug001_smoke_dataset/gold.csv
head -20 ~/iccad/dataset/bug001_smoke_dataset/meta.csv
```

预期结构：

```text
bug001_smoke_dataset/
  input.csv
  gold.csv
  meta.csv
  cases/
    case_000001/
      meta.json
      regr.log
      sim.log
      trace.log
    case_000002/
      meta.json
      regr.log
      sim.log
      trace.log
```

`input.csv` 是模型输入文件列表，`gold.csv` 是标签，`meta.csv` 是辅助元信息。

注意：当前 `input.csv` 里可能是服务器绝对路径，例如 `/home/meow/...`。如果后续在物理机本地直接消费数据，需要重写这些路径，或者让评测脚本按相对路径读取。

## 10. 打包数据集

进入数据目录：

```bash
cd ~/iccad/dataset
```

打包：

```bash
tar -czf bug001_smoke_dataset.tar.gz bug001_smoke_dataset
```

检查压缩包：

```bash
ls -lh bug001_smoke_dataset.tar.gz
tar -tzf bug001_smoke_dataset.tar.gz | head -40
```

## 11. 通过共享文件夹传回物理机

当前 VMware 共享文件夹设置：

```text
Host path: /home/xiongyanbai/Downloads
Share name: download
VM 内常见挂载点: /mnt/hgfs/download
```

先检查共享目录：

```bash
ls /mnt/hgfs
ls /mnt/hgfs/download
```

复制压缩包：

```bash
cp ~/iccad/dataset/bug001_smoke_dataset.tar.gz /mnt/hgfs/download/
ls -lh /mnt/hgfs/download/bug001_smoke_dataset.tar.gz
```

如果 `/mnt/hgfs/download` 不存在，先看 VMware 是否识别到共享名：

```bash
vmware-hgfsclient
```

如果能看到 `download`，但没有自动挂载，手动挂载：

```bash
sudo mkdir -p /mnt/hgfs
sudo vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
ls /mnt/hgfs/download
```

然后重新复制。

## 12. 物理机端检查

在物理机下载目录确认压缩包已经出现。建议复制到项目目录：

```text
D:\ICCAD\bug001_smoke_dataset.tar.gz
```

解压后检查：

```text
D:\ICCAD\_inspect_bug001_smoke_dataset\bug001_smoke_dataset
```

检查重点：

- 是否有 `input.csv`、`gold.csv`、`meta.csv`。
- `cases/` 下是否有对应数量的 case。
- 每个 case 是否都有 `meta.json`、`sim.log`、`regr.log`、`trace.log`。
- `gold.csv` 的 case 数是否和 `input.csv` 对齐。
- `regr.log` 是否显示预期 failure，而不是环境崩溃。

## 13. 常见问题和处理方式

问题一：`x86_64-conda-linux-gnu-c++: error: unrecognized command-line option '-f'`

原因通常是 conda 的 `CXX` 变量污染了 Xcelium 的 `xmsc` DPI 编译。解决方式：

```bash
source ~/iccad/env/ibex_run_env.sh
env | grep -E '^(CC|CXX|CFLAGS|CXXFLAGS|CPPFLAGS|LDFLAGS)='
```

如果这些变量还存在，手动清理：

```bash
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD
```

然后删除旧输出并重跑。

问题二：`undefined symbol: _ZSt28__throw_bad_array_new_lengthv`

这是 `librun.so` 加载到了不匹配的 `libstdc++.so.6`。优先使用 conda 的新 C++ 运行库：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$HOME/iccad/tools/riscv-isa-sim/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1"
```

然后重新运行仿真。

问题三：`NOFDPI Function spike_cosim_init not found`

这通常不是 `spike_cosim_init` 真不存在，而是前面的 `librun.so` 加载失败导致 DPI 函数没注册。先解决 `undefined symbol` 或 `librun.so` 加载问题。

问题四：`fatal: unrecognized input`

如果发生在 `git apply`，大概率 patch 文件为空或不是 unified diff。检查：

```bash
file /path/to/patch
wc -l /path/to/patch
sed -n '1,80p' /path/to/patch
```

问题五：`trace_core_00000000.log not found`

说明仿真没有跑到能生成 trace 的阶段。检查：

```bash
grep -RniE "error|fatal|failed|undefined symbol|NOFDPI|license|cannot|not found|segmentation|core dumped" out --include="*.log" --include="*.err" | tail -160
```

问题六：网络无法安装包

先连接网卡：

```bash
sudo nmcli device connect ens33
```

再修 DNS：

```bash
printf "nameserver 223.5.5.5\nnameserver 114.114.114.114\nnameserver 8.8.8.8\n" | sudo tee /etc/resolv.conf
```

如果 CentOS 源继续失败，需要改用可访问的 vault 或国内镜像源。

## 14. 交接时建议保留的材料

建议接手同学保留以下目录和文件：

```text
~/iccad/env/ibex_env.sh
~/iccad/env/ibex_run_env.sh
~/iccad/repos/ibex
~/iccad/ibex_dataset_tools
~/iccad/bug_patches
~/iccad/dataset/raw_runs_*
~/iccad/dataset/*_dataset
```

每次造完一批数据后，至少保存：

```text
原始 raw_runs 目录
汇总后的 dataset 目录
使用的 bug patch
使用的 manifest
运行命令记录
```

这样以后如果标签、日志或路径有问题，可以回溯到原始仿真输出。

## 15. 当前进度和后续展望

目前第一阶段和第二阶段下限目标均已完成，不再只是 `bug_001` smoke 验证。当前已经完成：

```text
patch 生成
Ibex DV 仿真
日志采集
dataset 汇总
validate_dataset.py 质量检查
tar.gz 打包
VMware 共享文件夹回传物理机
物理机解压复查
```

当前第二阶段数据集包含 16 个 buckets、240 个 cases：

```text
bug_001  alu        alu_add_sub_result_wrong
bug_002  alu        alu_compare_signedness_wrong
bug_003  alu        alu_shift_arithmetic_wrong
bug_005  branch     branch_condition_wrong
bug_008  loadstore  load_sign_extend_wrong
bug_009  loadstore  store_byte_enable_wrong
bug_010  csr        csr_op_set_clear_swap
bug_011  exception  illegal_instr_trap_missing
bug_012  pipeline   forwarding_data_wrong
bug_013  jump       jalr_target_base_wrong
bug_014  decode     opimm_operand_b_mux_wrong
bug_015  multdiv    mul_operator_decode_wrong
bug_016  branch     branch_target_wrong
bug_017  loadstore  load_byte_lane_wrong
bug_018  csr        mtvec_alignment_wrong
bug_019  system     ebreak_decode_wrong
```

每个 `bug_id` 当前包含 15 个 cases，且至少覆盖 2 个不同 test：

```text
riscv_machine_mode_rand_test
riscv_rand_instr_test
```

已经完成的数据质量改进：

- `collect_dataset.py` 默认生成相对路径，例如 `cases/case_000001/trace.log`。
- `validate_dataset.py` 会检查 `input.csv`、`gold.csv`、`meta.csv` 行数一致性。
- `validate_dataset.py` 会检查每个 case 是否包含 `sim.log`、`regr.log`、`trace.log`、`meta.json`。
- `validate_dataset.py` 会检查 `trace.log` 是否是 Ibex tracer 表头。
- `validate_dataset.py` 会过滤明显环境失败，例如 `undefined symbol`、`NOFDPI`、`Segmentation fault`、`core dumped`、license 错误。

因此，第 15 节原先提出的链路稳定、质量检查、路径规范化目标已经完成；第 17.2 节提出的第一阶段 `8 buckets x 10 cases = 80 cases` 目标已经完成；第 17.3 节提出的第二阶段下限 `16 buckets x 15 cases = 240 cases` 目标也已经完成。

当前已经覆盖的粗粒度功能域包括：

```text
alu
branch
jump
loadstore
csr
system
exception
pipeline
decode
multdiv
```

仍未覆盖或覆盖较弱的方向包括：

```text
interrupt
controller FSM
debug
pmp
compressed instruction
fetch
```

下一阶段建议目标：

```text
可选增强：16 buckets x 20 cases = 320 cases
32 buckets x 20-30 cases = 640-960 cases
```

不要一次性追求 64 个 buckets。更稳的做法是每新增 3-5 个 `bug_id`，就执行一次：

```bash
python scripts/collect_dataset.py \
  --raw-root ~/iccad/dataset/raw_runs_first_batch \
  --dataset-root ~/iccad/dataset/first_batch_dataset

python scripts/validate_dataset.py \
  --dataset-root ~/iccad/dataset/first_batch_dataset
```

只有 `validate_dataset.py` 输出 `num_errors: 0` 后，才打包回传物理机。

长期目标是构造多个规模的数据集：

```text
stage1_dataset_8bugs.tar.gz
stage2_dataset_16bugs_240cases.tar.gz
stage3_dataset_32bugs.tar.gz
stress_dataset_64bugs.tar.gz
long_logs_stress_dataset.tar.gz
```

其中 longer logs 不建议现在优先做。当前应优先补齐 root cause 类型和 bucket 数，等短日志数据集稳定后，再基于真实短日志扩增 longer logs 压力集。

## 16. 最短操作清单

如果环境已经稳定，接手同学每天只需要按这个最短流程走：

```bash
source ~/iccad/env/ibex_run_env.sh
cd ~/iccad/ibex_dataset_tools

python scripts/run_matrix.py \
  --ibex-root ~/iccad/repos/ibex \
  --manifest ~/iccad/ibex_dataset_tools/bug_families.json \
  --raw-root ~/iccad/dataset/raw_runs_bug001_smoke \
  --bug bug_001 \
  --test riscv_machine_mode_rand_test

python scripts/collect_dataset.py \
  --raw-root ~/iccad/dataset/raw_runs_bug001_smoke \
  --dataset-root ~/iccad/dataset/bug001_smoke_dataset

cd ~/iccad/dataset
tar -czf bug001_smoke_dataset.tar.gz bug001_smoke_dataset
cp bug001_smoke_dataset.tar.gz /mnt/hgfs/download/
```

传回后，在物理机检查压缩包是否能解压，并确认 `input.csv`、`gold.csv`、`meta.csv` 和 `cases/` 都存在。到这里，一批数据就算完成闭环。

## 17. 造数计划

赛事文件中的 benchmark 规模如下：

```text
Benchmark 1:  10 cases,    2 buckets,   public, 30s
Benchmark 2:  30 cases,    4 buckets,   public, 30s
Benchmark 3:  100 cases,   8 buckets,   hidden, 100s
Benchmark 4:  300 cases,   16 buckets,  hidden, 100s
Benchmark 5:  1000 cases,  32 buckets,  hidden, 100s
Benchmark 6:  3000 cases,  64 buckets,  hidden, 100s
Benchmark 7:  100 cases,   8 buckets,   hidden longer logs, 300s
Benchmark 8:  300 cases,   16 buckets,  hidden longer logs, 300s
Benchmark 9:  1000 cases,  32 buckets,  hidden longer logs, 300s
Benchmark 10: 3000 cases,  64 buckets,  hidden longer logs, 300s
```

题目里的 golden bucket 与 injected bug 一一对应。因此在我们自己的数据集中：

```text
1 个 bug_id = 1 个 injected bug = 1 个 golden bucket
```

`group` 是粗粒度功能域，例如 `alu`、`branch`、`loadstore`。`family` 是中粒度故障机制，例如 `alu_shift_arithmetic_wrong`。最终 `gold.csv` 应以 `bug_id` 作为 bucket 标签。

### 17.1 当前状态

当前已经验证过从 patch、仿真、采集、校验、打包、回传的完整链路。第二阶段下限数据集已经完成，覆盖：

```text
bug_001  alu        alu_add_sub_result_wrong
bug_002  alu        alu_compare_signedness_wrong
bug_003  alu        alu_shift_arithmetic_wrong
bug_005  branch     branch_condition_wrong
bug_008  loadstore  load_sign_extend_wrong
bug_009  loadstore  store_byte_enable_wrong
bug_010  csr        csr_op_set_clear_swap
bug_011  exception  illegal_instr_trap_missing
bug_012  pipeline   forwarding_data_wrong
bug_013  jump       jalr_target_base_wrong
bug_014  decode     opimm_operand_b_mux_wrong
bug_015  multdiv    mul_operator_decode_wrong
bug_016  branch     branch_target_wrong
bug_017  loadstore  load_byte_lane_wrong
bug_018  csr        mtvec_alignment_wrong
bug_019  system     ebreak_decode_wrong
```

当前每个 bug 已收集 15 个 cases，第二阶段完成规模为：

```text
16 buckets
240 cases
```

这已经达到第二阶段中型验证集的下限目标，并对齐 hidden benchmark 4/8 的 bucket 数要求。当前每个 `bug_id` 至少覆盖 `riscv_machine_mode_rand_test` 和 `riscv_rand_instr_test` 两个 test。

### 17.2 第一阶段：稳定小型集（已完成）

目标：

```text
8 buckets
每个 bucket 10 cases
总计约 80 cases
```

用途：

- 验证新增 patch 是否稳定。
- 验证 `collect_dataset.py` 和 `validate_dataset.py` 是否能持续处理新 failure。
- 初步测试聚类 baseline 是否能区分不同 root cause。

实际覆盖的 group：

```text
alu        3 个 bug_id
branch     1 个 bug_id
loadstore  2 个 bug_id
csr        1 个 bug_id
exception  1 个 bug_id
```

已纳入第一阶段的数据标签：

```text
bug_001  alu        alu_add_sub_result_wrong
bug_002  alu        alu_compare_signedness_wrong
bug_003  alu        alu_shift_arithmetic_wrong
bug_005  branch     branch_condition_wrong
bug_008  loadstore  load_sign_extend_wrong
bug_009  loadstore  store_byte_enable_wrong
bug_010  csr        csr_op_set_clear_swap
bug_011  exception  illegal_instr_trap_missing
```

第一阶段结束时建议保留的交付物：

```text
~/iccad/dataset/first_batch_dataset
~/iccad/dataset/stage1_dataset_8bugs.tar.gz
~/iccad/dataset/raw_runs_first_batch
~/iccad/bug_patches
~/iccad/ibex_dataset_tools/bug_families_first_batch.json
```

如果只需要提交或传输精简数据集，优先使用 `stage1_dataset_8bugs.tar.gz`；如果需要复查仿真全过程，则同时保留 `raw_runs_first_batch`。

第二阶段已经纳入的新增方向包括：

```text
pipeline_forwarding_data_wrong
mul_operator_decode_wrong
branch_target_wrong
load_byte_lane_wrong
mtvec_alignment_wrong
ebreak_decode_wrong
```

### 17.3 第二阶段：中型验证集（已完成下限目标）

目标：

```text
16 buckets
每个 bucket 15-20 cases
总计约 240-320 cases
```

实际完成：

```text
16 buckets
每个 bucket 15 cases
总计 240 cases
每个 bug_id 至少覆盖 2 个不同 test
```

用途：

- 对齐 benchmark 4/8 的 bucket 数。
- 测试模型或规则在更多 root cause 下是否还能稳定。
- 检查同一个 group 内的细粒度区分能力，例如多个 ALU bug、多个 LSU bug、多个控制流 bug。

实际做法：

- 每个 `bug_id` 至少覆盖 2 个不同 test：`riscv_machine_mode_rand_test` 和 `riscv_rand_instr_test`。
- 第一阶段已有每个 bucket 10 cases，第二阶段扩样时每个 bucket 再补 5 cases。
- 保留全部 `raw_runs`，不要只保留汇总后的 dataset。

推荐命名：

```text
raw_runs_stage2
stage2_dataset
stage2_dataset_16bugs_240cases.tar.gz
```

### 17.4 第三阶段：大规模短日志集

目标：

```text
32 buckets
每个 bucket 20-30 cases
总计约 640-960 cases
```

更理想的冲刺目标：

```text
64 buckets
每个 bucket 40-50 cases
总计约 2500-3200 cases
```

用途：

- 对齐 benchmark 5/6/9/10 的 case 数和 bucket 数。
- 压测算法的聚类复杂度和运行时间。
- 检查 pairwise 特征或 embedding 方法是否能在 1000-3000 cases 下跑完。

注意：

- 不要一次性生成 64 个 patch 后才验证。
- 每新增 3-5 个 `bug_id`，就采集并校验一次。
- 发现某个 patch 失败率太高、经常编译失败或日志不完整，应及时移出 manifest。

### 17.5 Longer Logs 放到最后

赛事 benchmark 7-10 的 `Max Lines` 是 `100M`，属于 longer logs。它们主要考验：

```text
I/O 性能
内存控制
日志抽样策略
超时鲁棒性
噪声过滤
```

当前阶段不要优先造 longer logs。原因是短日志数据的 root cause 覆盖还不够，过早制造长日志会增加存储和传输成本。

推荐策略：

```text
先造真实短日志数据集
再基于真实短日志人工扩增成长日志压力集
最后只用 longer logs 检查 parser 和 runtime
```

扩增长日志时不要改变核心错误片段。可以在日志头部、中部或尾部插入大量无关 UVM_INFO、普通 trace 行或重复背景噪声，使错误位置分布更接近真实长日志。

### 17.6 训练、验证、测试划分

不要按 case 随机划分。随机划分会让同一个 `bug_id` 的不同 seed 同时出现在训练和测试中，导致 root cause 泄漏，测试分数虚高。

建议按 `bug_id` 划分：

```text
train: 70% bug_id
valid: 15% bug_id
test:  15% bug_id
```

例如 32 个 `bug_id`：

```text
train: 22 个 bug_id
valid: 5 个 bug_id
test:  5 个 bug_id
```

如果只有 16 个 `bug_id`：

```text
train: 11 个 bug_id
valid: 2-3 个 bug_id
test:  2-3 个 bug_id
```

这样可以更真实地评估模型对未知 root cause 的泛化能力。题目本质是 failure bucketing，不是固定标签分类，因此本地 test 集最好包含训练时没见过的 `bug_id`。

### 17.7 每批数据的验收标准

每批数据完成后必须满足：

```text
validate_dataset.py 输出 num_errors = 0
input.csv 使用相对路径
gold.csv 行数与 input.csv 一致
每个 case 有 sim.log、regr.log、trace.log、meta.json
trace.log 表头为 Ibex tracer 格式
sim.log 不包含 undefined symbol、NOFDPI、Segmentation fault、core dumped、license 等环境失败
每个 bug_id 至少有 5 个有效 case
```

建议每批数据额外记录：

```text
生成时间
使用的 manifest
使用的 patch 列表
每个 bug_id 的 case 数
每个 bug_id 的主要失败表象
压缩包文件名
是否已回传物理机
```

当前命名建议：

```text
first_batch_dataset_6bugs.tar.gz
stage1_dataset_8bugs.tar.gz
stage2_dataset_16bugs_240cases.tar.gz
stage3_dataset_32bugs.tar.gz
stress_dataset_64bugs.tar.gz
long_logs_stress_dataset.tar.gz
```
