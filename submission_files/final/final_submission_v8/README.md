# EDA Regression Failure Bucketing

ICCAD 2026 Problem B 提交方案：给定一批 Ibex RISC-V CPU 回归仿真的失败日志
（`sim.log` / `regr.log` / `trace.log`），把由**同一根因 bug** 导致的 case 聚到
同一个 bucket。打分是 pairwise Balanced Accuracy = (TPR + TNR) / 2。

## 最终提交

| 项 | 值 |
|---|---|
| 提交包 | [`submission_files/final/final_submission_v8/`](submission_files/final/final_submission_v8/) |
| 模型 | siamese 编码器（5-seed Procrustes 对齐平均）→ k-means |
| 官方 dev 分（有 LLM，train-on-dev）| set1=1.000、set2=0.979（诚实分，已去 case_index 泄漏）|
| 运行时（32 核）| set1=3.5s、set2=7.4s、N=3000 ≈ 79s（< 100s 限制）|

接口：

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

输出 `Case,bucket` 两列 CSV。

## 模型

- **特征（364 维）**：LLM 嵌入（nomic，768→SVD 64）+ 失败签名（功能单元家族 + 分歧类型）
  + 测试名语义类别 + sim.log 首条 UVM_FATAL/ERROR 行的 char n-gram（128）
  + 分歧点窗口分布（48）+ trace 残差（96）。
- **trace 截断**：只解析 trace 尾段 5000 条指令（fatal 类型 bug 的 trace 会冲到几十万行，
  尾段才是判别信号，前部是噪声），`theta_trace_features.py` 的 `max_instructions`。
- **多进程并行**：trace 特征构建用 `multiprocessing.Pool`（32 进程），解决 N=3000 超时。
- **无 LLM 兜底**：`LLM_MODEL_CONFIG` 缺失/失败时 LLM 块零向量化，模型照常跑（set2 退到 0.733）。

## 核心结论（贯穿全程，详见 [handoff.md](handoff.md)）

1. **数据分布（1 bug = 多测试）是决定性的**——v4 扩展批补上测试多样性后，官方均值
   0.72 → 0.87，比任何模型侧改进都大。
2. **测试名是语义类别真信号，不是泄漏**（坑 #32）。
3. **LLM 当聚类/根因判别器走不通**（6 次失败，坑 #30/#35），只能当 embedding。
4. **trace 解析是 N=3000 超时的致命瓶颈**，必须「截断 + 多进程」（坑 #37）。

## 关键文件

| 文件 | 作用 |
|---|---|
| `siamese_predict.py` | 推理入口（PyInstaller 打包，NumPy 实现，无 PyTorch）|
| `run_siamese_train.py` | 训练（SupCon + 原型损失）|
| `run_siamese_procrustes_eval.py` | 5-seed Procrustes 集成评估 |
| `theta_siamese_model.py` | siamese 编码器 + 损失函数 |
| `theta_trace_features.py` | trace 特征（截断 + 多进程）|
| `failure_signature.py` | 失败签名提取 |
| `regr_fail_bucketing.py` | 自包含基线（源码后备）+ LLM 工具函数 |

## 环境

- Python：`/home/lishixian/miniforge3/envs/collab-overcooked/bin/python`（有 torch）
- LLM 配置：`LLM_MODEL_CONFIG` 环境变量（YAML 内容），本地开发用 `/home/lishixian/llm_qwen.yaml`
- 推理只跑 CPU，无 GPU / 无网络 / 无 pip 依赖

## 详细文档

- [handoff.md](handoff.md) — 完整的交接文档（战略、数据集、模型迭代、踩坑记录）
- [MATERIALS_INDEX.md](MATERIALS_INDEX.md) — 数据/竞赛文件导航
