# 历史实验脚本（已归档）

这里存放早期实验阶段的历史脚本（pair 模型、theta trilog、multiview、beta 版本等），
均已被当前的 **siamese 主线**（`run_siamese_train.py` + `siamese_predict.py`）取代。

- 这些脚本**不参与**当前的训练/推理/打包流程。
- 它们之间、以及它们和根目录核心模块之间可能有 import 依赖，直接运行可能报错
  （需要 `sys.path` 指回根目录）。仅作历史参考，不再维护。
- 当前核心模块保留在项目根目录，见根目录 [`README.md`](../README.md)。
