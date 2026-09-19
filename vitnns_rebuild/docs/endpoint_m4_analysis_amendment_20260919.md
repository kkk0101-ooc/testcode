# Endpoint M4 离线分析修订 A1

本修订冻结于 M4 GPU 生产运行期间、任何汇总结果读取之前。GPU 生产协议、task manifest、目标、cohort、模型、head/block 范围、Endpoint/JVP 计算和验证门均不变；现有 task 不重跑。

原冻结分析已覆盖全部 M4 主检验。为逐项覆盖 M4-B §11.4 和 M4-C，离线 CPU 汇总增加：

1. Q、K、routing、value、full，以及 identity、branch、output 各自的 signed magnitude、absolute mass 和 map cancellation 的 block-level image summary；
2. 每个模型、每种 residual 的图像级 branch_share 与 path_cancellation 跨 12 blocks Spearman；
3. early/middle/late 三个 phase 内 attention residual 减 MLP residual 的 identity share、path cancellation 和 output-identity patch TV 配对描述值。

这些字段全部由已冻结的原始逐 task JSON 确定，不增加检验 family，不读取图像、模型或 GPU，不改变 M4 主结论判据。修订实现位于 scripts/run_endpoint_m4_analysis_a1.py，须在运行前提交并推送，输出写入原 M4 fresh root 的版本化 A1 文件名。
