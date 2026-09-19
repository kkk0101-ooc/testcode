# M3-SUP-R2 聚合修正案

M3-SUP-R1 在生成补充结果后发现 phase-level CLS–patch 汇总顺序与用户要求不完全一致：R1 在 phase 内先合并多个 block 的单元，而正确合同应先在每张图、每个 block 内分别聚合 32 个 CLS channels 和 15×32 个 patch-token channels，形成 block-level paired difference，再在 phase 内跨四个 block 取中位数。逐 block 数值不受影响，但 R1 不签发科学结论。

R2 在 fresh root 中重跑全部补充分析，固定以下规则：

1. block-level CLS 与 patch 各自取中位数后相减；
2. early/middle/late/all 分别对 4/4/4/12 个 block-level paired differences 取中位数；
3. 每个图像只贡献一个 phase-level 值；
4. 不新增 confirmatory test，不改变原 M3 的 10 项 Holm 判定；
5. 其他分析复算而非从 R1 继承；24-target float64 cancellation 也在 R2 重做。

R1 的 `superseded_receipt.json` 永久保留该边界。
