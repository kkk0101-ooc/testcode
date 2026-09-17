# 通过学习图像理解视觉 Transformer 的内部响应与贡献分配

作者：[AUTHOR REQUIRED]

单位：[AFFILIATION REQUIRED]

## 稿件结构

1. 摘要
2. 引言
3. 相关工作
4. 方法
5. 实验设置
6. 结果与讨论
7. 局限
8. 结论

## 中心论点

Learning image 把 ViT 的指定内部响应或类别输出表示为输入坐标下的带符号贡献，从而将逐模块观察、干预评价和内部成分记账放入同一框架。固定 block 0/head 9 的 500 图实验显示，输入归因排序在 4x4 细粒度单元上的忠实度高于同目标 patch-token 基线（patch 内均匀分配）；该结果依赖评价尺度和内部坐标，也不构成对所有类别解释器的普遍优势。

## 图组安排

- Figure 1-3：研究视角、learning image 和干预协议。
- Figure 4-7：主效应、轨迹、尺度、层头与预测分层。
- Figure 8-11：全 head 图谱与独立解释器比较。
- Figure 12-14：探索性多尺度、分支份额和深度延伸。
- Figure 15-27：learning image 定义与 ViT 算子解析例。
- Figure 28-35：冻结真实图像的模块链与 classifier 输出。
