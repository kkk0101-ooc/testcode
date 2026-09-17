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

NNS-ViT 为视觉 Transformer 中的内部响应生成保持输入位置的 learning image，并将不同 token、channel、attention head、MLP、residual branch 和类别输出放在统一输入坐标下观察和组合。论文沿完整 ViT 计算链展示模型从输入图像中形成内部响应的过程，再通过输入干预、解释器比较和完整切分研究这些图的解释用途以及贡献在网络深度上的组织。

## 图组安排

- Figure 1-3：研究视角、learning image 和干预协议。
- Figure 4-7：主效应、轨迹、尺度、层头与预测分层。
- Figure 8-11：全 head 图谱与独立解释器比较。
- Figure 12-14：探索性多尺度、分支份额和深度延伸。
- Figure 15-27：learning image 定义与 ViT 算子解析例。
- Figure 28-35：冻结真实图像的模块链与 classifier 输出。
