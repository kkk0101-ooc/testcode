# Figure 15–21：learning image 的讲解与实测图

本次在已封存的 R8 Figure 1–14 之外增加七张图，不改动原图包。
来源包括《参考文章.pdf》、论文架构目录中的五份正式 PPTX 和三份 PDF；~$ 文件是 Office 锁文件。

| 图 | 内容 | 对照原论文 |
|---|---|---|
| 15 | 内部单元和输入坐标图怎样一一对应 | Figure 2 |
| 16 | 输入差值拆成四张保持位置的初始图 | Figure 3 |
| 17 | 两个线性单元的 learning image 如何生成与组合，数值完整可核算 | Figure 4 |
| 18 | patch projection、attention 和 residual 怎样进入 ViT 贡献计算 | Figure 5 |
| 19 | 同一类别目标的逐层 attention 分支 learning image，与该分支 token 响应的上下对照 | Figure 13 |
| 20 | 同一类别目标的不同 head learning image，固定首、中、末 head，不按漂亮程度选 head | 补充 head 维度 |
| 21 | 同一切分的 head/attention/旁路图相加恢复总输入图 | 补充完整性与正负说明 |

Figure 15–18 为明确标记的数学示意。输入差值种子是路径积分的等价表示，实际实现通过自动微分计算，不把每张种子图当成一个新样本分别送进非线性 ViT。

实测图固定使用 Y2-R2-R2 unified_100_unit_manifest 中 selection_order=0 的 VALID 样本。
地图直接读取已有封存数组，检查原文件 SHA-256 与原数组摘要。图 19 选 B0/2/4/6/8/11；图 20 选 B0/5/11、H0/5/11；图 21 固定 B5。
完成单次 CPU 前向，读取同一样本、同一模型各层 attention 分支的 token 响应，并核对类别 ID 和目标 logit 与封存记录的绝对误差不超过 1e-4。不新增归因积分或干预实验。

Figure 19 上排是带类别目标的输入贡献，按原 56×56 cell 坐标显示；下排是 attention 分支输出在 196 个图像 token 上的 RMS，形状 14×14。前者按输入位置，后者按内部 token 索引，二者不要求相等或相似。
图中正向展示按 signed cells 求正部后逐面板 P99 缩放；保存未着色输入、原始摘要、缩放参数和负向补充。跨面板亮度不比较原始贡献大小。
图 21 的等式用原始 signed 数组检查，统一 signed-log 色阶只用于看图，不对已归一化热图做求和。

旧 PPT 的高维 learning-image 设想用于补充“位置如何保留”的解释。旧 46%→78.12%、EE 和从彩色 PNG 计算熵的结果保留为历史讨论，不加入当前 ViT-B/16 实验结论。
