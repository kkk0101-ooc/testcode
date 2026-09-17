# R9：完整论文图集与真实模块 learning image

R9 将已有 Figure 1–23、正常前向与 learning image 四张对照图，以及新算的三个真实模块输出统一编为 Figure 1–31。旧图包保留。Figure 2 重画，Figure 19 采用已有单排正贡献候选版。图集包含方法图和实验图，最终正文与补充材料的取舍由论文版面决定。

## 新增真实输出的预定计算

- 固定已有 Y2 队列第一张输入 ILSVRC2012_val_00007922.JPEG，复用已校验的 224×224 RGB 数组及模型绑定。该选择来自既有展示协议。
- 固定 ViT-B/16，依次查看 patch embedding、B0 attention 分支输出（含输出投影）和 B0 第一次 residual add（MLP 之前）。保留位置编码、CLS、LayerNorm、softmax、偏置和全部 heads。
- 四个通道固定为 0、255、511、767，在计算图片之前按索引跨度确定。每个模块、通道的标量目标为该通道在 196 个 patch tokens 上的响应之和。CLS 参与 attention 计算，不计入这项空间输出聚合。
- 灰色参考在规范化输入中为零。响应 learning image 为该标量目标的输入 IG，RGB 带符号求和，原生 224×224；原始 RGB 归因也保存。它解释模块自身响应，不乘最终类别导数。
- 用 CPU float64 完整 attention 计算 64 和 128 点 Gauss–Legendre 积分，必要时提高至 256/512 点。要求相邻阶数地图 L1 相对差小于 0.005，完整性绝对误差小于 1e-4 + 0.001 × abs(目标变化)。不依据画面选通道或输入。
- 独立求 patch、attention、residual 三种目标导数。检验 L(residual)=L(patch)+L(attention)；位置编码的直接变化为零。patch 映射另与卷积权重乘输入差的解析式比较。
- 完整模型原图 logit 与已有记录误差应小于 1e-4；CPU float64 第一块输出与原规范 float32 前向按 1e-4 + 1e-4 × abs(值) 比较。

## 图号

Figure 1–14 保留现有实验图号，2 改为真实输入到单张完整 learning image 的定义图。Figure 15–23 汇总已有概念、初始化、生成、模块、类别成分与加法细节。Figure 24–27 原样纳入四张逐步对照图。Figure 28 展示真实三个模块的同目标通道 progression，29–31 分别展示四个固定通道在 patch embedding、attention、residual 的真实输出。

新实验图采用原输入第一列、模块或通道为列、learning image 与内部 token 响应为两排的组织方式，借鉴原 NNS Figure 13。完整 signed 图与单排正贡献图都留存；图中不再为每个模块并排堆两组正负图。非负显示采用每面板正部 P99，nearest，不把显示幅度当作跨模块比较。内部 token 响应使用原始有符号值，明确其坐标为 token 索引。

这次计算为单样本方法演示，既有 D/E/F/X/Y/S 总体统计原样引用。真实模块图的误差检查验证具体计算，不增加总体机制结论。
