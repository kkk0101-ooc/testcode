# R10：从 B0 MLP 到类别 logit 的真实 learning image 扩展

R10 继承已封存且通过校验的 R9 Figure 1–31，不原地修改 R9。新增 Figure 32–34，把真实前向展示从 B0 第一次 residual 延伸到 B0 MLP、B0 第二次 residual，并通过 Blocks 1–11、final norm、CLS 与 classifier head 到达预测类别 logit。

## 固定对象与模型顺序

- 输入仍为既有 Y2 冻结队列首图 `ILSVRC2012_val_00007922.JPEG`；参考仍为规范化空间中的零，即 RGB 灰色 0.5。
- 模型仍为已绑定的 `vit_base_patch16_224.augreg2_in21k_ft_in1k`，ViT-B/16、12 blocks、768 维 token。B0 MLP 为 `768 -> 3072 -> GELU -> 768`，classifier 为 `Linear(768,1000)`，使用 final-normalized CLS token。
- 前向顺序为

$$
a_0=\operatorname{Attention}(\operatorname{LN}_1(h_0)),\qquad
r_0^{(1)}=h_0+a_0,
$$

$$
m_0=\operatorname{MLP}(\operatorname{LN}_2(r_0^{(1)})),\qquad
r_0^{(2)}=r_0^{(1)}+m_0,
$$

$$
u=\operatorname{FinalNorm}(h_{12}),\qquad
\ell=W_{\mathrm{cls}}u_{\mathrm{CLS}}+b_{\mathrm{cls}}.
$$

实际实现保留模型中的 layer scale、drop path、dropout 等节点；在 eval 模式下随机丢弃关闭，但节点仍属于真实前向。

## Figure 32–34 的计算目标

Figure 32 和 33 固定通道 0、255、511、767。每个标量目标都是指定通道在 196 个 patch tokens 上的输出和：

$$
s_{M,c}(x)=\sum_{t=1}^{196}M(x)_{t,c},
$$

其中 $M$ 分别为 B0 MLP branch 输出 $m_0$ 与 B0 block 输出 $r_0^{(2)}$。CLS 参与前向，但不计入这两个空间输出和。图版沿用 R9：上排为 224×224 输入贡献图，下排为 14×14 signed token 值。完整 signed learning image 另存于 supplement。

第二次 residual 必须逐像素满足

$$
L_{s_{r_0^{(2)},c}}=L_{s_{r_0^{(1)},c}}+L_{s_{m_0,c}}.
$$

Figure 34 固定原图真实预测类别 235，以完整类别 logit 为标量目标：

$$
L_{\ell_{235}}[i]=(x_i-x_{0i})\int_0^1
\frac{\partial \ell_{235}(x_0+\alpha(x-x_0))}{\partial x_i}\,d\alpha.
$$

它通过全部 12 个 blocks、final norm、CLS 和 classifier head 反传到输入，输出一张 signed 224×224 learning image。类别标量没有 14×14 token 输出，因此 Figure 34 不套用模块通道页。

## 数值与范围边界

- CPU float64、完整模型自动微分、Gauss–Legendre 64/128 点；B0 模块在满足阈值后停止，较深的类别路径必要时独立提高至 256/512/1024/2048 点。相邻阶地图 L1 相对差要求小于 0.005。
- 每个标量目标检查 completeness；模块目标阈值为 `1e-4 + 0.001 * abs(delta)`，类别 logit 使用同一阈值。
- 检查 float32 规范前向与显式 float64 前向的一致性、类别 235 预测一致性、MLP/第二 residual 捕获一致性，以及第二 residual 的逐像素加法。
- 这些图仍是同一冻结图像上的方法展示，不新增总体机制或总体效果结论。
- R10 完整图包为 Figure 1–34；R9 文件和终态保持不变。

## 数值阶数升级记录

第一次正式计算到 2048 阶时，classifier 地图相对 1024 阶的 L1 变化为 0.0032659，已低于 0.005；但 completeness 绝对误差为 0.0113198，略高于固定阈值 0.0085666。该次运行按规则写出 `FAIL_CLASSIFIER_CONVERGENCE`，没有生成 PASS 收据。随后只把允许的最高积分阶数扩展到 4096；输入、模型、类别、参考、路径、批大小、收敛阈值和显示规则均不改变。
