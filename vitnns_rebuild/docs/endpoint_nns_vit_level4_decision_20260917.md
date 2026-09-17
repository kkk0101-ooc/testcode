# Endpoint NNS-ViT 最小验证与决策

日期：2026-09-17

结论：Endpoint NNS-ViT 的实现身份得到验证，但当前协议在 20 图探索性评价中未通过进入 50 图的预设门。因此停止扩展，不重跑 500 图。

## 方法身份

Endpoint NNS-ViT 定义为

$$
L_s^{\mathrm{end}}[i;x_0]=(x_i-x_{0i})\frac{\partial s(x)}{\partial x_i}.
$$

它与独立实现的 Gradient × Input 在 toy tests、单图完整 ViT pilot 和 20 图评价中逐元素一致。最大绝对误差为 0。该结果确认实现身份，不构成新的基础归因公式。

Path-integrated learning image 定义为

$$
L_s^{\mathrm{path}}[i;x_0]=(x_i-x_{0i})\int_0^1\frac{\partial s(x_0+\alpha(x-x_0))}{\partial x_i}d\alpha.
$$

两个版本继续使用不同方法名。

## Level 1–2

20 个测试全部通过，覆盖：

- 原论文无偏置 Linear–ReLU–Linear 的权重/端点门传播；
- bias 导致的响应重构差额；
- ReLU 零点的 PyTorch 次梯度约定；
- Conv、MaxPool winner 和 residual 加法；
- LayerNorm、GELU、softmax、single-head attention、multi-head attention 和 MLP；
- attention 的 Q-routing、K-routing 和 Value 一阶项之和。

原论文未提供可定位的官方代码仓库，因此当前证据是 paper-formula reproduction，不是 official-code parity。

## Level 3：单图完整链

固定图像 `ILSVRC2012_val_00007922.JPEG`，冻结 ViT-B/16，预测类别 235。真实链覆盖：

```text
patch embedding -> attention branch -> first residual
-> MLP branch -> second residual -> blocks 1..11
-> final norm -> CLS -> classifier logit
```

结果：

- endpoint 与独立 Gradient × Input 最大误差：0；
- 第一次 residual endpoint 加法误差：$2.08\times10^{-16}$；
- 第二次 residual endpoint 加法误差：$5.00\times10^{-16}$；
- classifier endpoint/path Pearson：0.6872；
- classifier endpoint/path sign agreement：0.7539；
- B0/H9 conditional Value-path endpoint/path Pearson：0.9987。

这说明 endpoint 与 path-integrated 版本在真实 ViT 的完整类别目标上产生实质差异，而在线性程度较高的固定 attention Value-path 目标上接近。

真实图：[模块链](../reports/nns_vit_endpoint_mvp_r2_20260917/figures/01_module_chain_endpoint_vs_path.png) · [classifier 与条件 Value-path](../reports/nns_vit_endpoint_mvp_r2_20260917/figures/02_terminal_targets_endpoint_vs_path.png)。

## Level 4：20 图、8 卡探索性评价

20 张本地 ImageNet 验证图在运行前按文件名 SHA-256 排序固定。8 个 H20 worker 使用原子 claim 动态领取任务；每张 GPU 均完成 2–3 张图。20/20 VALID，0 个科学或基础设施失败。

每张图比较 endpoint、path-integrated IG 和普通 gradient，包含：

- normalized-zero / RGB 0.5 reference；
- normalized-minus-one / RGB 0 reference；
- 水平翻转后逆变换稳定性；
- 4×4 cell 与 16×16 patch 的 insertion/deletion；
- 21 个干预比例；
- 运行时间。

| 指标 | Endpoint | Path-integrated | Endpoint − Path |
|---|---:|---:|---:|
| 4×4 combined fidelity 中位数 | 0.03639 | 0.10143 | −0.07435 |
| 16×16 combined fidelity 中位数 | 0.01161 | 0.02120 | −0.01493 |
| 水平翻转 Spearman 中位数 | 0.34104 | 0.47416 | — |
| 两 reference Pearson 中位数 | −0.07869 | 0.03416 | — |
| Endpoint / path 运行时间比中位数 | 0.01492 | 1 | — |

Endpoint 和 path 在 normalized-zero reference 下的 Pearson 中位数为 0.43648，符号一致率中位数为 0.65042。Endpoint 在 4×4 fidelity 上只在 1/20 张图高于 path；在 16×16 上为 5/20。

## 预设门结果

| 门 | 结果 |
|---|---|
| 20/20 有效 | PASS |
| endpoint 恒等误差 $\le10^{-6}$ | PASS |
| endpoint 4×4 fidelity 中位数 $>0$ | PASS |
| endpoint − path 4×4 fidelity 中位数 $\ge-0.05$ | FAIL |
| endpoint 两 reference Pearson 中位数 $\ge0.5$ | FAIL |
| endpoint flip Spearman 中位数 $\ge0.5$ | FAIL |
| endpoint/path 运行时间比中位数 $\le0.25$ | PASS |

决策为：

```text
NO_GO_CURRENT_ENDPOINT_PROTOCOL
```

该决策表示当前 endpoint 协议不进入 50 图和 500 图评价。它不否定将原始 NNS 的内部扫描组织方式适配到 ViT；它说明当前 `Gradient × Input` 形式虽然计算快，但在本次小样本中相较 path-integrated 版本显示出更低的 fidelity、reference 稳定性和翻转稳定性。

## 对论文的处理

当前证据支持以下定位：

> 原始 NNS 的端点传播在 ViT 上可实现为结构化的 Gradient × Input scanner。它可以索引模块、通道、head 和 branch，但当前 20 图结果不支持将该 endpoint 版本替换 path-integrated learning image 作为论文主归因方法。

R9 保持封存。现有 internal-target IG、conditional Value-path IG 和 component conductance 使用各自原方法名。Endpoint NNS-ViT 作为方法身份复现与负结果进入方法分析，不重命名既有结果，也不启动 500 图重算。

## 产物

- [方法身份说明](endpoint_nns_vit_method_identity_20260917.md)
- [Level 3 报告](../reports/nns_vit_endpoint_mvp_r2_20260917/pilot_report.json)
- [Level 4 协议](../reports/nns_vit_endpoint_level4_r1_20img_20260917/protocol.json)
- [Level 4 汇总](../reports/nns_vit_endpoint_level4_r1_20img_20260917/summary.json)
- [Endpoint 实现](../src/vitnns/endpoint_nns.py)
- [测试](../tests/test_endpoint_nns.py)
