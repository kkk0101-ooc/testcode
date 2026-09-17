# Terminology

| Preferred English | Preferred Chinese | Avoid | Meaning |
|---|---|---|---|
| learning image | 学习图像 | feature map when referring to attribution | Input-coordinate attribution for a specified scalar response |
| module-own-response learning image | 模块自身响应学习图像 | class contribution map | Explains a module response without a downstream class derivative |
| fixed-clean-attention conditional Value-path attribution | 固定 clean attention 的条件 Value-path 归因 | full attention attribution; full Q/K/V attribution | Attention is frozen at its clean value and the Value path remains live |
| complete-cut class-component map | 完整切分类别成分图 | layer attribution without a cut definition | Full-model attribution assigned to components of one complete internal cut |
| full-model integrated gradients | 完整模型积分梯度 | NNS | IG of the live full class-logit function |
| target-matched patch-token baseline with uniform within-patch allocation | 同目标 patch-token 基线（patch 内均匀分配） | matched baseline | Token-space contribution baseline used in Stage D/E |
| matched occlusion control | 匹配遮挡对照 | matched baseline | Stage S control matched in patch count, position, and shape |
| B0 MLP branch | B0 MLP 分支 | MLP head | FFN branch inside transformer block 0 |
| classifier head | 分类器头 | MLP branch | Linear(768,1000) after all blocks and final norm |
| fine 4x4 cell | 4x4 细粒度单元 | pixel-level attribution | Evaluation unit within a native 16x16 patch |
| fidelity difference / delta fidelity | 忠实度差值 | accuracy gain | Paired insertion/deletion AUC contrast defined by the frozen protocol |
| post-hoc exploratory | 事后探索性 | confirmatory; preregistered | Analysis defined or interpreted after outcome access |

## Abbreviations and symbols

- NNS: Neural Network Scanner.
- ViT: Vision Transformer.
- IG: Integrated Gradients.
- CLS: classification token.
- `x`: normalized clean input; `x_0`: normalized-zero gray reference.
- `L_s`: learning image for scalar target `s`.
- `A_clean`: attention tensor captured on the clean input and then frozen.
- `delta_fid`: image-level paired fidelity contrast; larger favors NNS under the frozen definition.
