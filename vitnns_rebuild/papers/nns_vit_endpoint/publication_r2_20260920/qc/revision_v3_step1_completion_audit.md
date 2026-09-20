# 修改意见 v3 第一步完成审计

## 目标与完成标准

目标是执行 `修改意见/修改意见v3.md` 的“第一步：收紧研究问题和创新定位”，仅进行文献、理论、论证和文稿重构，不启动第二步新增实验。完成要求为：修订双语题目、摘要、研究问题与贡献；扩展指定相关工作；增加方法定位表；说明计算图、切面与参数化依赖并给出反例或明确否定一般实现不变性；使 RQ4 与已执行实验完全一致；将训练因果语言收窄为参数状态差异；冻结可发表主张；逐项满足三条验收标准。

源意见 SHA-256：`5e39a561b98454c9311e551bdafa28d17a11d1cc6ce6a56566d05b94a8eb097c`。

## Prompt-to-artifact checklist

| 明确要求 | 状态 | 直接产物与核验 |
| --- | --- | --- |
| 核心贡献限定为 NNS 在 ViT 计算图上的结构化实现、路径分解和系统验证 | 完成 | 双语题目、摘要、Introduction 三项贡献与 Conclusion；均使用统一实现、图相对路径分解、系统验证三层定位 |
| 对比 Gradient × Input、conductance、FullGrad、GlobEnc、DecompX、AttnLRP、LibraGrad、activation patching、edge attribution | 完成 | 双语 Sections 2.2–2.5 逐项对比；`references.md` 增补 FullGrad、GlobEnc、DecompX、attribution patching 与 activation patching 来源 |
| 方法定位表比较研究对象、reference、内部路径、完整性、有限干预 | 完成 | 双语 Table 1 含 `Primary object`、`External reference`、`Internal-path decomposition`、`Completeness / closure`、`Finite-intervention evaluation` 五列；完整性按各方法对象解释 |
| 明确计算图、切面和参数化依赖 | 完成 | 双语 Section 4.4 首句及结论明确分解为 graph/cut/parameterization-relative |
| 等价重参数化反例或否定一般实现不变性 | 完成 | 双语 Section 4.4 给出 $R=\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 与 $R^{-\top}$ 反例：总 Endpoint 不变，隐藏坐标路径项由 $(x_1,0),(0,x_2)$ 变为 $(x_1,x_2),(0,0)$ |
| 重写 RQ4 | 完成 | Introduction RQ4 与 Section 7 均限定为规定有限替换任务中的成分/区域排序；固定模型、样本、层、head、reference、替换算子和预算 |
| 无多幅度实验时删除一阶近似边界主张 | 完成 | 主文不再含 $0.05,0.1,0.25,0.5,1.0$ 扫描或“一阶近似边界”题名；Discussion 明确 RQ4 不估计误差随幅度变化的边界 |
| “训练强化趋势”改为参数状态差异 | 完成 | 双语 Abstract、RQ3、Section 6.6、Conclusion 及双语 Supplement S10；正文明确该对照不识别训练过程因果效应 |
| 修订后的题目、摘要、RQ 与贡献列表 | 完成 | `zh/manuscript.md`、`en/manuscript.md` 及外层 `zh/en/full_draft.md` 镜像；双语结构、公式和数据表一致 |
| 扩展相关工作 | 完成 | 双语 Related Work 由概括性五段扩展为 NNS；梯度/路径/FullGrad；Transformer contribution/relevance；activation/edge patching；视觉归因与定位 |
| 计算图依赖与分解非唯一性小节 | 完成 | 双语 Section 4.4 |
| 冻结后的主张清单 | 完成 | `notes/revision_v3_frozen_claims.md`：K1–K3 贡献矩阵、C1–C7 主张、RQ4 合同与禁用扩展 |
| 每项贡献列已有工作、本文新增和直接证据 | 完成 | Introduction 三项贡献逐项使用该结构；冻结清单 K1–K3 再以表格绑定原始 JSON/报告 |
| 不把链式法则写成主要创新 | 完成 | 双语贡献段明确 chain rule 和 $x\odot\nabla_xs(x)$ 仅界定方法；冻结清单将其列为禁用创新表述 |
| RQ4 与计划执行实验完全一致 | 完成 | RQ4 合同只引用已经执行的完整替换、两种 reference、已注册组件、block/head/region/budget 与 Spearman/Capture/Regret/AUC；未启动第二步实验 |
| 防御性写作清理 | 完成 | `qc/revision_v3_step1_writing_audit.md`；保留必要证据边界，防御性套话扫描为零 |
| 校验器实际覆盖 v3 第一步 | 完成 | `validate_publication_r2.py` 新增 v3 题名、八张表、指定 RQ4、无未执行幅度主张、非唯一性、参数状态边界、九类方法、参考文献和冻结清单检查；不再沿用旧的多幅度文本代理 |

## 验收标准结论

1. 每项贡献均在正文和冻结清单中指出先例、技术增量和直接证据：通过。
2. 链式法则及 Gradient × Input 身份只用于数学定位，不作为主要创新：通过。
3. RQ4 只询问现有完整替换实验可以回答的排序问题，未暗示未执行的多幅度扫描：通过。

## 验证记录

- `validate_publication_r2.py`：PASS，1,716 项检查，0 errors；覆盖双语题名/RQ/表格/公式、v3 专项规则、正文图与 source hashes、入口镜像及 165 个真实显示面板复算。
- `validate_endpoint_supplement_v1.py --paper`：PASS，32,091 项检查，0 errors；覆盖 456 个补充任务（64 feature、192 random、200 target）和 source/payload/closure/论文链接。
- `verify_publication_r2_methods.py`：PASS；9 个 image-model 任务完成原生图复算，Endpoint 最大绝对误差为 0，Grad-CAM 最大绝对误差为 $2.794\times10^{-9}$，rollout 最大绝对误差为 $1.041\times10^{-17}$。
- `validate_revision_v3_step1.py`：PASS，26/26 项；按源意见逐项检查双语正文、参考文献、冻结主张及 M3/E3/E4/E6/补充实验原始 JSON，覆盖全部显式交付物与验收条件。
- 双语主文 display equations、数据表行、章节编号和 Figure 1–11 路径一致；外层 `full_draft.md` 与便携正文镜像一致。
- `git diff --check`：PASS。

## 审计结论

修改意见 v3 第一步的全部显式要求均已有直接文稿、文献、证据绑定或验证覆盖。多幅度干预、Q/K/V 总体统计、独立抽样复现等属于第二步，未在本轮执行，也未用现有代理指标冒充完成。
