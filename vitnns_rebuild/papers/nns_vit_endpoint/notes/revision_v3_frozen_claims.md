# 修改意见 v3 第一步：冻结主张清单

冻结日期：2026-09-21。适用正文：`publication_r2_20260920/zh/manuscript.md` 与 `publication_r2_20260920/en/manuscript.md`。本清单冻结第一步完成后可进入题目、摘要、研究问题、贡献和结论的主张；后续新增实验只有在直接证据完成并同步修改本清单后才能扩大范围。

## 贡献—先例—新增—证据矩阵

| ID | 已有工作 | 本文新增内容 | 直接证据 | 冻结边界 |
| --- | --- | --- | --- | --- |
| K1 | Dou et al. 提出 NNS learning image；Gradient × Input 给出当前点输入加权梯度；conductance 给出 reference-path 上的隐藏单元归因 | 为 ViT token-channel、通道聚合、attention head、residual edge 和类别目标建立统一注册及输入坐标实现 | `reports/nns_vit_endpoint_m3_r1_20260919/terminal_receipt.json`：冻结 atlas 含 1,228,800 个 token-channel 目标；`m3a_fidelity.json`：5 类兼容网络、5,000/5,000 公式回归；`m3a_operator_validation.json`：88 个算子方向 | $x\odot\nabla_xs(x)$ 身份和链式法则是方法界定，不是主要创新 |
| K2 | GlobEnc、DecompX、AttnLRP、LibraGrad 已传播 Transformer token contribution 或 relevance；一般 path attribution 已存在 | 对冻结 ViT 实现的真实 Q/K/V、head、residual identity/branch 边建立实际 Jacobian 的切面记账 | `reports/nns_vit_endpoint_e3_closure_r2_20260918/validation_report.json`：76,800 个正向闭合、28,800 个遗漏/重复负控；`reports/nns_vit_endpoint_e4_intervention_r3_20260918/terminal_receipt.json`：6,144 个局部组件检查 | 完整性仅指指定计算图合法切面上的局部微分闭合；不表示函数值完整性、因果完整性或实现不变性 |
| K3 | Activation patching 直接测量 reference-specified finite effect；attribution/edge patching 与 AtP* 用局部梯度和位移近似有限替换 | 在两个固定 ViT checkpoint 上把结构路径层深统计、参数状态对照和规定有限替换任务置于统一验证流程 | `reports/nns_vit_endpoint_m3_r1_20260919/atlas_tests.json`；`reports/nns_vit_endpoint_m4_r1_20260919/m4_tests.json` 与 `residual_block_summary.csv`；`publication_r2_20260920/evidence/supplement_v1/training_analysis.json`；`reports/nns_vit_endpoint_e6_release_r3_20260918/family_decisions.json`；`publication_r2_20260920/evidence/supplement_v1/m2_inherited.json` | 经验范围限于两个固定模型、冻结样本/目标和已执行替换算子；不宣称普遍优越或训练因果效应 |

## 可进入正文的冻结主张

| Claim | 冻结表述 | 状态与证据 | 禁止扩大为 |
| --- | --- | --- | --- |
| C1 数学身份 | 在精确当前点局部导数且无额外归一化、截断或 relevance redistribution 时，标量 NNS learning image 等于同目标 Gradient × Input。 | `SUPPORTED_IDENTITY`；命题 1 与 `m3a_fidelity.json` | 新基础归因公式；一般函数值完整分解 |
| C2 结构闭合 | 在冻结 ViT 实现的指定合法切面上，Q/K/V、multi-head 及 residual identity/branch 项可加和恢复目标的局部输入微分。 | `SUPPORTED_CORRECTNESS`；E3 76,800 正向检查、28,800 负控及 E4 6,144 局部检查 | 闭合证明有限干预效用；首次提出 Transformer 路径分解 |
| C3 分解非唯一性 | 路径项依赖指定计算图、切面和参数化；等价线性重参数化可保持总 Endpoint 不变而改变分项。 | `SUPPORTED_THEORETICAL_COUNTEREXAMPLE`；双语 Section 4.4 的 $R$ / $R^{-\top}$ 构造 | 一般实现不变性；仅由端到端函数决定的唯一解释 |
| C4 层深规律 | 两个固定 checkpoint 的冻结抽样目标呈空间扩展；深层 attention 的 routing 相对尺度与 Value-only error 提高，attention 与 MLP residual 呈报告的不同趋势。 | `SUPPORTED_IN_FROZEN_SAMPLE`；M3 `atlas_tests.json`、M4 结果与正文 Figures 6–10 | 所有 ViT、全部内部单元或未量化的 Q/K/V 空间分布规律 |
| C5 参数状态 | 训练参数状态与三个随机参数状态在空间扩展幅度和部分路径趋势上存在报告的差异。 | `SUPPORTED_PARAMETER_STATE_DIFFERENCE`；`training_analysis.json`，32 共享图/模型、3 随机状态、12 blocks | “训练强化/导致”该规律；训练过程的因果效应；多个训练 checkpoint 的一致趋势 |
| C6 成分排序 | 在规定成分替换中，局部路径图与有限效应的关联广泛但非普遍：12 个 model–reference–family 中 11 个满足原 family 条件，原 all-family omnibus 保持 `NOT_SUPPORTED`。 | `SUPPORTED_WITH_BOUNDARY`；`family_decisions.json` 与正文 Table 4 | 所有 family 均支持；局部图等于有限因果效应 |
| C7 区域排序 | 在规定 reference、block/head、$4\times4$ 候选池和最高 25% 面积预算下，regional Taylor 的排序优于 clean-only Endpoint；Endpoint 的原区域效用复合主张未获支持。 | `SUPPORTED_TASK_COMPARISON`；正文 Tables 6–8、`m2_inherited.json`；原 C4b=`NOT_SUPPORTED` | 无条件方法优越性；一阶误差随干预幅度变化的边界 |

## RQ4 冻结合同

RQ4 冻结为：局部路径图能否在规定的有限替换任务中排序成分和空间区域。正文只使用已执行设计：两个固定 checkpoint、冻结图像和目标、已注册组件，block 5 的三个 heads 与补充 blocks 2/5/10，normalized-zero/blur references，完整替换，$4\times4$ 候选区域以及最高 25% 面积预算。估计量为成分级 Spearman、Capture/Regret 和区域 absolute-effect AUC。

当前证据不包含 $\alpha$ 多幅度扫描，因而不提出“一阶近似在多大扰动范围有效”或“误差随幅度增长”的主张。

## 变更规则

以下表述保持禁用：新基础归因算法、链式法则创新、一般实现不变性、路径分解唯一、Q/K/V 总体空间分布规律、训练强化/训练因果效应、未限定的有限干预效用、普遍方法优越性。若后续实验改变任一结论，应保留本清单作为 v3 第一步快照，新增带日期的后继版本，不在本文件中追写结果。
