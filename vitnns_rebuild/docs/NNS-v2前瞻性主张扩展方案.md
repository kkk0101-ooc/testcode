# NNS-v2 前瞻性主张扩展方案（Stage X）

> 建立日期：2026-09-08。
> 状态：`PLANNING_BASELINE`；尚未完成 X0 protocol freeze，尚未授权新数据下载、模型引入、GPU 计算或 formal outcome access。
> 当前唯一下一任务：`X0_PROTOCOL_AND_LITERATURE_FREEZE_ONLY`。
> 上游状态：C10、Stage D D0–D10、历史 Stage E E0–E4 和 Stage F 已封存工作包均保持只读。
> 下游状态：`F3_FINAL_FIGURE_PACKAGE_ONLY` 与 `STAGE_G_WRITING_G0_ONLY` 暂缓，等待 X6 决策。

## 1. 目的与路线决议

现有证据足以支撑限定于固定 ViT-B/16、fixed-clean-attention head-direction target、primary `(0,9)`、matched patch baseline 和 fine-4×4 fidelity endpoint 的论文。Owner 选择在完成最终图包和论文写作前，前瞻性检验更广的 class-logit、input-resolution、spatial、localization、跨模型和跨数据集主张。

Stage X 是独立研究，不是 Stage F 补图。它不得改变以下历史事实：

- C10 acceptance `BLOCKED`、decision `NO_GO_NOVELTY`、spatial `FAIL_SPATIAL`、Holm `HOLM_BLOCKED_UNDERSPECIFIED`；
- Stage D `GO_FIDELITY_SCOPE` 及其 500 图、固定 `(0,9)`、fine-4×4 scope；
- Stage E scientific validity `MIXED`、E2 budget waiver 和 E3 orders `117`、`352` typed failures；
- F3C 50 图注册主对比 `class-targeted NNS - ViT Grad-CAM` 显著为负。

Stage X 的结果可以形成新的前瞻性结论，但不能回写、重命名或覆盖上述结论。若 Stage X 未通过，项目恢复现有 scoped paper 路线，并从已封存 Stage F 工作包执行 F3-Final。

## 2. 大主张的可检验定义

| 研究目标 | Stage X 可检验命题 | 允许的论文措辞 |
|---|---|---|
| 广泛优于 ViT 可解释方法 | 在 X0 截止日冻结的 comparator、模型、数据集和指标集合上，NNS-v2 的总体 faithfulness 更高，并在预设比例的 model×dataset cells 中保持优势 | `outperformed the evaluated baselines across the preregistered benchmark` |
| 空间稳定 | 在冻结的几何和光度扰动下，逆对齐 attribution 达到绝对稳定阈值，并相对最强注册基线满足优越或非劣标准 | `stable under the preregistered perturbation suite` |
| 完整类别解释 | 对原模型 raw class logit 提供覆盖冻结计算图、满足数值 completeness 和路径 accounting 的端到端 attribution | `end-to-end class-logit attribution with numerical and architectural completeness` |
| 输入像素分辨率 attribution | attribution 直接定义在输入 tensor 的 `[224,224]` 像素坐标，不由 patch replication 或显示插值产生，并通过亚 patch 分辨率验证 | `input-pixel-resolution attribution` |
| 优于 Grad-CAM | 在同一 class-logit target、输入、扰动 evaluator、统计单位和 formal cohort 上，NNS-v2 的注册主指标优于 ViT Grad-CAM | `outperformed ViT Grad-CAM on the preregistered endpoint` |
| Object localization | 在真实 segmentation mask 或 bbox 上，NNS-v2 的注册定位指标优于基线 | `improved localization on the evaluated annotated datasets` |
| 跨模型/跨数据集泛化 | 在冻结的 model×dataset cells 上方向一致，并通过分层统计和最低 cell-coverage 门槛 | `generalized across the evaluated models and datasets` |

有限实验不能证明逻辑意义上的“所有现有方法”“所有 ViT”或“完整语义理解”。Stage X 的最高级结论为 `GO_BROAD_BENCHMARK_SCOPE`，不用 `UNIVERSAL_SUPERIORITY`。

## 3. 当前证据为何不能直接支持这些主张

1. Stage D 主方法解释固定 `(0,9)` head-direction Value-path scalar，不解释原模型 class logit。
2. F3C 的 class-targeted NNS 冻结所有 clean attention tensors；它解释条件模型，不是 unmodified model 的完整计算图。
3. Stage D 的正式 endpoint 是每个 16×16 patch 内的 non-overlapping fine-4×4 cells，不是单像素。
4. F3C 50 图注册主对比中，`NNS - ViT Grad-CAM` mean 为 `-0.0558543281`，95% CI `[-0.0708505871, -0.0410011760]`，unadjusted sign-flip `p=0.0000099999`。扩大同一方法的样本量不能解决这一方向。
5. C10 的 4-pixel translation 和 center-crop/resize stability 失败；新实验必须使用新方法、新 cohort 和前瞻性门槛。
6. 当前 formal model binding 只有一个 ViT-B/16 checkpoint。仓库虽有多个权重文件，但尚未证明它们代表多个独立、可比较模型。
7. 当前 `assets/data/ILSVRC2012_val/` 有 1000 张 JPEG；仓库内没有可用于正式 localization 的 bbox、segmentation mask 或 XML annotation。

因此 Stage X 先解决方法定义与验证，再扩大样本、模型和数据。

## 4. NNS-v2 方法合同

### 4.1 解释目标

Primary target 固定为 clean input 上原模型的 predicted-class raw logit。每个样本同时记录 true class、predicted class、runner-up class 和预测 margin。True-class analysis 与 predicted-class analysis 分线报告，不共享统计分母。

NNS-v2 使用 unmodified model forward。Patch embedding、class/position embedding、Q、K、V、softmax attention、head concatenation、attention output projection、residual、MLP、LayerNorm、bias 和 classifier head 均纳入计算图与路径 accounting；dropout/drop-path 在冻结的 evaluation mode 下处理。任何冻结 attention、替换算子、detach 或 surrogate forward 都定义为单独条件变体，不得作为 NNS-v2 primary。

### 4.2 输入像素分辨率输出

Primary scientific attribution 为与 normalized input tensor 对齐的 signed `[3,224,224]` array；论文使用的 signed `[224,224]` map 由冻结的 RGB channel reduction 从该数组产生。科学 map 不经过 resize、插值、patch replication、max pooling、smoothing、threshold 或 colormap。

“Input-pixel-resolution”通过以下条件验收：

- 每个输入坐标具有独立 attribution entry；
- RGB reduction 前后 signed mass 在冻结容差内守恒；
- 1×1、2×2、4×4、8×8、16×16 排序/干预结果按冻结协议报告；
- synthetic sub-patch fixtures 能区分同一 patch 内不同像素位置；
- 模型参数随机化和 target-swap 后 attribution 发生预期变化；
- display transform 与科学 map 在类型和 API 上隔离。

真实图像的 fidelity 曲线仍可使用固定 21 个累计 fraction 点；每个 fraction 的 mask 从单像素排序构建，不要求逐像素运行 50,176 次 forward。

### 4.3 Learning-image decomposition

NNS-v2 必须提供超出普通 input IG 重命名的结构化分解。X0 应冻结基于 path-integrated contribution/conductance 的精确定义，X1 证明以下恒等关系：

1. total input attribution 的和逼近 `logit_c(x)-logit_c(x0)`；
2. 对每个注册 computational cut，cut 内 neuron/component maps 的和重构同一 total attribution；
3. multi-head attention cut 的 head maps 经原生 concatenation/output projection accounting 后重构 attention branch；
4. residual identity、attention branch 和 MLP branch 在各自切点不遗漏、不重复；
5. layer-wise maps 是对同一总归因在不同 cut 的视图，不跨层相加，避免重复计算同一信号；
6. softmax、LayerNorm 和乘法交互留在原始 autograd path 中，或使用 X0 明确定义且经 synthetic proof 验证的传播规则。

X1 若不能同时通过 numerical completeness 和 architectural accounting，`PASS_CLASS_LOGIT_COMPLETENESS` 不成立，后续只能把方法称为 partial attribution。

### 4.4 必须输出的数值对象

- total signed pixel attribution；
- positive class-support 和 negative contribution views；
- 每个注册 cut 的 neuron/component maps；
- per-head decomposition；
- attention/residual/MLP branch decomposition；
- reconstruction residual、completeness error 和 convergence schedule；
- target、model、input、baseline、code、dependency 和 random-seed provenance；
- structured `VALID` / `FAILED` / `NA` terminal。

## 5. Comparator registry

X0 必须执行截至协议冻结日的系统文献检索，并冻结纳入/排除标准、官方论文、官方实现 commit、license 和模型兼容性。不得在 formal outcomes 可见后增加或删除 comparator。

### 5.1 Class-targeted primary family

最低候选集为：

- ViT Grad-CAM；
- full-model Integrated Gradients；
- Transformer Attribution / Chefer et al., CVPR 2021；
- ViT-CX, IJCAI 2023；
- AttnLRP, ICML 2024；
- LeGrad, ICCV 2025；
- RISE 和/或 occlusion，作为 model-agnostic class-score reference；X0 必须在 outcome access 前冻结最终纳入方式。

只有能对冻结模型产生 class-targeted spatial attribution、具有可核验实现且满足依赖/license 合同的方法进入主 family。方法特有的 layer choice、aggregation 和 normalization 使用论文默认值或独立 development split 冻结，不能用 formal outcome 调参。

X0 的候选官方来源种子清单如下；X0 仍需记录检索截止日、检索式、版本、实现 commit 和纳入/排除理由：

- Transformer Attribution / Chefer et al., CVPR 2021：<https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html>；
- ViT-CX, IJCAI 2023：<https://www.ijcai.org/proceedings/2023/174>；
- AttnLRP, ICML 2024：<https://proceedings.mlr.press/v235/achtibat24a.html>；
- LeGrad, ICCV 2025：<https://openaccess.thecvf.com/content/ICCV2025/html/Bousselham_LeGrad_An_Explainability_Method_for_Vision_Transformers_via_Feature_Formation_ICCV_2025_paper.html>；
- ROAD evaluator：<https://github.com/tleemann/road_evaluation>；
- Sanity Checks for Saliency Maps：<https://arxiv.org/abs/1810.03292>；
- ImageNet-S dataset：<https://github.com/LUSSeg/ImageNet-S>。

### 5.2 Secondary reference family

- attention rollout；
- raw CLS attention；
- random map；
- center prior；
- edge/gradient prior。

Class-agnostic 方法和无模型先验单列报告，不进入“class-targeted methods”主检验家庭。

### 5.3 共同执行合同

每个 model×dataset cell 中，所有方法共享：

- 输入图像、预处理和 target class；
- unmodified model checkpoint；
- perturbation tensors 和 corruption baselines；
- fraction grid、tie priorities 和 evaluator；
- image-level statistical unit；
- failure/NA、retention 和 resource accounting；
- 不进入定量链的 display transform。

## 6. 模型与数据矩阵

### 6.1 模型

最低跨模型矩阵为：

- ViT-S/16；
- ViT-B/16；
- DeiT-S/16；
- DeiT-B/16。

ViT-B/32 用于 patch-size replication。Swin-T 作为 hierarchical/windowed architecture 扩展，仅在 X1 完成 shifted-window、patch-merging 和相应 decomposition 适配后进入 formal matrix。若不执行 Swin-T，论文把架构范围写为 evaluated ViT/DeiT models。

每个模型必须独立冻结：

- exact architecture/model ID 和 source；
- checkpoint file SHA-256、canonical tensor digest 和 parameter count；
- preprocessing、input resolution 和 label mapping；
- native forward/hook parity；
- class-logit target parity；
- NNS-v2 completeness；
- comparator adapter tests；
- X2 resource profile。

### 6.2 数据

最低数据矩阵为：

1. ImageNet-1k：class-logit faithfulness 主实验；
2. ImageNet-S：pixel-mask localization；
3. ImageNet-V2：同类别空间的 distribution-shift replication；
4. 一个独立任务数据集：CUB-200-2011、Oxford-IIIT Pet 或 Stanford Cars，由 X0 根据 annotation、license 和 checkpoint availability 冻结。

ImageNet-S 是 ImageNet 衍生定位数据，不能单独承担跨数据集分类泛化。独立任务数据集必须使用已冻结且在该任务上有效的 checkpoint；如需 fine-tuning，训练协议、split、seed 和 checkpoint selection 在解释 outcome access 前冻结。

当前本地 1000 张 ImageNet JPEG 及既有 C10/Stage D/F3C 样本默认只用于历史证据或 development。只有 prior-access ledger 明确证明未用于方法选择的样本才能进入新 formal cohort。新增数据需记录来源、版本、license、下载校验、样本 manifest、标签和 annotation hash。

### 6.3 样本隔离

- X1：synthetic fixtures 与少量已访问 audit images；
- X2：每个主模型 30–50 张 development images，不进入 formal statistics；
- X3：ViT-B/16 上新的 500–1000 图 ImageNet faithfulness cohort，以及预冻结的 ImageNet-S localization cohort；
- X4/X5：使用各 cell 预冻结、与 development 隔离的 formal manifests；
- 样本选择不得读取 attribution、method difference、localization score 或 stability outcome。

## 7. 六类评价合同

### 7.1 Faithfulness

至少报告：

- insertion AUC；
- deletion AUC；
- sufficiency；
- comprehensiveness；
- ROAD MoRF/LeRF。

X0 选择一个 primary faithfulness statistic（优先评估 ROAD combined score 是否适用）；若使用多个 co-primary endpoints，必须同时冻结 family-wise decision rule。Primary corruption/evaluator 与 NNS-v2 path baseline 分别冻结；black/zero、channel mean、Gaussian blur 和 inpainting/ROAD debiasing 作为预注册 sensitivity family。方向统一为“更高表示更好”或在数据表中显式记录 direction。Primary target 为 clean predicted-class raw logit。

### 7.2 Spatial stability 与 equivariance

几何扰动至少包括：

- horizontal/vertical translation `±4`、`±8` pixels；
- center crop/resize；
- horizontal flip；
- rotation `±10°`。

光度扰动至少包括 Gaussian noise、brightness、contrast 和 JPEG compression。几何 attribution 逆变换至共同坐标，只在共同有效域比较。指标至少包括 signed Spearman、top-k Jaccard、SSIM、attribution-mass displacement 和 center-of-mass error。

每项同时报告 all-sample fixed-target、prediction-unchanged subset 和 prediction flip rate。C10 translation/crop failure 保持历史状态；NNS-v2 只能形成新的版本化结果。若方法包含 transformation averaging，测试集必须包含未参与 averaging 的 holdout transformations。

### 7.3 Input-pixel-resolution validation

除第 4.2 节的 provenance 和 synthetic tests 外，正式数据报告 1×1、2×2、4×4、8×8、16×16 intervention sensitivity。单像素结论要求 attribution 排序单位为输入像素，不能由 14×14 或 56×56 map 插值形成。

### 7.4 Class specificity 与 completeness

至少测试：

- predicted class、runner-up class 和 true class 的 target swap；
- class-logit numerical completeness；
- 删除 positive support 后目标 logit 的下降；
- 删除 negative contribution 后的方向响应；
- 从 classifier 向输入逐级执行 model-parameter randomization；
- 必要时执行独立 label-randomization model，但不得把该训练成本混入 primary benchmark。

随机化结果以 map similarity 和 target sensitivity 报告；视觉变化不能替代数值判据。

### 7.5 Object localization

优先使用 pixel segmentation mask。Primary localization endpoint 建议为 class-macro pixel AUPRC；energy inside ground-truth mask、pointing game 和预冻结 top-k mass IoU 为 secondary。若使用 bbox，结果单独标记为 box localization，不与 mask 指标池化。

Primary localization population 为 target 与标注类别一致且模型预测正确的预注册样本，同时报告完整 manifest 数、correct coverage、每类 coverage 和 all-sample secondary。正向 class-support attribution用于 primary；负向 contribution 单列。Faithfulness 与 localization 是两个不同结论，不互相替代。

### 7.6 跨模型与跨数据集

层级为：

```text
image → class → model×dataset cell → registered benchmark aggregate
```

每个 cell 报告 paired effect、95% CI、有效/失败/NA、正确率覆盖、耗时、显存和保留空间。不同 cell 不简单池化图像；总体结果使用 X0 冻结的分层估计或 meta-analytic aggregation。

## 8. 统计和决策合同

### 8.1 统计原则

- statistical unit 为 image；tie repeats、扰动重复和多个 target 不增加独立 `n`；
- 主分析使用 paired image-level differences；
- class imbalance 使用 class-macro summary 或分层 bootstrap；
- 每个 class-targeted comparator 构成预注册 comparison family；
- secondary contrasts 使用 Holm 或 X0 冻结的 closed-testing procedure；
- 未达显著不等于等价；noninferiority 需要在 formal outcome access 前冻结具有科学含义的 margin；
- required-primary invalidity 抑制 complete-case effect claim，failure/NA 不补零、不删行、不换样；
- development pilot 只用于实现、方差、预算和 margin 校准，不进入 formal effect estimate。

### 8.2 分级决策

Stage X 分别输出：

1. `PASS_METHOD_VALIDITY`；
2. `PASS_CLASS_LOGIT_COMPLETENESS`；
3. `PASS_INPUT_PIXEL_RESOLUTION`；
4. `PASS_FAITHFULNESS`；
5. `PASS_SPATIAL_STABILITY`；
6. `PASS_LOCALIZATION`；
7. `PASS_CROSS_MODEL`；
8. `PASS_CROSS_DATASET`；
9. `GO_BROAD_BENCHMARK_SCOPE`、对应 `NO_GO_*` 或 `INCONCLUSIVE_*`。

这些状态独立保存。例如 localization PASS 不能覆盖 faithfulness FAIL，cross-model PASS 不能覆盖某个 comparator 的 primary inferiority。

### 8.3 `GO_BROAD_BENCHMARK_SCOPE` 推荐门槛

X0 必须在 formal outcome access 前冻结最终数值阈值。推荐结构为：

- 对每个注册 class-targeted comparator，benchmark-level primary faithfulness paired difference 的 95% CI lower `>0`；
- 至少 80% 的 model×dataset cells point estimate 为正；
- 任一 required cell 均未超过预冻结 inferiority margin；
- localization co-primary 对 Grad-CAM 的 CI lower `>0`；
- spatial stability 达到绝对阈值，并相对 X0 预指定的 stability reference 至少 noninferior；不得按 formal 结果事后选择“strongest baseline”；
- completeness、input-pixel provenance 和 sanity gates 全部通过；
- expected/valid/failed/NA 与所有预算状态完整。

通过后允许声称在注册 benchmark 上取得广泛优势，不使用逻辑意义上的 universal claim。

## 9. Stage X 执行顺序

### X0：protocol、literature 与 benchmark freeze

在读取任何新 empirical outcome 前完成：

- systematic comparator search、官方论文/实现/commit/license 清单；
- NNS-v2 数学定义、computational cuts 和 completeness tolerances；
- 模型、checkpoint、数据集、样本 split 和 prior-access ledger；
- target、sign、baseline、fraction、perturbation 和 localization contracts；
- primary/secondary endpoints、family、bootstrap/test、multiplicity 和 truth table；
- failure/NA/retry、retention、资源预算和停止规则；
- closed schemas、typed plan、synthetic fixtures、false-pass audit 和 `X1_IMPLEMENTATION_ONLY` authorization。

X0 不下载数据、不加载模型、不读取 formal images、不运行 attribution。

### X1：NNS-v2 implementation and validation

只使用 synthetic fixtures 和冻结的已访问 audit images，实现并验证：

- unmodified class-logit target；
- total pixel attribution；
- neuron/head/layer/branch decomposition；
- numerical and architectural completeness；
- native-forward、target-swap、sub-patch、randomization 和 tamper tests；
- comparator adapters 和统一 evaluator；
- typed terminal、provenance、resource instrumentation 和 retention。

X1 terminal PASS 只授权 X2 pilot。

### X2：独立 development pilot

每个主模型使用 30–50 张不进入 formal statistics 的样本。Pilot 用于检查 numerical stability、选择冻结 IG/convergence schedule、测量资源、估计方差和冻结 effect/noninferiority margins。

停止门为：

- completeness、class sensitivity、input-pixel provenance 或基础 stability 任一失败；或
- NNS-v2 对 Grad-CAM 的方向仍明显为负，且达到 X0 冻结的 futility boundary。

触发停止门后发布 immutable `NO_GO_BROAD_CLAIM_DEVELOPMENT`，恢复现有 scoped paper；不得通过调色、平滑、挑层、挑图、删除困难样本或反复更换 endpoint 继续同一确认性路线。

### X3：ViT-B/16 formal benchmark

只在 X2 PASS 后执行：

- 新的 500–1000 图 ImageNet-1k faithfulness cohort；
- 预冻结的 ImageNet-S localization cohort；
- 全部 class-targeted comparators 和 secondary references；
- faithfulness、stability、input-resolution、class specificity、localization、sanity 和 efficiency；
- 独立 CPU aggregation、statistics、decision 和 sealing。

X3 先判断 NNS-v2 是否解决现有 F3C 的 Grad-CAM 负向结果。X3 未通过时不执行 X4/X5。

### X4：跨模型复制

在冻结的 ViT-S/16、ViT-B/16、DeiT-S/16、DeiT-B/16 上运行同一合同。ViT-B/32 和 Swin-T 作为预注册扩展 cell。每个 cell 独立验收，不能以其他模型的强结果抵消失败。

### X5：跨数据集与 distribution-shift 复制

依次执行 ImageNet-V2 和一个独立任务数据集。模型适配或 fine-tuning 与解释方法评估使用分离的开发和正式 split。

### X6：统计、决策与封存

X6 只消费 sealed X3–X5 evidence，生成分 cell、分 endpoint 和 benchmark-level 结果，签发第 8.2 节状态。它不修改任何 Stage C–F root。

### X7：图表回接与写作授权

X7 根据 X6 决策：

1. 将 Stage X 结果作为独立 evidence 加入新的 F3-Final；
2. 更新 Figure 2–6 和 supplementary，保留 Stage D/F3C 的既有结果；
3. 生成新的最终图包、caption/source inventory 和 terminal completion；
4. terminal PASS 后授权 `STAGE_G_WRITING_G0_ONLY`。

## 10. 资源和成本策略

Stage D 500 图两方法实际约 `1.8504` H20-GPU-hours；F3C 50 图五方法 formal compute 约 `0.2083` H20-GPU-hour。它们只能作为实现下界，不能直接作为 NNS-v2、多 comparator、多扰动和多模型预算。

采用漏斗式投入：

1. X0：CPU-only 文献、协议、schema 和 manifest 工作；
2. X1：synthetic 为主，仅允许冻结的少量已访问图；
3. X2：小 pilot，实测每 method×model×image 的 wall/GPU/memory/storage；
4. X3：只在 X2 通过后投入单模型 formal；
5. X4/X5：只在 X3 通过后扩展。

预算公式至少覆盖：

```text
models × datasets × images × methods × targets × perturbations × fractions × repeats
```

X2 completion 必须给出保守预算、每 cell 上限、总 GPU-hour 上限、峰值显存、CPU-hour、下载量和 retained bytes。超预算不自动缩小 formal cohort；变更须形成前瞻性 amendment。

## 11. Evidence、retention 与复现

- 每个阶段使用新 `evidence/stage_x_*` 和 `reports/stage_x_*` root；
- formal evidence 以 numeric arrays、tables、manifests 和 terminal receipts 为权威；
- 图片只从 sealed numeric evidence 渲染，不回流统计；
- L0 保存逐图 scalar/terminal，L1 保存支撑主张的 attribution 和 curves，L2 仅保存预注册 audit subset 的完整 component decomposition；
- 数据、annotation、模型、代码、依赖、随机数和 official implementation 均使用 SHA-256/commit 绑定；
- 每个阶段使用 pending-to-terminal completion、source-recomputing validation 和 false-pass audit；
- formal outcome 一旦形成，失败记录不可覆盖，修复使用新 root 和前瞻性 recovery amendment。

## 12. False-pass 条件

以下任一情况必须阻止 broad claim：

1. 将 F3C 的负向 Grad-CAM 结果隐藏、删除或重新命名；
2. 用现有 Stage D 500 图或已查看 F3C 50 图作为 NNS-v2 confirmatory holdout；
3. formal outcome 可见后改变 NNS-v2、comparator、target、layer、metric、margin 或 sample；
4. 将 class-agnostic attention 与 class-targeted 方法放入同一 primary family；
5. 用逐面板归一化 PNG 计算 metric 或比较幅值；
6. 从 patch/cell map 插值得到所谓 input-pixel attribution；
7. 只展示视觉效果好、正确分类或成功运行的样本；
8. localization 没有真实 mask/bbox，或只用视觉重合作为证据；
9. 将 localization、faithfulness、stability、completeness 中一项的 PASS 替代其他项；
10. 跨模型/跨数据集时池化全部图像而不报告 cell-level effect；
11. 将无显著差异写成等价；
12. X2 未通过仍启动 X3，或 X3 未通过仍启动 X4/X5；
13. 修改 C10、D10、E4、Stage F 或 F3C 的既有证据和 archive refs；
14. 未完成 terminal completion、source audit 和 Git/LFS publication 就启动 Stage G。

## 13. 与 Stage F、Stage G 的关系

本文件的 2026-09-08 owner 决议取代此前“立即执行 F3-Final”的调度顺序，但不修改历史 authorization token：

```text
现有 Stage F 工作包（只读基线）
→ Stage X / X0–X6
→ X7 回接并执行扩展版 F3-Final
→ Stage G / G0–G7
```

若 X2 或 X3 触发 `NO_GO_BROAD_CLAIM_DEVELOPMENT`，则不再投入 X4/X5，恢复原 scoped 路线：

```text
existing Stage F evidence
→ original-scope F3-Final
→ Stage G
```

Stage G 只能写入 X6 实际通过的 claim levels。未通过的目标作为结果或 limitation 报告，不以计划目标替代实证结论。

## 14. 当前唯一下一任务：X0

下一轮只执行 `X0_PROTOCOL_AND_LITERATURE_FREEZE_ONLY`，交付：

1. 版本化 protocol、acceptance、config/schema 和 typed plan；
2. 截止日、检索式、纳入/排除标准、论文与官方实现 registry；
3. NNS-v2 数学定义、computational-cut diagram 和 completeness proof obligations；
4. 模型/checkpoint/data/annotation inventory 与缺失资产清单；
5. development/formal split 和 prior-access ledger；
6. primary/secondary endpoint、multiplicity、margin 和 truth table；
7. X1 synthetic test specification、retention、预算模板和 false-pass audit；
8. terminal X0 completion；PASS 时只授权 `X1_NNS_V2_IMPLEMENTATION_ONLY`。

X0 不执行 NNS-v2、comparators、formal inference、GPU campaign、定位统计或论文图表。
