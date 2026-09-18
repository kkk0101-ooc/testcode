# 4. 实验设置

## 4.1 模型与预处理

主实验使用冻结的 `vit_base_patch16_224.augreg2_in21k_ft_in1k`。模型含 12 个 transformer blocks，每个 block 含 12 个 attention heads，embedding 宽度为 768；输入为 224x224 RGB，patch 大小为 16x16。Stage D 冻结的 state-dict SHA-256 为 `2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`。R10 单图扩展另记录 checkpoint 文件 SHA-256 `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d`。两者是不同绑定层级的摘要，不在正文中互换。

规范预处理将 RGB 图像短边以 bicubic resize 到 248，随后 center crop 为 224x224，并用 mean 和 standard deviation 均为 $(0.5,0.5,0.5)$ 的变换规范化。Learning image 的 reference 是规范化后的全零张量，对应原 RGB 空间中的均匀灰色 0.5。模型在 evaluation 模式运行。

## 4.2 数据队列与样本角色

Stage D 主队列包含 ImageNet-1k 验证集的 500 张图像。样本在 outcome access 之前通过冻结的元数据算法从 50,000 条候选中排序选取，并与早期 Stage C 样本按 sample ID、路径和图像摘要保持不重叠。全部 500 张图像进入确认性分母，不按预测正确性筛选。

C10 的 15 图 holdout 是较早的开发与门控队列，只作为历史结果报告，不与 Stage D 合并。Stage E 在访问 D10 结果后进行尺度、上采样、层头和预测分层分析，属于 secondary/descriptive analysis。类别解释器比较使用独立的 50 图正式队列。完整切分的分支份额分析使用 100 图队列；Stage X2-MECH 的多尺度分析和 Stage S 的深度扩展按各自冻结有效分母报告。

Figures 28-35 使用冻结样本 `ILSVRC2012_val_00007922.JPEG`。该图的官方类别和模型 clean argmax 均为 class 235（German shepherd）。这一样本只用于计算对应和方法展示，不进入新的总体推断。

## 4.3 数值积分与计算检查

Stage D 的 Value-path learning image 使用 float32 Gauss-Legendre quadrature，阶数按 $[512,1024,2048,4096,8192]$ 递增，并保留第一个满足 conservation 阈值的结果。每个图像的方法对共享 clean input、reference、target、空间投影、fractions 和 tie priorities。任何方向退化、target delta 退化、非有限值、形状错误或缺失 artifact 都按冻结类型记为 invalid，而不换样或填零。

R10 的真实模块页使用 float64 CPU 计算。B0 MLP 与第二 residual 在 128 阶通过 convergence 和 completeness；完整 classifier 路径因非线性更强，递增到 4096 阶。所有 residual identity 都在未着色 RGB 数组上逐坐标验证。

## 4.4 主干预协议

主目标固定为 block 0/head 9 的 fixed-clean-attention CLS head-direction Value path。NNS 与同目标 patch-token 基线都投影到 3,136 个 4x4 cells。Insertion 从灰色 reference 恢复单元，deletion 从 clean input 替换单元；每个方向包含 21 个 fractions 和 5 个共享 tie repeats。图内先对 repeats 求均值，再形成唯一的图像级 delta-fidelity。

确认性汇总报告 positive count、mean、median、sample standard deviation 和 10,000 次图像级 paired-median percentile bootstrap 95% CI。冻结 PASS 规则要求全部 500 图有效、至少 400 个严格正差值、median 不低于 0.02 且 CI 下界大于 0。C10 与 Stage D 各自保留独立的决策状态。

## 4.5 Secondary/descriptive sensitivity

Stage E 在同一 500 图队列上比较 4x4-cell 与 native 14x14-patch 评价，并描述四个预定坐标 $(0,9)$、$(4,2)$、$(6,10)$ 和 $(11,2)$。预测分层按模型是否正确分类报告。Stage E 的结果不重选主坐标，也不改变 Stage D 决策。E2 的实际成本超出原预算后使用了明确、仅限该阶段的 owner waiver；E3 在 block 11/head 2 保留两个 IG conservation typed failures，因此 E4 的 package/workflow 通过而 scientific validity 为 `MIXED`。

## 4.6 独立类别解释器比较

50 图比较中的 NNS 目标是所有 blocks 的 attention 固定在 clean 状态时的条件类别 logit。Value tensors、head concatenation、output projections、residual paths、MLPs、final norm 和 classifier 保持活动。四个冻结对照为 ViT Grad-CAM、full-model IG、attention rollout 和 raw CLS attention。前两者为类别相关方法；rollout 和 raw CLS attention 不使用类别目标。

Primary contrast 是 NNS minus ViT Grad-CAM，使用未校正的注册检验。NNS minus full-model IG、rollout 和 raw CLS attention 构成 Holm 校正的 secondary family。每个 contrast 的统计单位均为图像，报告 mean、bootstrap 95% CI 和注册 p value。逐面板归一化 atlas 只用于空间形态观察。

## 4.7 完整切分与深度分析

完整切分实现首先验证总 NNS 图与同合同 full-model IG 的 identity，再在 135 个 cuts、38,107 个 components 上验证逐 cut 重构。Stage Y 将完整类别贡献分别切分为第一处 residual 的 identity/attention 和第二处 residual 的 residual/MLP。对每个切分，分支绝对贡献份额以该切分两项 absolute mass 之和为分母；两处切分的份额不相加。

深浅层对比和其他机制端点按冻结 family 使用 Holm 校正。六个 claim 分别输出 `SUPPORTED`、`NOT_SUPPORTED`、`INCONCLUSIVE` 或 `INVALID`，不合并为总体 PASS。

## 4.8 稳定性与事后探索分析

Stage X 在 200 条预期 clean routes 上评价几何稳定性，其中 199 条有效；预定 broad-claim gate 未通过。随后 M2 在 CPU-only 条件下对既有 signed maps 进行 pixel、4x4-cell 与 native-patch 聚合、SSE 分解和跨尺度 association 分析，标记为 post-hoc exploratory。

Stage S 同样作为 post-hoc exploratory 分析，报告目标区域对齐、matched occlusion、深度对齐差值和线性 probe gain。P2 的最终阈值在 outcome access 后确定；原始更严格规则未通过。该分析用于生成后续假设，不改变 Stage D、Stage X 或 Stage Y 的原决策。

## 4.9 可复现性与证据绑定

模型、图像、样本顺序、目标、数值积分、失败类型、统计 seeds 和输出摘要均记录在机器可读 manifest、receipt 与 inventory 中。R10-R4 图包含 35 张 PNG/PDF/SVG，terminal status 为 `PASS_PRESENTATION_RENDER`，且声明 `new_inference=false`。稿件中的数值从冻结 JSON/CSV 读取；PNG 仅用于排版和人工查看。
