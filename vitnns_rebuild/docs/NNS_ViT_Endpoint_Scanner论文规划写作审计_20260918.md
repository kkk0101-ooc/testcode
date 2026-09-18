# NNS-ViT Endpoint Scanner 论文规划：写作审计

审计对象：[论文思路、规划与步骤](NNS_ViT_Endpoint_Scanner论文思路与规划与步骤_20260918.md) 和新数学图的图注。采用 academic-defensive-writing-auditor；本页审计论文规划的措辞，不评价投稿可接受性。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| 贡献定位 | 用户材料：不是一种新的基础归因公式 | D14 / MIXED | Minor | 是 | 反复否定会遮蔽技术内容 | 希望直接看到具体增量 | 陈述身份后列接口、分解与评价 | 它等于内部标量目标上的 Gradient × Input。技术内容是目标注册、ViT 真实算子适配、结构分解及其评价。 |
| 原 NNS-CNN 忠实性 | 用户材料：忠实的架构适配 | NECESSARY_CAVEAT | Minor | 是 | 单独使用“忠实”缺少可检验条件 | 需要原公式、目标门控和代码范围 | 绑定局部 Jacobian 和 signed 单元图 | 条件是使用与网络一致的门控/局部 Jacobian、包含所有路径，并比较原始 signed 图。 |
| 现有 20 图 | 用户材料：Endpoint 更适合作为低成本内部扫描工具 | D10 / MIXED | Minor | 是 | 从单次小样本成本结果推广工具优势 | 希望分开历史测量与新研究问题 | 直接列数据和原决策 | 时间比中位数 0.01492 限于该 pilot；新论文的扫描定位需要新的实验问题与预定指标。 |
| 干预预测 | 用户材料：检查 head attribution mass 是否预测 logit 变化 | CLEAN + NECESSARY_CAVEAT | Minor | 是 | 原句不防御，但微分方向尚未绑定 | 需要区分空间贡献与 replacement 方向 | 增加精确公式及强基线 | $a_k=\langle g_k,J_{z_k,x}x\rangle$；reference replacement 的一阶预测是 $p_k^{ref}=\langle g_k,z_k-z_k^{ref}\rangle$。 |
| 数值闭合 | 当前稿：方法身份由链式法则建立；解释是否有用由跨图实验回答 | CLEAN | Minor | 是 | 无防御性重复 | 能区分证明与实现验证 | 保留一次范围说明 | 原句保留；在证据表写明 hooks/cotangents/记账验证。 |
| 多卡状态 | 当前稿：现有演示固定 CPU，EP1 新增 GPU worker/队列 | CLEAN | Minor | 是 | 陈述真实实现状态 | 知道哪些步骤可直接运行 | 保留并提供设计合同 | 当前 API、未来入口及 1 卡/多卡一致性验收分开记录。 |

## Paper-level defensive-writing score

**2/10（编辑判断）**。主线为定义、证据、待执行实验；范围限定集中在方法身份、历史结果、干预方向和执行状态。规划中的操作约束用于复现，不属于对假想审稿意见的回应。

## Highest-priority removals

1. 不把“新基础算法”或“忠实”反复写成口号，直接列 Jacobian 条件与接口增量。
2. 用 20 图的实际数值和原决策替代对性能弱项的推测性解释。
3. 把“干预验证成功”留给有比较指标和区间的结果，规划中保持待验证。
4. 将 CPU 演示与待开发多卡生产能力分开，避免将计划写成实现事实。

## Necessary caveats to preserve

- 原论文单神经元规则的门控一致性、非光滑点约定及官方代码可得性。
- Endpoint、baseline-relative extension、完整 Path IG 与条件 Value-path 的不同身份。
- 向量内积、完整不重叠切面、residual 输入边与 bypass。
- 数值闭合验证实现；有限干预需要独立的方向与实测。
- 20 图 multiplier 语义、原 NO_GO 决策，以及 energy/GPU 入口待实现的状态。

## Repeated phrase audit

规划全文不使用 “We emphasize”“To avoid misunderstanding”“不应误解”“为了公平”“值得注意” 等审稿人回应式引导。方法身份、独立统计单位与多卡规则在对应章节各定义一次，执行表只引用完成条件。数学图保留简短范围说明，以区分解析例子、局部坐标和训练模型实测。
