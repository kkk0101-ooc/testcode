# G×I 第五步防御性写作终审

范围为中英文标题、摘要、引言、贡献、相关工作、结果解释、讨论和结论。采用 academic-defensive-writing-auditor 的 D1–D14 分类。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 标题 | 从 NNS 到 ViT 的宽泛历史叙事 | D14 | Major | 否 | 没有直接呈现技术对象 | 先按普通 G×I 扩展理解 | 标题直接命名双索引对象与审计 | 双索引局部 Jacobian 扫描与结构审计 |
| 摘要 | 用“结构化、可闭合、可审计”概括方法 | D10/D14 | Major | 否 | 属性词代替了对象定义 | 追问新增数学对象 | 给出 $\mathcal L_{e,i}$、$e/i$ 与 valid cut | 已重写 |
| 引言 | 从“总梯度不能区分”直接跳到框架 | D1/D14 | Minor | 混合 | 像在预答“只是 G×I” | 要求明确技术差值 | 正面定义两个自由索引和 target condition | 已重写 |
| Related Work | “每个单项都有先例”后重复非主张 | D2/D12 | Minor | 是一次 | 边界正确但略像新颖性辩护 | 关注是否有可证伪标准 | 改为最强近邻字段和失效条件 | 已增加逐字段反例 |
| 讨论 | “本文的唯一定位是” | D1 | Minor | 否 | 语气像回应审稿人 | 期待事实而非定位宣告 | 直接陈述测量原语与技术增量 | 已重写 |
| 结论 | 用统一框架收束全部证据 | D11 | Minor | 否 | 可能掩盖 closure 与 intervention 不同 | 追问因果边界 | 用两句分别界定正确性和行为用途 | 已重写 |

## Paper-level defensive-writing score

1/10。入口先定义对象，随后给证据和一次边界。未发现 reviewer-facing stock phrase、鼓励性形容词、弱结果辩护或遗漏实验自我辩护。

## Highest-priority removals completed

1. 删除以宽泛历史叙事承担技术定位的旧标题。
2. 用双索引公式替换摘要中的属性词堆叠。
3. 删除讨论中“唯一定位”的预答辩语气。
4. 结论不再用统一框架句掩盖 correctness 与 finite utility 的区别。
5. 最强近邻直接作为反例条件，不用弱比较或单项非主张制造差异。

## Necessary caveats to preserve

- 基础 Endpoint 是同目标 G×I。
- 分项相对于指定图、切面和参数化定义。
- 闭合是局部微分记账，不是有限干预因果正确。
- 四个 checkpoint 是公开参数状态，不是独立训练种子。
- Q 的模型依赖趋势、H4 四项反向和 all-family 未获支持必须保留。
- `NOV-JOINT-1` 受 2026-09-22 文献截止日约束。

## Repeated phrase audit

现役双语主稿中 `We emphasize that`、`It should be noted that`、`We do not claim`、`To avoid misunderstanding`、`Taken together`、`需要强调的是`、`应当指出`、`为避免误解` 和 `我们不声称` 均为 0。
