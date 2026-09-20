# 修改意见 v3 第一步：防御性写作审计

审计范围：双语题目、摘要、研究问题、贡献列表、相关工作、计算图依赖、RQ4、参数状态结果与结论。采用 `academic-defensive-writing-auditor` 的 claim → evidence → concise boundary 规则。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| 摘要 | NNS 的扩展“而非新的基础输入归因量”只在段末说明 | D14 / MIXED | Major | 是 | 基础身份与技术增量分离过晚，容易先读成重新命名 Gradient × Input | 会追问基础公式是否新颖 | 先给数学身份，再正面陈述真正增量 | 身份用于界定基础量；贡献是 ViT 真实边的统一实现、图相对分解和系统证据 |
| 贡献列表 | 四项贡献把链式法则、实现、结构分解和经验结果并列 | D14 / MIXED | Major | 否 | 定义性结论容易被包装成理论创新 | 会认为贡献拆分夸大 | 按已有工作—新增内容—直接证据重组为三项 | NNS–ViT 实现；指定图与切面的结构记账；层深与规定替换任务的系统验证 |
| RQ4 | “在多大程度上对应有限干预”及“一阶近似边界” | MIXED | Critical | 是 | 问题范围暗示未执行的多幅度误差曲线 | 会判定问题与实验不匹配 | 收窄到实际设计 | 对固定模型、reference、替换算子和预算，评价成分与区域排序 |
| 参数状态结果 | “训练后参数强化其幅度” | D13 / MIXED | Major | 否 | 单训练 checkpoint 与随机状态对照被写成训练过程效应 | 会要求多个训练 checkpoint 或配对轨迹 | 只报告组间差异 | 训练参数状态与随机参数状态存在报告的幅度和趋势差异；不识别训练因果效应 |
| 非唯一性 | 只报告合法切面闭合 | MIXED | Major | 是 | 闭合可能被误读为唯一分解或实现不变 | 会追问等价参数化是否改变路径项 | 给出具体反例并一次界定 | $R/R^{-\top}$ 等价重参数化保持总 Endpoint，但改变隐藏坐标分项 |

### Paper-level defensive-writing score

2/10（修订后）。核心段落先陈述技术对象和证据，再给一个影响解释的边界；没有用审稿人预答辩或宣传形容词补偿范围限制。

### Highest-priority removals

1. 删除把链式法则身份列为主要创新的结构。
2. 删除 RQ4 中未由多幅度实验覆盖的“一阶近似边界”表述。
3. 删除“训练强化”及等价的训练因果语言。
4. 避免以“本文不声称”反复防御，改为正面定义图相对分解和规定任务。
5. 将不同方法的完整性对象写入表头和单元格，避免用一个无条件 Yes/No 制造虚假等价。

### Necessary caveats to preserve

- 标量 NNS 与 Gradient × Input 的等价需要精确当前点局部导数且无额外重分配。
- 路径分解依赖计算图、切面和参数化，不具有一般实现不变性。
- 局部微分闭合不等于函数值完整性或有限干预效用。
- 参数状态对照不识别训练过程的因果效应。
- RQ4 仅覆盖已执行的完整替换、reference、层/head、候选区域和面积预算。

### Repeated phrase audit

双语主文扫描以下套话为 0：`We emphasize that`、`It should be noted that`、`We do not claim`、`To avoid misunderstanding`、`Taken together`、`需要强调的是`、`应当指出`、`为避免误解`、`我们不声称`。
