# G×I 第四步防御性写作审计

审计范围为现役中英文 Results、Discussion 和 Conclusion。采用 `academic-defensive-writing-auditor` 的 claim → evidence → concise boundary 规则。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Section 6.3 | “不是普通 G×I 加 hooks”式辩解 | D1 / D14 | Major | 否 | 从假想质疑而非信息差异出发 | 要求可检验的必要性 | 用总图、edge scalar 与双索引图组的 retained indices 比较 | Table 4 直接列出可恢复问题和实际结果 |
| Section 6.4 | 仅用 routing share 支撑空间重组 | D10 | Major | 是 | 标量尺度不足以证明空间位置变化 | 会认为 $i$ 索引没有实证作用 | 降级 share 为支持证据，增加 400 图 Q/K/V spatial entropy 和 Full–Value patch TV | Q/K/V 总体结果进入主文，16 图原始图组作为独立描述性支持 |
| Section 6.5 | 仅用 identity share 描述 residual 机制 | D12 | Major | 是 | edge scalar 不能排除整体幅度解释 | 会追问双索引必要性 | 用 400 图 output–identity spatial TV 与区间作为核心结果 | Attention 增、MLP 降，逐模型给区间 |
| Section 6.6 | 只报告 H1–H3 正向复现 | D4 / D9 | Critical | 是 | 会形成选择性报告 | 阴性 H4 是解释边界的直接证据 | 同表报告 H1–H4，摘要和结论保留 H4 | 四 checkpoint 的 H4 均为负 |
| Section 7 | 把 Endpoint–Taylor 写成统一方法排名 | D13 / D14 | Critical | 是 | 忽略 clean-only 与 displacement-aware estimand 不同 | 结论跨任务越界 | 单设行为标定与 estimand 边界 | 只在规定 reference/replacement 下比较 |

### Paper-level defensive-writing score

2/10。结果先报告对象、统计单位、效应量和区间，再给一次范围边界；没有用正面形容词补偿阴性结果，也没有删除 `NOT_SUPPORTED`、H4 反向或跨模型相反趋势。

### Highest-priority removals

1. 不用“不是 G×I 加 hooks”作为结果段开场。
2. 不让 routing share 或 identity share 单独承担双索引必要性。
3. 不把四 checkpoint 的 H1–H3 正向结果与 H4 阴性结果拆开报告。
4. 不把 Taylor 的 matched-task 优势改写为普遍方法排名。

### Necessary caveats to preserve

- Attention Q/K/V 空间必要性结果为每模型 400 图；Full–Value 描述性支持为每模型 16 图；residual spatial TV 为每模型、每 residual 类型 400 图。
- E5 新增 checkpoint 是公开训练方案下的参数状态，不是独立训练种子。
- E5 H4 的相反方向与原 all-family `NOT_SUPPORTED` 均须保留。
- 路径图的局部结构表示与 reference-aware finite intervention prediction 是不同 estimand。

### Repeated phrase audit

现役双语主稿未出现 `We emphasize that`、`It should be noted that`、`We do not claim`、`To avoid misunderstanding`、`Taken together`、`需要强调的是`、`应当指出`、`为避免误解` 或 `我们不声称`。
