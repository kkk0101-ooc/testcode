# 修改意见 v3 第二步防御性写作审计

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 双语报告多幅度结论 | 小幅干预时预测更接近真实效应，幅度增大时误差扩大 | CLEAN | — | 是 | 直接对应 1600/1600 单调曲线和排序下降 | 结论与数据一致 | 保留 | 不变 |
| 双语报告 Q/K/V 结论 | Q/K/V 具有不同的总体层深轨迹，不能概括为三者统一扩散或收缩 | NECESSARY_CAVEAT | — | 是 | 后半句防止把混合方向压成单一结论 | 会要求保留路径差异 | 保留一次 | Q/K/V 具有不同的总体层深轨迹 |
| 双语报告独立抽样结论 | 该结果将结论从单一抽样目标推进为两个固定 checkpoint 的五组独立抽样中稳定复现 | CLEAN | — | 是 | 积极陈述已评价范围 | 范围清楚 | 保留 | 不变 |
| 双语报告结尾 | 适用范围仍为两个固定 ImageNet checkpoint 和冻结 cohort | NECESSARY_CAVEAT | — | 是 | 该限定改变外推解释 | 会接受范围限定 | 只在结尾出现一次 | 不变 |
| 后继主张清单 | 不支持给三条路径赋予统一空间变化方向 | MIXED | Minor | 是 | 负式表达，但直接区分支持与不支持的经验方向 | 能快速识别主张边界 | 清单中保留；正文改为正式表述 | 三条路径呈不同空间变化方向 |
| 统一协议恢复段 | 原根与 R1 根分别因回放门 fail-closed | CLEAN | — | 是 | 属于可复现性与数据继承事实，不进入论文结果解释 | 工程细节放置合理 | 保留在审计文档 | 不变 |

### Paper-level defensive-writing score

**1/10。** 双语报告使用“结论 → 数值证据 → 一句适用范围”，没有 reviewer-facing prebuttal、结果辩解、促销形容词或遗漏实验的长篇自我防卫。

### Highest-priority removals

1. 第三步整合主文时，不重复原根/R1 恢复过程；仅引用最终 R2 验证收据。
2. 不把 Q 的跨模型异向写成“结果仍然有意义”；直接报告 ViT-B 增、DeiT-S 近零。
3. 不在摘要、结果、讨论和结论四处重复“只覆盖两个 checkpoint”；摘要或结尾保留一次即可。

### Necessary caveats to preserve

- 一阶边界只对应规定的 head-output 插值变量、两个 reference 和五个幅度。
- Q/K/V 的空间方向并不统一，尤其 Q 不跨模型一致。
- 五组独立抽样提高了抽样稳定性证据，但总体仍是两个固定 checkpoint。
- 第二步没有增加训练 checkpoint，训练因果措辞继续禁用。

### Repeated phrase audit

双语报告未出现 `We emphasize that`、`It should be noted that`、`We do not claim`、`Taken together`、`需要强调的是`、`应当指出`、`为避免误解` 或 `我们不声称`。`适用范围` 只在结论出现一次。
