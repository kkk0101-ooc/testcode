# Revision v6 防御性学术写作审计

审计对象为最终匿名双语稿。依 academic-defensive-writing-auditor 的 D1–D14 分类，按摘要、贡献、结果、讨论和结论复核。保留会改变科学解释的条件，移除内部主张 ID 和回复审稿人式论证。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| §2.5 | 若一个既有方法覆盖这些字段，本文的方法新颖性主张即失效 | D1/D14 | Major | MIXED | 模拟审稿反驳，未增加方法信息 | 注意力从对象差异转向自辩 | 合并为来源范围和直接对象比较 | Table 1 限于所核查来源；比较当前点微分的 use 与位置索引及其审计 |
| 贡献 | 三项贡献反复列数千审计计数与 NOV ID | D10/D14 | Major | DEFENSIVE | 用计数和内部 ID 强化贡献 | 难辨技术贡献与验证工作量 | 方法贡献压缩两点，实验结果另列 | 双索引与闭合合同；针对 use 错误的结构审计 |
| §4.3 | 命题增量不是把证明本身写成创新 | D2 | Minor | DEFENSIVE | 对假想误解预辩解 | 链式法则身份反而不清楚 | 直接说明来源 | 命题由链式法则得到 |
| §6.3 | 总图、scalar、双索引反复证明不可替代性 | D12/D14 | Major | MIXED | 同一信息差异多次重述 | 结果叙事显得像回复意见 | 保留信息表及一次空间量说明 | 熵和 TV 需要指定分项的空间分布 |
| §7.1 | 接近 0.10 的失败和通过互相辩护 | D4 | Major | NECESSARY_CAVEAT | 有用边界但易成为近门槛解释 | 可能误读为将失败合理化 | 保留原规则、实际数值和另一种区间条件 | 0.0996 未满足原点估计规则，all-family 未获支持 |
| §8 | 每段再次列不能解释的对象 | D3/D8 | Major | MIXED | 限制堆叠打断结果 | 看不清最终可回答问题 | 集中一段范围 | 图像区间条件于固定 targets/checkpoints，路径项依赖指定实现 |
| Abstract / conclusion | H4 阴性与坐标敏感性 | D12 check | None | NECESSARY_CAVEAT | 影响增量效用和外推，非防御性 | 能直接判断贡献范围 | 保留并简写 | 分辨率增加未自动提高有限干预预测；DeiT-S Q 敏感 |

## Paper-level defensive-writing score

2/10。仍有必要的证据范围句，主要集中在统计和讨论。双语稿不使用“我们不声称”“为避免误解”“We emphasize that”“We do not claim”等模板。

## Highest-priority removals

1. 主文全部 NOV 和实验阶段标签移出，证据索引继续保留历史定位。
2. 删除相关工作中重复的新颖性失效辩护。
3. 方法贡献与经验结果分开叙述。
4. 将高密度审计计数集中到 Table 3 和结构审计表。
5. 训练、目标和抽样控制的长段落移入补充，正文只保留解释作用。

## Necessary caveats to preserve

Gradient × Input 身份；全局与分支闭合；指定实现下分项非唯一性；normalized-input 原点；图像总体条件；H4 负结果；all-family 未获支持；三坐标中的 Q 例外；Taylor 的额外位移信息。

## Repeated phrase audit

最终自动扫描将报告上述模板和内部 ID 的出现次数。H4 在摘要、贡献、RQ4、结果、讨论和结论重复是本轮明确要求，属于必要结论同步，不按免责声明堆叠删去。所有实例均说明同一经验边界，未增加“普遍有效”或“首次”等表述。
