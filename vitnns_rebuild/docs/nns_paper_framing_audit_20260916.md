# 论文主线与 Figure 1 表述审计

审计对象：论文写作前置条件与结论证据图表映射_20260916.md。依据 academic-defensive-writing-auditor 的 claim → evidence → concise boundary 原则，结合本轮用户对首图价值表达的要求。技术判断另以 Stage C/X 协议和 D10、E4、X1 真实证据核对。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| 1.2 总论点 | 比整块 patch 归因更细的内部解释 | D14 / MIXED | Major | 保留固定目标条件 | 以标准输入归因的分辨率代替技术贡献 | 追问 IG 或更小 patch 是否已经做到 | 把用途与原创性分开 | 将选定内部响应与输入区域的正负贡献联系起来 |
| 3.0 对照定位 | 隔离 patch 内细分价值的受控消融 | 技术过度归因 / MIXED | Major | 必须保留坐标差异 | 把受控条件说得比实现更强 | 质疑逐 patch 总量是否匹配 | 改称同目标 token 空间对照 | 比较输入归因排序与 token 贡献的均匀空间展开 |
| 3.0 表格 | 每个原生 patch 总量，两边都保留 | 技术定义错误 | Critical | 必须修正 | 将 token 贡献等同于输入区域贡献 | 深层跨位置混合使该假定失效 | 删除通用逐 patch 恒等 | 分别声明整体守恒，CLS 单列，不假定逐 patch 相等 |
| 3.1 意义 | 不是把热图画得更平滑或更好看 | D1/D14 / DEFENSIVE | Minor | 否 | 用反驳替代研究用途 | 看不到解释能用于何事 | 直接写观测 | 归因排序与同一内部目标的干预响应相联系 |
| 3.3 head 意义 | 平均所有层头可能淹没结构 | 未检验推断 / MIXED | Major | 保留四坐标条件 | 用可能性解释 head-level 必要性 | 追问是否做了平均对照 | 删去未验证的操作后果 | 四个预定坐标具有不同效应，明确所解释的目标 |
| 3.6 完整对账 | 结果不是漏路径或重复计算产生 | D1 / MIXED | Minor | 保留数值误差和切分条件 | 实现辩护多于可用信息 | 不清楚成分图增加什么分析维度 | 先写成分索引的用途 | 同一总归因按 head、MLP、residual 查看，并在每个完整切分内重构 |
| 5 Figure 1 | 图像→NNS/基线→fidelity→机制 | CLEAN，但信息职责混合 | Major | 无需堆叠全部边界 | 首图变成第二张方法流程图 | 仍不知道为什么要读这篇论文 | 改为价值、实例、证据 | 内部 head → 实测 signed 输入贡献 → 500 图干预结果 |
| 1/9 写作准备 | 科学前置条件已齐 | MIXED | Major | 保留数据已齐与投稿尚未完成 | 可能把数据完备当作新颖性审定 | 追问相对 conductance 的技术增量 | 增加文献技术对照任务 | 可组织论文证据，贡献声明待与既有 neuron/path attribution 逐项核对 |

## Paper-level defensive-writing score

修订前 4/10：主要问题是以反驳和粒度差异解释价值，重复边界次之。修订后 2/10：事实、用途和证据在前，局部必要条件保留在对应节中。这是本映射文档的编辑评分，不是完整论文评审或科学质量评分。

## Highest-priority removals

1. 删除“ViT 的 patch 表示必然丢失内部贡献信息”的总论点。
2. 修正 token c_j 等同输入 patch 总贡献的定义。
3. 删除当前 baseline 为纯 patch 内细节消融的说法。
4. 首图去掉分块教学，改用真实归因案例与正式干预结果。
5. 分别解释 Value-path IG、完整 class-logit cut decomposition 与同合同 IG identity。
6. 把方法文献对照加入正式贡献声明前的待办。

## Necessary caveats to preserve

- 主干预结果的目标、模型、fine-cell 单位与 n=500。
- 正向排序实验不为每个 signed cell 签发独立因果结论。
- token 空间贡献与输入归因坐标不同，尤其深层可跨 patch 混合。
- NNS-v2 total 与同合同 full-model IG 同一；完整切分的成分才是额外分析索引。
- Stage X/Y 的未支持或无效端点，以及 Stage S 的 post-hoc exploratory 性质。

## Repeated phrase audit

未提供完整 manuscript，因此不进行全文重复短语统计。已检查映射文档第 1、3、5、7、8、9、10 节中“patch 内部是总价值”的重复叙事，并将其统一为内部解释用途、细粒度实证端点和方法差异三个层次。
