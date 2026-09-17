# R9 大纲与图注的表述核对

采用 academic-defensive-writing-auditor，检查论文大纲和双语图注。保留目标、样本、参考及推断性质，删除以形容词替代证据的说法。此文件为写作工作记录，不进入论文正文。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| 旧 PPT 背景 | 一套完整框架，揭示学习行为 | D10/D14 | Major | No | 泛化用词替代计算对象 | 不清楚解释什么 | 具体定义 | 给选定 ViT 响应配一张输入坐标贡献图。 |
| 旧 PPT 实验 | 高效地提取判别性区域 | D10 | Major | Yes | 从旧指标扩大结论 | 模型与数据是否对应 | 换为当前实证 | 指定 head 的 500 图 fine-cell 配对差值全部为正。 |
| 旧图讨论 | 不能单凭图断言算法全程正确 | D2/D12 | Minor | Yes | 以否定陈述堆叠证明要求 | 不清楚各实验作用 | 写明各自问题 | 重构检查计算对应，干预检查排序用途，分支统计检查贡献分配。 |
| Figure 2/3 跨图引用 | Figure 3 评价 Figure 2 定义的 f | CLEAN，技术修正 | Major | Yes | Figure 2 已更换解释对象 | 会把演示目标当作主实验目标 | 更新引用 | Figure 3 仍评价原 B0/H9 条件 Value-path；Figure 2 演示模块响应图。 |
| 真实图与原论文 Fig.13 | 学习图与 feature map 高度相关 | D10 | Major | Yes | 借用原 CNN 结论 | ViT 相关性数据何在 | 定义对照量 | 上排是输入位置贡献，下排是内部 token 值；本次为单输入展示。 |

## Paper-level defensive-writing score

编辑判断：2/10。大纲按问题、公式、结果和来源展开，主要限制集中于目标表、具体实验性质和讨论。方法原创性按继承关系描述。

## Highest-priority removals

1. 用计算对象替代“揭示思考、学习行为”等宽泛陈述。
2. 不把旧模型的分类准确率与 EE 并入当前实验。
3. 将单样本图像观察与跨图像统计分别命名。
4. 去除 Figure 3 对旧 Figure 2 目标的过时引用。

## Necessary caveats to preserve

新三模块图的固定输入/通道与 B0 范围；自身响应和类别成分的区别；原生 224×224 与既有 56×56 cell 分辨率；显示尺度；主干预目标与尺度；未检出差异和等效的区别；Y 的 INVALID 分母；X/S 的 post-hoc exploratory 性质和原 P2 规则未通过。

## Repeated phrase audit

精确短语计数见 R9 的 writing_phrase_audit.json。对“We emphasize”“For a fair comparison”“We do not claim”及对应中文防御性套话作全文扫描；研究性质的必要标签不据此删除。
