# Revision v7 防御性写作审计

审计对象为 v7 最终中英文 Markdown 主文及其 LaTeX/PDF 镜像。审计按 academic-defensive-writing-auditor 的 D1–D14 分类执行，目标是 claim → evidence → concise boundary，不删除会改变科学解释的边界。

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| Abstract | “Continuous component-replacement correlations are positive, but the original all-family claim is not supported” | D12 candidate | Minor | NECESSARY_CAVEAT | 否定项紧随正结果 | 阴性 omnibus 是主结论的一部分，保留可增加可信度 | Keep | 保留当前句；它同时报告效应方向和冻结判定 |
| Introduction, contribution paragraph | “Path maps do not outperform activation × gradient in H4” | D2 candidate | Minor | NECESSARY_CAVEAT | 形式上是非主张句 | 这是直接阴性结果，不是预防审稿意见 | Keep | 保留当前句，并以 H4 数值支撑 |
| Related Work §2.5 | “Not reported refers only to the versions reviewed” | D8 | Minor | MIXED | 容易读成文献免责条款 | 一次范围界定可接受，重复会显得防御 | Keep once | 只在相关工作表前保留一次，补充材料不重复扩写 |
| §4.3 | “Pointwise equality therefore cannot certify structural completeness” | D13 candidate | Minor | NECESSARY_CAVEAT | 使用绝对否定 | 该结论由平方支路反例严格支持 | Keep | 保留当前形式 |
| §4.4 | “not unique explanations … not generally implementation invariant” | D2 | Minor | NECESSARY_CAVEAT | 连续两个否定边界 | 重参数化反例直接决定解释范围 | Keep | 保留一次；不在摘要重复 |
| §5.3 | “The bootstrap does not generate p values” | D2 candidate | Minor | NECESSARY_CAVEAT | 否定表述 | 直接修复极小 p 值来源混淆 | Keep | 保留，并紧接 exact sign test 来源 |
| §6.6 | “The added weights are not independent training seeds” | D2 candidate | Minor | NECESSARY_CAVEAT | 典型非主张句式 | 改变四 checkpoint 证据的外推含义 | Keep | 保留一次，不称外部复现 |
| §7.1 | “does not establish uniform within-type ranking” | D12 candidate | Minor | NECESSARY_CAVEAT | 对 pooled 结果作限制 | v7 分层结果表明确要求该收缩 | Keep | 保留，并前置分层数值 |
| §7.1 regional task | “no stable or practically useful ranking gain” | D4 candidate | Minor | NECESSARY_CAVEAT | 直接报告弱结果 | 清楚报告区域阴性结果，无借口性解释 | Keep | 保留；Taylor 信息预算差异只陈述一次 |
| §8.2 | 方法局限集中列出一阶、参数化、模型、数据和目标边界 | D3/D8 candidate | Minor | NECESSARY_CAVEAT | 多项限制集中出现 | 放在单独局限节比散落全文更清晰 | Keep in one section | 保留集中段，正文其余位置只保留解释所需边界 |

### Paper-level defensive-writing score

**2/10，clean。** 当前主文没有 reviewer-facing prebuttal、致歉式解释、缺实验辩护、公平性自我声明或宣传性补偿。否定句主要承载冻结阴性结果和定义边界。

### Highest-priority removals

已在本轮完成的高优先级清理：

1. 将主文 confirmatory 改为“结果访问前本地冻结的主分析”或“预先规定的稳健性分析”。
2. 将“复现”改为“在独立图像队列和额外公开 checkpoint 上保持同方向”。
3. 将 11/12 从摘要第一结果降为历史 provenance，正文优先报告连续分层结果。
4. 删除“未见”等容易构成全领域否定的单元格式表述，主文只保留五个最近邻对象级比较。
5. 将方法局限集中到 §8.2，减少相同免责声明在贡献、结果和结论中的重复。

### Necessary caveats to preserve

- 基础 Endpoint 等于同目标 Gradient × Input，不是新的基础归因公式。
- 结构合法切面与点态微分闭合不同，单点闭合不能发现零导数结构遗漏。
- H4 四 checkpoint 为负，区域 local map 无稳定实用增益。
- 原 all-family omnibus 未获支持；v7 分层分析是 post-hoc diagnostic。
- 结果条件于固定 checkpoint、target、reference、replacement contract 和 ImageNet patch-16 ViT。
- 分项依赖计算图、内部坐标和切面，不具一般实现不变性。

### Repeated phrase audit

英文主文精确扫描：we emphasize=0、to avoid misunderstanding=0、for a fair comparison=0、we believe=0、taken together=0、promising/encouraging=0、confirmatory=0。not supported 只用于冻结阴性结论；post-hoc 只用于分析身份。中文主文无“需要强调”“值得注意”“避免误解”“综上所述”或“原确认性判断”等审稿回复式结构。

结论：PASS_DEFENSIVE_WRITING_AUDIT。
