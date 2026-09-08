# 阶段 G 论文写作指导：vitnns_rebuild SCI 初稿工作流

> 当前命名：阶段 E 保留为已封存的消融与敏感性分析；阶段 F 专指 F0–F3 论文图表生成；阶段 G 专指论文写作。
> 上位科学范围：[论文1当前实现审计、缺口与后续计划](../../论文1实现审计与后续计划.md)。
> 执行门禁：[阶段 G 论文写作规划](阶段G规划.md)。
> 图表输入：[阶段 F 规划](阶段F规划.md)与 [F3 工作包和产物索引](阶段F_F3工作包与产物索引.md)。

## 本项目不可覆盖的写作约束

这份指导由通用 traecli 写作流程改造成 `vitnns_rebuild` 的项目规范。它只指导如何组织、起草、审查和封存论文，不授权新增实验、重算统计、挑选更好看的案例或修改阶段 F 图表。

1. C10 与 Stage D 必须分 cohort 报告：`n=15` 与 `n=500` 不得池化。
2. C10 历史 `NO_GO_NOVELTY`、`FAIL_SPATIAL`、`BLOCKED` 和 `HOLM_BLOCKED_UNDERSPECIFIED` 必须原样披露。
3. Stage D 的结论只能写成冻结模型、目标、primary `(0,9)`、matched baseline 和 fine-4×4 endpoint 下的 `GO_FIDELITY_SCOPE`。
4. 历史消融阶段 E 是 post-D10 secondary/descriptive analysis；其 E2 owner budget waiver、E3 的两个 typed failures 和 E4 `MIXED` scientific validity 不得隐藏。
5. F3-HeadAtlas 与 F3-ClassCompare 是阶段 F 的补充图形/比较工作包，不得写成另两个顶层阶段；不同 method target 和独立面板缩放必须在图注中披露。
6. 不得把 fixed-clean-attention Value-path 写成完整 Q/K/V、完整 attention 或普遍 class explanation；fine-4×4-cell 也不得写成已证明的原生单像素 attribution。
7. 图像更清晰只改善展示，不改变 fidelity、统计显著性或类别判别能力。
8. 所有数字、图号、图注和主张都必须能反向定位到冻结 source record；缺证据时使用 `[EVIDENCE REQUIRED]`，缺文献时使用 `[REFERENCE REQUIRED]`。
9. Dou et al. 2023 的 *Understanding neural network through neuron level visualization* 是 NNS 在 FCN/CNN 上的核心来源文献；本文只能把它作为问题来源和方法背景，不能把其 FCN/CNN 结果写成 ViT 证据。

阶段 G 的推荐顺序不是机械的七天日历，而是 `G0` 初始化与证据绑定、`G1` scientific story/evidence matrix/figure cards、`G2` Results and Discussion、`G3` Methods and Experimental、`G4` Introduction and references、`G5` Conclusion and Abstract、`G6` 双语/审稿/QC、`G7` 编译和封存。后文的 Day 1–7 对应这些内部里程碑。

如果实验基本结束、数据已经拿到，但论文一直拖着没写完，可以把 traecli 当成一个“论文项目助手”来用。

这里说的“一周完成”，是指已经有主要实验数据、研究问题基本明确，并在一周内完成数据整理、论文结构设计、Results and Discussion、Introduction、Experimental、Abstract、Conclusion，以及最后一轮全文检查，最终得到一版可以交给导师继续修改的 manuscript v1.0。

如果实验还没做完，或者连文章准备回答什么科学问题都没有想清楚，AI 并不能把这一部分凭空补出来。

下面这套流程更适合已经进入论文写作阶段的硕博研究生。

## 中英文版本要求

写论文时，应分别创建中文版本和英文版本文件，不要把中英文内容混写在同一个文件中。两个版本应保持章节结构、图表编号、实验数据和核心结论一致。英文版用于正式投稿，中文版用于梳理逻辑、内部讨论和导师审阅。

本项目推荐的目录结构如下：

```text
vitnns_rebuild/
├── AGENTS.md
├── manuscript/
│   ├── zh/
│   │   ├── outline.md
│   │   ├── abstract.md
│   │   ├── introduction.md
│   │   ├── experimental.md
│   │   ├── results_discussion.md
│   │   └── conclusion.md
│   └── en/
│       ├── outline.md
│       ├── abstract.md
│       ├── introduction.md
│       ├── experimental.md
│       ├── results_discussion.md
│       └── conclusion.md
│   ├── notes/
│   ├── references/
│   ├── tables/
│   ├── figure_manifest.json
│   └── qc/
├── docs/
│   ├── 阶段F规划.md
│   ├── 阶段F_F3工作包与产物索引.md
│   ├── 阶段G规划.md
│   └── 阶段G论文写作指导.md
├── evidence/stage_g_writing/
└── reports/stage_g_writing_*/
```

论文图片不复制进新的临时 `figures/` 目录。`manuscript/figure_manifest.json` 应通过路径、SHA-256 和 Stage F source record 引用冻结图包。

核心文献材料位于 `../相关文章/Understanding neural network through neuron level visualization/`，其中 `README.md` 给出 DOI、期刊、页码和中英文拆分文件。阶段 G 应把该目录登记为 `primary_reference_anchor`。后续加入 `../相关文章/` 的其他文献可以进入 related work、背景或对照方法矩阵，但默认不改变这篇 NNS 源文献的核心地位。

## 1. 不要把 traecli 当成“高级聊天框”

很多人用 AI 写论文的方式是：

> 发一段数据 → 让 AI 写一段结果 → 再发另一段数据 → 再写一段 → 最后把所有内容拼起来。

这样的问题很明显：前后上下文容易断掉。写到后面经常出现以下情况：

- 前后术语不一致；
- 同一个结果解释两遍；
- 不同章节之间机制矛盾；
- 数字改了，但正文没有同步；
- Introduction 和 Results 根本不是同一条逻辑。

更合适的方法是直接把整篇 SCI 当成一个项目管理。这样 traecli 每次处理的是同一个完整项目，而不是一段段孤立的聊天记录。

## 2. 第一步先写 AGENTS.md

正式写论文之前，建议先为 traecli 建立一个长期规则文件。

例如：

```markdown
# Project

This repository contains an SCI manuscript.

# Language versions

- Maintain separate Chinese and English manuscript files.
- Store Chinese files under `manuscript/zh/`.
- Store English files under `manuscript/en/`.
- Keep section structure, figure and table numbering, numerical values, and scientific conclusions consistent between the two versions.
- Do not mix Chinese and English manuscript content in the same file.

# Writing requirements

- Use formal academic English in the English version.
- Use accurate and concise academic Chinese in the Chinese version.
- Prefer concise sentences.
- Avoid exaggerated claims.
- Do not invent data.
- Do not invent references.
- Do not invent experimental parameters.
- Preserve numerical values.
- Distinguish experimental observation from interpretation.
- If evidence is insufficient, mark `[EVIDENCE REQUIRED]`.
- If a reference is needed, mark `[REFERENCE REQUIRED]`.

# Writing logic

Each Results and Discussion paragraph should preferably contain:

1. Observation
2. Comparison
3. Quantitative evidence
4. Interpretation
5. Supporting evidence
6. Scientific implication

# Terminology

Use terminology consistently throughout the manuscript.
```

以后不用每次重新告诉 traecli：

- “不要乱编数据。”
- “不要写得太夸张。”
- “句子不要太长。”
- “不要生成假文献。”

这些都可以作为整个论文项目的长期约束。

## 3. Day 1：先搭论文逻辑，不写正文

第一天最重要的任务不是写英文，而是回答三个问题：

1. 这篇文章到底研究了什么？
2. 最重要的发现是什么？
3. 所有实验结果之间是什么关系？

可以让 traecli 先读取现有的数据、图、实验记录、组会笔记和已经写过的零散文字，然后生成：

```text
notes/scientific_story.md
```

提示词可以这样写：

```text
Read all available project files.
Do not write the manuscript yet.
First reconstruct the scientific story.
Create notes/scientific_story.md containing:
1. Central research question
2. Main findings
3. Evidence supporting each finding
4. Proposed explanation
5. Logical relationships between different results
6. Potential contradictions
7. Missing evidence
8. Recommended Results and Discussion structure
Do not invent information.
```

这一轮的目的不是得到英文，而是检查这篇论文到底有没有形成完整的证据链。

## 4. 做一张 Evidence Matrix

这是非常推荐的一步。把论文中的主要结论全部列出来，然后逐一确认每个结论到底有什么证据。

例如：

| Scientific claim | Direct evidence | Supporting evidence | Confidence |
| --- | --- | --- | --- |
| Claim A | Experiment 1 | Experiment 2 | Strong |
| Claim B | Experiment 3 | Literature | Moderate |
| Claim C | None | Indirect evidence | Weak |

让 traecli 自动检查：

```text
Create an evidence matrix for all major scientific claims.
For each claim identify:
- direct evidence
- indirect evidence
- literature support
- alternative explanation
- confidence level
Classify confidence as:
- Strong
- Moderate
- Weak
- Unsupported
```

这一轮会筛掉很多典型的“AI 论文句子”：看起来非常专业，但其实没有实验支撑。

## 5. Day 2：一次整理好数据、计算和图片

### 本项目覆盖规则

本节下面的通用数据处理建议不构成 `vitnns_rebuild` Stage G 的执行权限。该项目的计算、统计和图片在 Stage A–F 完成；Stage G 不读取 raw JPEG/模型、不重新运行数据处理、统计、attribution、fidelity 或绘图。Day 2 在本项目中改为：核验 Stage F figure inventory、建立 Figure Cards，并把已经封存的数字绑定到 manuscript source table。发现缺图或 source mismatch 时回到 Stage F 的新 revision 处理，不能在写作目录内临时修图。

论文最浪费时间的地方之一，就是写到一半不断回去找 Excel、重新计算、找原图、修改百分比，或者确认某个数据到底是多少。

所以第二天建议专门处理数据。能用 Python 自动化的，尽量自动化。

例如：

```text
data/
├── raw/
└── processed/
scripts/
figures/
```

要求 traecli：

```text
Read the raw data.
Write reproducible Python scripts for:
- data processing
- statistical analysis
- curve fitting
- plotting
- numerical comparison
Do not modify raw data.
Save calculated values separately.
Export publication-quality figures.
```

以后如果数据发生变化，只需重新运行脚本，不要手动修改十几个数字。这样可以避免最常见的问题：图里的数字改了，但正文和摘要还是旧版本。

## 6. 为每张 Figure 建立一个 Figure Card

建议为每一张论文图单独建立一页笔记。

例如：

```text
notes/Figure_1.md
notes/Figure_2.md
notes/Figure_3.md
```

统一写成：

```markdown
# Figure

## What is shown

## Main observations

## Quantitative results

## Comparison

## Possible interpretation

## Supporting evidence

## Alternative explanation

## Claims that cannot currently be made
```

这样做之后，traecli 在写 Results 时不需要直接面对几十个散乱文件。它读取 Figure Card 就能知道这张图展示了什么、可以得出什么结论，以及不能得出什么结论。

## 7. Day 3–4：先写 Results and Discussion

实验型 SCI 最适合先写 Results and Discussion。因为写完这一部分以后，你才真正知道文章最后讲了什么。不要先写 Abstract，也不建议第一天就写 Introduction。

中文版本和英文版本应分别写入以下文件：

```text
manuscript/zh/results_discussion.md
manuscript/en/results_discussion.md
```

可以先用中文版本确认论证逻辑，再生成对应英文版本；生成英文版本后，需要复核两者的数据、图表编号和结论是否一致。

## 8. 不要让 traecli 一次写完整的 Results

比如这种提示：

```text
帮我写 3000 字 Results and Discussion。
```

一般效果不会很好，常见结果是：

- 很长、很顺，但信息密度低；
- 大量重复实验现象；
- 机制解释非常泛；
- 每一段都长得差不多。

更好的方法是一段一段生成。

例如：

```text
Read:
- notes/Figure_2.md
- notes/scientific_story.md
- AGENTS.md

Draft only the first paragraph of Section 3.1.

Purpose:
Describe the main experimental observations.

Structure:
1. observation
2. comparison
3. quantitative evidence
4. preliminary interpretation

Do not discuss mechanisms that are not supported yet.

Length:
150-180 words.
```

下一段再单独写：

```text
Draft the next paragraph.

Purpose:
Explain the possible mechanism.

Use only evidence already available in this project.
Clearly distinguish observation from interpretation.
```

这样生成的内容会更可控。

## 9. Results 最好统一使用一种逻辑

可以要求 traecli 的每一段尽量遵循以下结构：

```text
Observation
↓
Comparison
↓
Quantitative evidence
↓
Interpretation
↓
Supporting evidence
```

例如，不要只写：

> Sample A performs better than Sample B.

而应该继续回答：

- 好多少？
- 在什么条件下更好？
- 哪个实验能证明？
- 原因是什么？
- 有没有其他可能的解释？
- 这个结果对全文意味着什么？

这才是 Discussion。

## 10. 写完 Results 后，先让 traecli 当 Reviewer

写完 Results 后不要马上润色，可以先用下面的提示词进行审查：

```text
Act as a critical reviewer.
Read the entire Results and Discussion.
Do not rewrite it.
Identify:
1. unsupported claims
2. causal claims without sufficient evidence
3. contradictions between sections
4. repeated interpretations
5. results described without discussion
6. mechanisms proposed without evidence
7. missing quantitative comparisons
8. statements requiring references
Save the report as:
notes/results_review.md
```

很多时候，AI 最有价值的用途不是替你写，而是帮你找漏洞。

## 11. Day 4：写 Experimental

### 本项目覆盖规则

这里的 “Materials” 模板是通用示例。`vitnns_rebuild` 应写成 Dataset and cohort、Model and preprocessing、Attribution methods、Intervention-based fidelity evaluation、Statistical analysis、Resource accounting 和 Reproducibility/provenance。所有参数从冻结 config、manifest 和 completion 提取；不知道的字段保留 `[PARAMETER REQUIRED]`，不得按常见 ViT 设置补全。

Experimental 相对简单，因为主要是事实。可以先把内容整理成：

```text
Materials
Sample preparation
Experimental procedure
Characterization
Property measurements
Data analysis
Statistical analysis
```

然后让 traecli 检查 reproducibility：

```text
Review the Experimental section.
Check whether the following information is missing:
- material specifications
- suppliers
- preparation parameters
- temperature
- pressure
- time
- instrument model
- measurement conditions
- calibration method
- number of repeated measurements
- statistical method
Do not invent missing parameters.
Mark missing information as:
[PARAMETER REQUIRED]
```

这里要特别注意：AI 绝对不能帮你“合理补全”实验参数。不知道就是不知道。

## 12. Day 4：统一全文术语

写到这里以后，很容易出现同一个东西有三四种叫法。所以建议让 traecli 建立：

```text
notes/terminology.md
```

其中记录：

```text
Preferred term
Abbreviation
Symbol
Unit
Alternative wording to avoid
```

然后扫描中英文全文：

```text
Scan both the Chinese and English manuscript versions for consistency.
Check:
- terminology
- abbreviations
- units
- sample names
- symbols
- figure numbering
- table numbering
- significant figures
- consistency between Chinese and English versions
Do not change scientific meaning.
```

这一轮非常节省人工时间。

## 13. Day 5：再写 Introduction

Introduction 放到 Results 后面写，通常更快。因为到这一步，你已经知道：

- 文章真正解决了什么问题；
- 最有价值的结果是什么；
- 哪个 mechanism 最值得突出。

一个常规的 Introduction 可以控制在四个逻辑层次：

```text
Research background
Existing approaches
Unresolved problem
What this study does
```

不要一开始就写成一篇大综述。

## 14. 不要让 AI 凭记忆生成参考文献

这是最容易翻车的地方之一。不要直接说：

```text
给我找 20 篇相关文献并生成引用。
```

然后直接复制到论文中。正确方式是先通过正规的数据库和检索工具找到真实文献，然后把 PDF、DOI、EndNote/BibTeX 和文献笔记统一放进：

```text
references/
```

本项目的第一条文献记录必须来自 `../相关文章/Understanding neural network through neuron level visualization/README.md`，并在 `manuscript/references/literature_notes.md` 中固定为：

```text
primary_reference_anchor:
  title: Understanding neural network through neuron level visualization
  authors: Hui Dou; Furao Shen; Jian Zhao; Xinyu Mu
  venue: Neural Networks 168 (2023), 484-495
  doi: 10.1016/j.neunet.2023.09.030
  role: NNS source method for FCN/CNN; this manuscript extends and revalidates the idea for ViT
```

任何写作段落引用这篇文章时，应使用直接事实：它将 NNS 应用于 FCN/CNN 并提出 learning image / neuron-level visualization。不要写成“已有工作已经证明 NNS 在 ViT 上有效”。

再让 traecli 整理：

```text
Read all literature files.
Create literature_notes.md.
For every paper extract:
- research question
- method
- main findings
- proposed mechanism
- limitation
- relevance to this manuscript
Do not generate references that are not present in the provided files.
```

## 15. 做一个 Claim–Reference Matrix

Introduction 和 Discussion 写完以后，可以建立：

| Claim | Reference needed? | Available source |
| --- | --- | --- |
| Statement 1 | Yes | Ref 3 |
| Statement 2 | Yes | Ref 6 |
| Current experiment | No | Statement 3 |

然后让 traecli 检查：

```text
Scan Introduction and Discussion.
For every literature-dependent statement, classify it as:
[CITED]
[REFERENCE REQUIRED]
[REFERENCE OPTIONAL]
```

这样比等到投稿前再手工找漏引高效得多。

## 16. Day 5：Introduction 也先做 Outline

不要直接使用：

```text
帮我写 Introduction。
```

先使用：

```text
Read:
- notes/scientific_story.md
- manuscript/en/results_discussion.md
- references/literature_notes.md

Create an Introduction outline.
Do not draft prose.
For each paragraph specify:
- purpose
- main claim
- supporting literature
- transition to the next paragraph
```

确认逻辑以后，再逐段完成：

```text
Draft Paragraph 1 only.
```

## 17. Day 6：写 Conclusion

Conclusion 不需要重新复述整篇论文，一般回答以下问题：

```text
What was done
What was found
How it can be explained
Why it matters
```

控制在比较紧凑的长度，不要在 Conclusion 里突然提出正文没有讨论过的新机制。

## 18. 最后写 Abstract

Abstract 最后写效率最高，可以采用以下结构：

```text
Background
Problem
Method
Key results
Mechanism
Significance
```

最值得注意的是尽量放定量结果。少写：

```text
significantly improved
greatly enhanced
excellent performance
```

多写：

```text
increased from X to Y
decreased by X%
was X times higher than
```

在 SCI 摘要里，数字通常比形容词更有说服力。

## 19. 做一次 Abstract–Results–Conclusion 一致性检查

提示词如下：

```text
Compare:
- Abstract
- Results and Discussion
- Conclusion

Check:
1. whether every numerical value in the Abstract appears in Results
2. whether every conclusion is supported
3. whether every mechanism mentioned in the Abstract is discussed
4. whether terminology is consistent
5. whether any claim is exaggerated
6. whether the Chinese and English versions express the same data and conclusions
Do not rewrite.
```

这一步非常适合由 traecli 批量检查。

## 20. Day 6：英文润色不要只说“make it academic”

很多 AI 味特别重的论文，都是从这个提示开始的：

```text
Make this more academic.
```

然后全文出现：

```text
remarkably
notably
intriguingly
pivotal
profound
groundbreaking
```

材料、化学、工程类论文一般不需要这么多修饰。更好的提示是：

```text
Edit this paragraph for an SCI journal.
Requirements:
- shorten unnecessarily long sentences
- remove redundant expressions
- use precise technical language
- preserve all numerical values
- do not add new claims
- do not exaggerate significance
- avoid promotional wording
Do not change scientific meaning.
```

这样生成的语言会自然很多。

## 21. Day 7：让 traecli 模拟 Reviewer

最后一天建议让 traecli 分角色审稿。

Reviewer 1：

```text
Focus on novelty and significance.
```

Reviewer 2：

```text
Focus on scientific logic and evidence.
```

Reviewer 3：

```text
Focus on experimental reproducibility and statistics.
```

统一要求：

```text
Provide:
- Major comments
- Minor comments
- Potential rejection reasons
```

然后逐条处理。

## 22. 再做一次“最危险结论”检查

这个提示非常实用：

```text
Identify the 20 strongest scientific claims in the manuscript.
For each claim, classify the evidence as:
- Directly demonstrated
- Strongly supported
- Reasonably inferred
- Speculative
- Unsupported
For speculative or unsupported claims, suggest a more cautious wording.
```

很多论文的问题不是实验差，而是结论比证据走得太远。

## 23. 最后运行一次全文 QC

可以让 traecli 最后统一检查：

```text
Perform final manuscript quality control.
Do not rewrite the manuscript.
Check:
- figure citation order
- table citation order
- reference consistency
- abbreviation definitions
- units
- symbols
- numerical consistency
- significant figures
- grammar
- repeated sentences
- repeated conclusions
- section transitions
- unsupported claims
- missing references
- missing experimental parameters
- consistency between Chinese and English manuscript files
Output:
final_QC_report.md
```

最后再由作者根据 QC report 逐项修改，并对照检查中英文两个版本文件，确保科学内容完全一致。

## 24. 防御性写作审计

阶段 G 必须使用 Academic Defensive Writing Auditor 的原则审查中文和英文稿件。目标不是删除必要的不确定性或扩大结论，而是避免论文读起来像提前写好的 rebuttal。

优先使用下面的论证顺序：

~~~text
claim → evidence → concise boundary
~~~

避免下面的防御顺序：

~~~text
anticipated objection → disclaimer → justification → weakened claim → another disclaimer
~~~

### 24.1 需要检查的模式

- reviewer-facing prebuttal，例如 “We emphasize that” 或 “It should be noted that”；
- 重复的 non-claim disclaimer，例如多次写 “We do not claim that”；
- 同一句堆叠 although、while、despite、only、may、might 等限定；
- 没有诊断证据却为较弱结果寻找理由；
- 用长段文字辩解为什么没有做某项实验；
- 用 “For a fair comparison” 自我证明公平，而不是直接陈述控制变量；
- 重复使用 preliminary、exploratory、within-scope 等范围标签；
- 法律条款式列出论文不主张的一切；
- 与有效性、可复现性或解释无关的过程性失败披露；
- 用 strong、promising、robust、compelling 等形容词补偿证据；
- “Taken together, these results demonstrate” 等自动总结句；
- 把标准操作重新命名并以此防御 novelty。

### 24.2 necessary caveat 与 defensive wording 的判定

逐句回答五个问题：

1. 删除后是否会实质性误导证据解释？
2. 该句是否报告了有依据的条件、不确定性或限制？
3. 该句的主要功能是否只是回答想象中的审稿质疑？
4. 能否改写为直接说明已评估、已观察或范围之外的事实？
5. 同一 caveat 是否已在其他位置出现？

每句分类为 `NECESSARY_CAVEAT`、`DEFENSIVE`、`MIXED` 或 `CLEAN`。修改规则是：事实先行；范围用正面句式表达；每项 claim 最多保留一个会改变解释的 caveat；没有技术信息的防御句直接删除。

例如：

- 不写：“For a fair comparison, all methods use the same backbone.”
- 写：“All methods use the same frozen ViT-B/16 backbone.”
- 不在每节重复：“We do not claim spatial robustness.”
- 在 Limitations 集中写一次：“Translation and center-crop stability did not meet the C10 criterion.”
- 不写：“Although the improvement is modest, the result remains encouraging.”
- 写实际效应量和区间；没有证据支持的原因不补写。

### 24.3 本项目必须保留的 caveat

以下边界直接影响科学解释，不能以“去防御化”为由删除，但应各自在最相关位置简洁出现一次：

- C10 的 `NO_GO_NOVELTY`、`FAIL_SPATIAL` 和 `BLOCKED`；
- C10 `n=15` 与 Stage D `n=500` 分 cohort、不池化；
- Stage D 只支持 `GO_FIDELITY_SCOPE`；
- 历史消融阶段 E 是 post-D10 descriptive analysis，且 E4 scientific validity 为 `MIXED`；
- F3-ClassCompare 各方法的 target/class-agnostic 差异；
- panel-independent scaling 不能用于跨方法幅值比较；
- display improvement 不改变 fidelity 或类别判别性。

这些 caveat 应优先写成测量结果或正面范围句，而不是 “we emphasize / we do not claim / to avoid misunderstanding” 式声明。

### 24.4 固定审计输出

在 `manuscript/qc/defensive_writing_audit.md` 输出：

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|

随后给出：

1. 0–10 的 paper-level defensive-writing score；
2. 3–10 项 highest-priority removals；
3. necessary caveats to preserve；
4. repeated phrase audit。

G6 只有在中英文稿分别完成该审计、必要 caveat 未丢失、defensive wording 已处理后才能 PASS。
