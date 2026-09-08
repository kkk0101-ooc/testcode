# 如何用 traecli 在一周内完成一篇 SCI 初稿：研究生实操版

如果实验基本结束、数据已经拿到，但论文一直拖着没写完，可以把 traecli 当成一个“论文项目助手”来用。

这里说的“一周完成”，是指已经有主要实验数据、研究问题基本明确，并在一周内完成数据整理、论文结构设计、Results and Discussion、Introduction、Experimental、Abstract、Conclusion，以及最后一轮全文检查，最终得到一版可以交给导师继续修改的 manuscript v1.0。

如果实验还没做完，或者连文章准备回答什么科学问题都没有想清楚，AI 并不能把这一部分凭空补出来。

下面这套流程更适合已经进入论文写作阶段的硕博研究生。

## 中英文版本要求

写论文时，应分别创建中文版本和英文版本文件，不要把中英文内容混写在同一个文件中。两个版本应保持章节结构、图表编号、实验数据和核心结论一致。英文版用于正式投稿，中文版用于梳理逻辑、内部讨论和导师审阅。

推荐的目录结构如下：

```text
SCI_Paper/
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
├── data/
├── figures/
├── references/
├── scripts/
└── notes/
```

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
