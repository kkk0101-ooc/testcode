# S6 post-hoc exploratory 论文结果报告

状态：PASS_POST_HOC_EXPLORATORY

## 核心结论 [post-hoc exploratory]

NNS 识别的 patch 与语义前景正向对齐，遮挡这些 patch 引起的目标 logit 变化大于遮挡空间匹配的对照 patch。NNS 对齐和表示层 probe gain 均表现为深层高于浅层。两种深度对比在逐图层面近乎不相关，因此它们反映互补的深度现象，而不是一条已证实的统一因果机制。

## 论文结果段 [post-hoc exploratory]

在 R12 选择的 240 张图像中，位置对照校正后的 NNS–前景对齐为 0.0626（ordinary BCa 95% CI [0.0574, 0.0679]；235/240 valid）。匹配遮挡效应为 0.0749（ordinary BCa 95% CI [0.0571, 0.0927]；224/240 valid），通过 outcome access 后设定的 S5 探索性门槛。NNS 对齐的 deep-minus-shallow 差异为 0.0363（ordinary BCa 95% CI [0.0316, 0.0410]）。在完整 S3-R7 D2 封存队列中，probe gain 的 deep-minus-shallow 差异为 0.0462（ordinary BCa 95% CI [0.0441, 0.0482]；3424/3482 valid）。D1 与 D2 在 233 个共同有效样本上的 Pearson 相关为 0.0016，未显示逐图耦合关系。

## Endpoint decisions [post-hoc exploratory]

| Endpoint [post-hoc exploratory] | n total | n valid | n NA | n failed | Estimate | Ordinary BCa 95% CI | Adjusted interval | Adjusted p | State |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| P1 | 240 | 235 | 5 | 0 | 0.062600 | [0.057391, 0.067894] | lower=0.051442 | 1.88e-19 | SUPPORTED_POST_HOC_EXPLORATORY |
| P2 | 240 | 224 | 16 | 0 | 0.074940 | [0.057084, 0.092727] | lower=0.034745 | 1.61e-05 | SUPPORTED_POST_HOC_EXPLORATORY |
| D1 | 240 | 235 | 5 | 0 | 0.036314 | [0.031615, 0.040972] | [0.025816, 0.046813] | 4.14e-05 | SUPPORTED_POST_HOC_EXPLORATORY |
| D2 | 3482 | 3424 | 58 | 0 | 0.046152 | [0.044088, 0.048244] | [0.043101, 0.049202] | 2.47e-119 | SUPPORTED_POST_HOC_EXPLORATORY |

P1、P2 和 D1 的 adjusted interval/p 来自 S5 的 family-size-three L1 exploratory 分析。D2 是独立的 two-direction exploratory endpoint，不与前三项合并成新的 confirmatory family。

## P2 原规则与 S5 规则 [post-hoc exploratory]

原 P2 规则为 ORIGINAL_RULE_NOT_PASSED：点估计 0.074940 未超过 0.10，调整后下界 0.034745 未超过 0.10，ordinary BCa half-width 0.017855 未达到 ≤0.015。S5 使用的 ≥0.07 点估计门、>0 adjusted lower-bound 门和 ≤0.02 half-width 门均在 outcome access 后指定。

## 解释范围 [post-hoc exploratory]

结果支持在所分析样本中报告正向的 NNS–前景对齐、匹配遮挡响应和两个正向深度对比。D2 衡量 probe 可解码性，不能替代 NNS 证据；D1 与 D2 的相关性接近零，不支持统一或因果性的深度形成机制。结果不恢复原 S4-R10 STOP_SUCCESS，也不作为 preregistered、outcome-blind、confirmatory 或 strong-FWER-preserved 证据。
