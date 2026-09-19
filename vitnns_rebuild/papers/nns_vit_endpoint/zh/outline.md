# NNS-ViT Endpoint Scanner：论文大纲（R3）

日期：2026-09-20。本文档将原确认性实验与阶段 M（M1–M6）组织为一篇独立论文，
不覆盖 path-integrated NNS-ViT 主稿。正文图号绑定已封存的
`ENDPOINT-ALL-FIGURES-R2` Figure 1–40；M4 Review-R2、M6 和 signed/all-head 页面从同一
图包的 `supplement/` 引用。投稿排版可重新编号，但不得改变图像内容、数字或结论。

论文入口：

- 中文主稿：`papers/nns_vit_endpoint/zh/full_draft.md`；
- 英文主稿：`papers/nns_vit_endpoint/en/full_draft.md`；
- 证据矩阵：`papers/nns_vit_endpoint/notes/endpoint_scanner_paper_evidence_matrix_r1.md`；
- 论文级图清单：`papers/nns_vit_endpoint/figure_manifest.json`；
- 图包：`reports/nns_vit_endpoint_all_figures_r2_20260919/`。

## 1. 标题与中心论点

英文题目：

> **NNS-ViT: Auditable Endpoint Maps and Structured Jacobian Decomposition for Internal Targets in Vision Transformers**

中文题目：

> **NNS-ViT：视觉 Transformer 内部目标的可审计 Endpoint 图与结构化 Jacobian 分解**

中心论点：

> **Endpoint is an auditable internal-target representation, not a universal intervention estimator.**

对应中文：

> **Endpoint 的主要价值是统一、可审计地表示 ViT 内部目标的局部输入依赖，而不是成为所有有限干预任务中的通用最优预测器。**

基础对象是

$$
E_s(x)=x\odot\nabla_xs(x).
$$

该对象在相同 scalar target 下就是 normalized-input 坐标中的 Gradient × Input。论文贡献
不在重新命名基础公式，而在原始 NNS 忠实性桥接、异构目标 registry、结构切面、独立数值
审计和 ViT 机制分析。

## 2. 五个研究问题

1. **RQ1：原始 NNS 忠实性。** 原始 NNS learning-image propagation 是否可形式化为局部
   Jacobian propagation，并与 Endpoint 数值等价？
2. **RQ2：统一内部目标扫描。** 同一 registry 是否能够完整、可审计地扫描异构 ViT
   scalar targets？
3. **RQ3：结构与数值可审计性。** Q/K/V、routing/value 和 residual identity/branch cuts
   是否闭合，并通过 JVP/VJP、有限差分和负向控制？
4. **RQ4：ViT 内部机制。** Endpoint atlas 揭示了哪些层深、token、attention 和 residual
   路径规律？
5. **RQ5：行为相关性与解释边界。** Endpoint 与 finite replacement 的关系是什么，
   reference-specific 任务何时需要 displacement-matched estimator？

## 3. 四项贡献

1. 把原始 NNS 输入值初始化与当前状态传播形式化为局部 Jacobian propagation，并以五类
   兼容网络的 5,000 个 numerical comparisons 和 11 类 ViT 算子测试建立忠实性桥接。
2. 构建统一 registered-scalar interface，覆盖 token-channel、head、Q/K/V、routing/value、
   residual branch、聚合 target 与类别输出。
3. 定义完整 attention/residual differential cuts，以 76,800 个正向闭合、28,800 个负控和
   6,144 个 analytic JVP checks 验证。
4. 用 M3/M4 atlas 研究空间扩展、routing/value 和 residual 机制，并用 M1/M2/M5 对有限
   干预相关性与 reference 边界进行独立标定。

## 4. 正文结构

### 1 Introduction

- 解释对象的异构性与统一输入坐标需求；
- Identity、Correctness、Utility 三层证据必须分开；
- 五个研究问题与四项贡献；
- Figure 1：真实懒熊输入到 ground-truth class-logit Endpoint 的方法概览。

### 2 Related Work

- 原始 NNS、内部单元可视化与 Gradient × Input；
- Integrated Gradients、conductance 与 path attribution；
- attention rollout、Chefer/AttnLRP、LibraGrad 与 Transformer 结构分析；
- activation patching、activation-space Taylor 与有限干预评价；
- Figure 2：Original NNS-CNN、ViT Endpoint 与 Path IG 的 estimand 区分。

### 3 Method

- 以 $\mathcal L(x;x)=\operatorname{diag}(x)$ 和链式法则定义 Endpoint；
- 说明 normalized-input origin、bias/CLS/position embedding 的非归因范围；
- 定义 registered scalar targets、registry-complete scan 和分层 token-channel audit；
- 定义 Q/K/V、routing/value、head 和两处 residual cuts；
- 区分 local differential correctness 与 finite replacement utility；
- Figures 5、10、11、13、22、34–40 支持方法、registry 和真实 map 阅读。

### 4 Experimental Design

- ViT-B/16 与 DeiT-S/16，标准 timm 0.9.16 unfused eval backend；
- 共享 400 图 evaluation cohort；M3 使用 deterministic-hash 100 图子集；
- 图像是主要统计单位，10,000 次 image bootstrap、符号检验和 Holm 校正；
- display-normalized maps 只用于空间形态，数值来自 raw signed arrays。

### 5 Results

#### 5.1 RQ2：registry-complete scan

- 800/800 image-model tasks；336,800 target evaluations；
- wall time 1,752.71 s；四卡上界 1.9475 GPU-hours；retained data 9.290 GB；
- Figures 12、14。

#### 5.2–5.3 RQ3：结构闭合与局部微分正确性

- 76,800 positive closures、28,800 negative controls、6,144 analytic JVP checks；
- Figures 16–18。

#### 5.4–5.6 RQ5：行为标定与边界

- C4a 11/12 family 满足全部条件，all-family omnibus 保持 `NOT_SUPPORTED`；
- C4b 0/4 family 达到联合 +0.05 门，regional Taylor 在四个 family 中均更高；
- M1 strong-baseline 比较单独报告，不改签原 C4a；
- Figures 20–27、36。

#### 5.7 RQ1/RQ4：NNS 忠实性与 token-channel atlas

- 五类网络 5,000 个等价比较全部通过；
- 每 model-image 6,144 个 token-channel targets，共 1,228,800 项；
- entropy 增加、own-token locality 下降、normalized distance 增加；
- CLS–patch、cancellation 和 float64 recovery 按封存结果报告；
- Figures 29–33、37。

#### 5.8 RQ4：attention/residual 机制

- late attention 的 routing-relative path scale 和 Value-only discrepancy 增加；
- mean-query saturation 与 routing/error 为负相关，CLS-query sensitivity 更弱；
- attention residual 的 spatial change 增加，MLP residual 的 spatial change 下降；
- MLP path cancellation 在两模型中呈相反显著趋势；
- M4 Review-R2 明确分母、partial rank、spatial TV 与 M3–M4 trajectory bridge。

### 6 Discussion

- 6.1 Identity、Correctness、Utility 证据分层（Figure 28）；
- 6.2 component ranking 与 spatial ranking 的任务差异；
- 6.3 C4a 边界 family：0.099565，CI 跨 0.10；
- 6.4 评价指标应匹配使用目的；
- 6.5 spatial broadening 与 signed cancellation；
- 6.6 目标条件化 routing 与子层特异 residual 传递；
- 6.7 M5：Endpoint 坐标原点、reference invariance 与 Taylor displacement matching；
- 6.8 M6：选择封存后才读图的自动机制案例 atlas。

### 7 Limitations

- 两个 ImageNet 分类 ViT，不推广到所有 Transformer；
- M3 是分层抽样，不是全神经元穷举；
- Endpoint 依赖输入坐标原点，不满足一般 baseline-difference completeness；
- M4 path shares 是目标条件化 relative norm scale，不是函数值或因果贡献百分比；
- M2 空间干预限定 block、head、region pool、budget、reference 和 outcome；
- ViT-B Path-IG 校准未达到 61/64，不发布跨模型 Path 结论；
- 显示缩放图只支持空间形态阅读。

### 8 Conclusion

按以下顺序收束：原始 NNS 忠实性 → 统一扫描 → 结构可审计性 → ViT 机制 → 行为边界。
最后一句固定为 Endpoint 是 auditable internal-target representation，而不是 universal
intervention estimator。

## 5. 图表安排

论文级 manifest 的 16 张主文推荐图为 Figure 1、2、5、10、11、14、16、18、20、26、28、
29、30、31、32、33，依次覆盖方法、registry、结构审计、M3 机制和行为边界。当前 Markdown
额外内嵌真实 map 阅读图和 M4 Review-R2、M6 supplement，投稿时可按
版面移动到补充材料。统一图包还保留：

- Figure 1–40 的 PNG/PDF/SVG；
- Figure 12、13、17、24、34–40 的 signed companions；
- 两模型各 12 个 block 的 all-head pages；
- M2、M3、M4、M4 Review-R2 和 M6 supplemental figures。

## 6. 写作状态与作者待办

主稿、图清单和机器验证已经按阶段 M 与统一 R2 图包同步。投稿前作者仍需：

1. 填写作者、单位、通讯作者、基金与利益冲突；
2. 确定目标期刊/会议及主文图数；
3. 完成 BibTeX、DOI、页码、venue 和 license 审计；
4. 提供匿名仓库链接、归档 DOI 与数据访问声明。
