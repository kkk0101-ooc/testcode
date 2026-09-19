# NNS-ViT Endpoint Scanner：论文大纲（R1）

日期：2026-09-18。本文档把已完成的 ENDPOINT-E2–E6 证据组织成一篇独立论文，
不覆盖此前 path-integrated NNS-ViT 主稿。论文图号暂时沿用封存图包的 Figure 1–28、
M3 补充 Figure 29–33 和独立细粒度展示包 Figure 34–40；
投稿排版时可在不改图像内容、数值和结论的前提下重新编号。

论文初稿：

- 中文：`papers/nns_vit_endpoint/zh/full_draft.md`；
- 英文：`papers/nns_vit_endpoint/en/full_draft.md`；
- 证据矩阵：`papers/nns_vit_endpoint/notes/endpoint_scanner_paper_evidence_matrix_r1.md`；
- 防御性写作审计：`papers/nns_vit_endpoint/qc/endpoint_scanner_full_draft_r1_defensive_writing_audit.md`；
- 图包：`reports/nns_vit_endpoint_unified_paper_figures_r2_20260918/`。
- 细粒度蓝底展示图：`reports/nns_vit_endpoint_fine_grained_blue_figures_r1_20260919/`。

## 1. 题目与一句话主线

英文题目：

> **NNS-ViT Endpoint Scanner: Registered Internal-Target Scanning and Structured Decomposition of Vision Transformers**

中文题目：

> **NNS-ViT Endpoint Scanner：视觉 Transformer 的注册内部目标扫描与结构化分解**

一句话主线：

> Endpoint Scanner 不提出新的基础归因公式，而是把原始 NNS 的输入值 seed 和终点
> Jacobian 传播原则系统适配到 ViT，并用目标注册、完整结构切分和有限干预分别回答
> “能否扫描”“分解是否正确”“成分大小是否预测干预”“空间细节是否带来额外价值”。

## 2. 投稿定位与贡献边界

投稿定位固定为：

```text
REPRODUCTION_UNIFIED_IMPLEMENTATION_AND_SYSTEMATIC_EMPIRICAL_AUDIT
```

基础定义为

$$
L_s^{\mathrm{Endpoint}}(x)=x\odot\nabla_x s(x).
$$

因此不能把基础量写成不同于 Gradient × Input 的新算法。论文贡献写成：

1. 把原始 NNS 的终点传播原则形式化为真实 ViT Jacobian，并明确输入坐标原点、
   LayerNorm、GELU、softmax attention、MLP 和 residual 的作用。
2. 建立两个冻结 ViT 的注册内部目标目录，在 400 张共享 evaluation 图像上完成
   336,800 个有限 target evaluation，并报告实际资源。
3. 定义 head、Q/K/V、routing/value 和两处 residual 的完整 edge cuts，使结构成分
   回到同一输入坐标并重构注册对象。
4. 用解析 JVP、匹配 activation replacement 和空间定向干预把 Identity、Correctness
   与 Utility 分开检验。
5. 给出同时包含正结果和负结果的经验边界：RQ3 呈现广泛但非普遍的 family-level 支持；
   RQ4 不支持，regional Taylor 在全部四个 family 中更强。

不能写成：

- 新的 input attribution 公式；
- 首次内部目标 attribution、首次 Q/K/V 分解或首次 activation patching；
- Endpoint 普遍优于 IG、regional Taylor 或其他解释器；
- 闭合或 JVP 一致性直接证明因果效用；
- 高分辨显示图天然具有更强空间信息。

## 3. 摘要逻辑

摘要按五句组织：

1. 问题：ViT 的 token、head、routing/value、MLP 和 residual 对象难以在统一输入坐标中
   扫描和核对。
2. 方法：对注册标量目标计算 $x\odot\nabla_xs(x)$，并通过完整 edge cuts 分解同一个
   最终目标。
3. 规模与正确性：两个模型、400 图、336,800 target evaluations；76,800 个正向闭合、
   28,800 个负例和 6,144 个 condition-scaled JVP checks。
4. Utility：C4a 11/12 family 完全满足标准；唯一例外 0.099565，其 CI 跨越 0.10；
   RQ3 广泛但非普遍地支持关联，冻结 12/12 omnibus 字段仍为 `NOT_SUPPORTED`。
5. 负结果与含义：C4b 0/4 family 同时胜过四个 mandatory baseline，regional Taylor
   全部更强，说明结构成分的全局排序效用不自动转化为空间区域选择增益。

## 4. 正文结构

### 1 Introduction

#### 1.1 问题

- ViT 内部解释对象横跨 token、channel、head、branch 和 residual stream。
- 只画 attention 或 feature map 不能说明对象如何依赖输入，也不能检查结构记账是否完整。
- 只证明链式法则身份也不能回答解释是否能预测有限干预。

#### 1.2 研究问题

- RQ1：能否在冻结 ViT 上完整、有限且可审计地扫描注册内部目标？
- RQ2：head、attention differential paths 和 residual cuts 能否无遗漏、无重复地闭合？
- RQ3：Endpoint 成分质量能否预测匹配 activation replacement 的有限效应？
- RQ4：完整输入分辨 component map 能否在同预算区域干预中超过总 G×I、raw gradient、
  regional Taylor 和随机区域？

#### 1.3 贡献

采用第 2 节的五点贡献，前言不放完整阈值表和恢复历史。

图：Figure 1（研究全景）。

### 2 Related Work

#### 2.1 NNS、单元可视化与 Gradient × Input

- Dou et al. (2023) 是 NNS-CNN 的方法来源。
- Captum NeuronGradient/InputXGradient 说明内部 scalar selector 与基础 G×I 均有明确先例。
- 本文贡献是 ViT 结构适配、统一注册、完整切分和实证审计。

#### 2.2 IG、conductance 与输入分辨内部归因

- IG 描述参考输入到当前输入的累计变化；Endpoint 描述当前点的一阶输入结构。
- Conductance 和 path attribution 提供隐藏单元输入分辨归因先例。
- Path IG 校准失败只限制跨模型 Path 比较，不改变 Endpoint 身份与 Endpoint-only 实验。

#### 2.3 Transformer 解释与结构分析

- attention rollout、Chefer/AttnLRP、visual analytics、LibraGrad；
- raw attention、relevance、修改 backward 的 attribution 与真实 Jacobian Endpoint 分列。

#### 2.4 Activation patching 与解释评价

- activation replacement 直接测有限效应；activation-space Taylor 是一阶近似；
- 评价必须固定 corruption/reference、目标、预算和统计单位。

图：Figure 2（方法身份）。Figure 3–4 放补充材料。

### 3 Method

#### 3.1 Endpoint learning image

$$
\mathcal L(x;x)=\operatorname{diag}(x),
\qquad
\mathcal L(v;x)=\sum_a J_{v,u_a}(x)\mathcal L(u_a;x),
$$

从链式法则得到

$$
L_s(x)=x\odot\nabla_xs(x).
$$

说明 canonical 输入是模型实际接收的 normalized RGB；当 $x=(r-\mu)/\sigma$ 时，
$x_i\partial_{x_i}s=(r_i-\mu_i)\partial_{r_i}s$。

#### 3.2 注册目标

- token-channel unit；
- patch-channel sum；
- patch-channel energy；
- attention-head internal coordinate；
- ground-truth 与 frozen clean-top1 raw logit；
- final CLS 和冻结中间模块目标。

ViT-B/16 为 457 个目标，DeiT-S/16 为 385 个目标。

图：Figure 7、11；Figure 7 建议补充，Figure 11 建议补充或方法附图。

#### 3.3 ViT 真实算子

- patch embedding；
- LayerNorm 和 GELU 的真实 Jacobian；
- unfused standard attention 的 Q/K/V、softmax、concat 和 output projection；
- 两次 residual 和 MLP；
- final norm 与 classifier。

图：Figure 5（主文），Figure 6、9（补充）。

#### 3.4 Structured endpoint decomposition

对最终标量 $F$、切面成分 $z_k$：

$$
E^C_{k,i}=x_i\left\langle\nabla_{z_k}F(x),
\frac{\partial z_k(x)}{\partial x_i}\right\rangle.
$$

完整 cut 满足

$$
\sum_kE^C_{k,i}=x_i\frac{\partial F(x)}{\partial x_i}.
$$

分别定义 head branch/total、Q+K+V、routing+Value、attention residual 与 MLP residual。

图：Figure 10（主文），Figure 15、17（补充）。

#### 3.5 Correctness 与 finite utility

组件 signed sum 和 absolute mass 为

$$
a_k=\sum_iE_{k,i},\qquad m_k=\sum_i|E_{k,i}|.
$$

解析局部方向合同：

$$
a_k=\left\langle\nabla_{z_k}F,J_{z_k,x}x\right\rangle.
$$

有限 replacement 效应：

$$
e_k=F(x)-F(x;z_k\leftarrow z_k^{\mathrm{ref}}).
$$

RQ3 主统计量是每图 $\operatorname{Spearman}(m_k,|e_k|)$，按图像聚合。

图：Figure 18–19。

#### 3.6 Spatial intervention

- block 5、每模型预注册三个 heads；
- 4×4 个 56×56 非重叠区域；
- 预算 0、1/16、2/16、4/16；
- normalized-zero 与 blur reference；
- baseline：total G×I、raw gradient、regional Taylor、20 个冻结随机排序；
- 指标：三 head 平均的 absolute isolated-effect AUC÷0.25。

图：Figure 23；真实示例用 Figure 24。

### 4 Experimental Design

#### 4.1 Models and data

- ViT-B/16 与 DeiT-S/16，均为 12 blocks；
- ImageNet validation 来源的 64/64/400 discovery/calibration/evaluation；
- 三集合图像与类别互斥，两个模型共享同一 400 图 evaluation cohort；
- resize 248、center crop 224、各模型冻结 normalization；
- canonical timm 0.9.16 unfused eval backend。

#### 4.2 Confirmatory analysis

- 12 个 C4a family，4 个 C4b family，28 个全局 Holm tests；
- 10,000 次 paired image-cluster bootstrap；
- 双侧精确符号检验；
- C4a 阈值 0.10，C4b 阈值 0.05；
- 有效覆盖、非零样本数与 CI 宽度门；
- 统计单位始终是图像。

#### 4.3 Audit and reproducibility

- immutable roots、source hashes、exact task coverage、negative controls；
- 独立 source-recomputing validators；
- 展示归一化图不进入统计。

### 5 Results

#### 5.1 RQ1: registered scanning is complete and tractable

- 800/800 image-model tasks；
- 336,800/336,800 finite target evaluations；
- wall time 1,752.71 s；4-card upper bound 1.9475 GPU-hours；
- formal throughput 192.16 targets/s；retained 9.290 GB。

图：Figure 12（固定样例）和 Figure 14（规模与资源）。

#### 5.2 RQ2: structural cuts close under positive and negative controls

- 76,800 正向闭合通过；
- 28,800 遗漏/重复负例全部拒绝；
- 最大 relative L2 `1.4144569831842091e-05`，低于 `1e-4`。

图：Figure 16。

#### 5.3 Local differential correctness

- 6,144 个 condition-scaled analytic JVP checks 通过；
- 最大 absolute error `2.648076753197781e-05`；
- 最大 `error/sum_abs_products` 为 `6.948755071487438e-06`。

图：Figure 18。

#### 5.4 RQ3: finite component effects are predicted overall

- 12 个 family 中 11 个满足全部确认性条件；
- family 中位相关范围 0.099565–0.306320；
- 唯一例外为 ViT-B × normalized-zero × routing/value：
  0.099565，95% CI [0.074783, 0.122609]，Holm $p=3.00\times10^{-14}$；
- 其置信区间跨越 0.10，因此 family 和 12/12 omnibus 字段保持 `NOT_SUPPORTED`；
- 论文解释为 `BROAD_BUT_NON_UNIVERSAL_FAMILY_LEVEL_SUPPORT`。

图：Figure 20（全 family）和 Figure 21（边界放大）。

#### 5.5 RQ4: full-resolution spatial maps do not add registered gain

- 0/4 C4b family 同时胜过四个 mandatory baseline；
- Endpoint−random 为 0.001069–0.010689，方向均正但低于 0.05；
- Endpoint−regional Taylor 为 −0.023433 至 −0.035630，四个 family 均为负；
- Endpoint 与 total G×I/raw gradient 接近，方向混合。

图：Figure 24（真实 map）、Figure 26（全部 contrasts）、Figure 27（family summary）。

### 6 Discussion

#### 6.1 Identity、Correctness、Utility 必须分层

- Identity 来自链式法则；
- Correctness 由闭合、负例和 JVP 检查支持；
- Utility 需要有限干预，不能从前两层推出。

#### 6.2 RQ3 与 RQ4 为什么可以同时成立

- RQ3 排序的是“哪个结构成分整体更重要”；
- RQ4 排序的是“同一 head 的哪个输入区域更重要”；
- 全局 component mass 能关联整体 replacement effect，不意味着其像素分布是最佳区域选择器；
- regional Taylor 直接使用候选区域的 activation displacement，因此更贴近 RQ4 的有限替换任务。

这段解释是由实验设计支持的任务差异，不进一步声称普遍机制。

#### 6.3 RQ3 边界例外

- 报告精确差值，不四舍五入为通过；
- 在正文只解释一次 strict omnibus 与 paper-level synthesis 的区别；
- 不为该 family 编造模型机制解释。

#### 6.4 对解释方法评价的含义

- 高分辨图、结构闭合、局部正确、有限效应预测和空间选择是不同属性；
- 评价应与预期用途对应；
- 强 baseline 不能被随机 baseline 的小幅优势替代。

图：Figure 28（结论层级）。

### 7 Limitations

集中写一次：

1. 两个 ImageNet 分类 ViT，不覆盖其他架构、任务和训练域；
2. canonical Endpoint 依赖输入坐标原点，不满足 baseline-difference completeness；
3. RQ4 限于 block 5、三个 heads、4×4 区域池、两种 reference 和当前 raw-logit AUC；
4. Path IG 的 ViT-B 校准未通过，因此没有跨模型 Endpoint–Path 正式结论；
5. 显示图按面板缩放，只能比较空间形态，数值主张来自 raw signed arrays。

### 8 Conclusion

结论保留三句话：

1. Endpoint Scanner 为 ViT 内部对象提供统一、可闭合、输入坐标化的扫描框架。
2. 有限干预支持结构成分级预测的整体有效性，但保留一个注册边界例外。
3. 空间排序未超越强局部 Taylor 基线，说明结构化解释需要按具体用途分别验证。

## 5. 主文与补充材料图表安排

### 主文建议

| 位置 | 图包编号 | 作用 |
|---|---:|---|
| Fig. 1 | 1 | 全研究链与四个 RQ |
| Fig. 2 | 2 | Endpoint、NNS-CNN 与 Path 身份 |
| Fig. 3 | 5 | Attention Q/K/V 微分路径 |
| Fig. 4 | 10 | 完整 structured edge cuts |
| Fig. 5 | 12 | 冻结 first-eight 双模型图册 |
| Fig. 6 | 14 | 注册规模、吞吐与资源 |
| Fig. 7 | 16 | 正向闭合与负例 |
| Fig. 8 | 20 | 12 个 C4a family |
| Fig. 9 | 21 | RQ3 唯一边界例外 |
| Fig. 10 | 24 | 真实 component 与 baseline maps |
| Fig. 11 | 26 | 16 个 C4b baseline contrasts |
| Fig. 12 | 28 | Identity–Correctness–Utility 总结 |

### 补充材料建议

- S1–S6：图包 Figure 3、4、6、7、8、9；
- S7–S9：Figure 11、13、15；
- S10–S13：Figure 17、18、19、22；
- S14–S16：Figure 23、25、27。

## 6. 表格安排

- Table 1：模型、预处理、目标数和 evaluation cohort；
- Table 2：RQ1 扫描规模与资源；
- Table 3：RQ2 闭合与局部 JVP correctness；
- Table 4：12 个 C4a family 的 estimate、CI、Holm p 和状态；
- Table 5：4 个 C4b family 相对四个 baseline 的 paired median difference。

## 7. 写作状态与作者待办

当前初稿已经有完整论文逻辑和全部封存数字。投稿前仍需作者完成：

1. 决定目标期刊/会议、篇幅和主文图数；
2. 填写作者、单位、通讯作者、基金与利益冲突；
3. 对 provisional bibliography 做最终 DOI、页码、大小写和 BibTeX 审计；
4. 决定标题是否保留 `NNS-ViT` 品牌词；
5. RQ3 使用“广泛但非普遍”，M1 与原确认性分析分开报告。
