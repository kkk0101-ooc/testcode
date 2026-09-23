# Revision v7 六维对抗式终审报告

## 总评

终审对象是最终中英文 Markdown、由其确定性生成的中英文 LaTeX/PDF、双语补充材料、共享 references.bib 和 v7 r4 机器证据。本报告不继承 v6 的“可送审”判断；所有结论重新核对当前文件、原始逐图值、独立复算、编译日志、PDF 文本/metadata 和视觉结果。

当前稿的核心贡献边界清楚：基础 Endpoint 是同目标 Gradient × Input；新增内容是目标条件化的 use×input 双索引局部微分、结构合法切面合同及其审计，以及固定 ViT 参数状态上的空间分布分析。v7 分层结果削弱了旧 pooled Table 6 叙事，终稿据此将 A/V 与 attention/MLP 分开，并把 pooled family 降为历史混合描述。H4 和区域任务的阴性结果完整保留。

## 六维复审

### 1. 研究问题与科学价值

| 问题 | 最终稿位置 | 直接证据 | 影响 | 裁决 |
| --- | --- | --- | --- | --- |
| 双索引是否只是 Gradient × Input 重命名 | 引言、§3.1、§4.1、§6.3 | 总图消去 use，edge scalar 消去输入位置；Q/K/V entropy 与 residual TV 需两个索引 | 基础公式无新颖性；联合索引与切面审计提供可回答的新对象 | 通过 |
| 与强近邻的差异是否具体 | §2 Table 1、S4/S5a | 五个近邻族按对象、索引、重构目标比较 | 不再以“把梯度用于 ViT”主张贡献 | 通过 |
| 经验发现是否被包装为普遍机制 | 摘要、§6、§8 | 结论限定固定 checkpoint、target、reference 和 replacement contract | 科学价值定位为方法/审计与受限状态描述 | 通过 |

### 2. 研究设计与因果逻辑

| 问题 | 最终稿位置 | 直接证据 | 影响 | 裁决 |
| --- | --- | --- | --- | --- |
| 结构 cut 与数值闭合是否混淆 | §4.3 | 平方支路零导数反例；机器 counterexample | 单点闭合不再被用作结构完整性的证明 | 通过 |
| 命题 2 是否对任意 target/cut 过度适用 | §4.1 target 表、§4.3 | 目标须在 cut 下游且全部输入路径经过 cut；registry 只声明目标 | 457/385 个目标不再被统称适用同一 residual cut | 通过 |
| 分支分组、全局 cut 和替换变量是否混用 | §4.2、§4.6、S16 | Q/K/V 只闭合 branch；加入 bypass 才闭合 Total Endpoint；A/V/head/branch 是实际替换变量 | Identity、Correctness、Utility 分离 | 通过 |
| 区域任务是否被误写为像素总效应 | §4.6、§7.1 | 混合输入产生指定 head activation，再插入 clean forward | 结论限于两阶段指定-head replacement | 通过 |

### 3. 方法学与统计推断

| 问题 | 最终稿位置 | 直接证据 | 影响 | 裁决 |
| --- | --- | --- | --- | --- |
| p 值是否错误归因于 10,000 bootstrap | §5.3 | bootstrap 只给区间；p 来自图像级双侧精确符号检验 | 极小 p 值来源透明 | 通过 |
| tie、单双侧和 multiplicity 是否明确 | §5.3；pvalue_audit.csv | tie 排除二项计数但留在 n/CI；原 28 项与 H1–H4 分开 | 避免测试族混淆 | 通过 |
| A/V 与 residual 混合是否扭曲 Table 6 | §7.1 Table 6、S18 | ViT-B zero A/V 为 0.049/0.224，attention/MLP 为 -0.028/0.287；O+B residual 为 -0.128 | pooled 相关不能解释为类型内部统一排序能力 | 通过，主张已收缩 |
| H4 是否可独立复算 | §6.6、S18；H4 CSV | 四 checkpoint 为 -0.114/-0.186/-0.186/-0.136；独立误差 $2.22\times10^{-16}$ | 公式、NA、计数与 Holm 完整 | 通过 |
| 自定义指标与质量诊断是否有依据 | §4.5、S18 | locality 半径 0/1/2 同方向；TV 在四分位均存在 | 不选择新主指标，也不声称排除噪声 | 通过 |

### 4. 结果解释与外推边界

| 问题 | 最终稿位置 | 直接证据 | 影响 | 裁决 |
| --- | --- | --- | --- | --- |
| 11/12 是否掩盖 all-family 阴性 | 摘要、§7.1、§8、S2.1 | 11/12 只作 provenance；all-family 始终 NOT_SUPPORTED | 主结果按连续效应与分层报告 | 通过 |
| 分层后是否仍支持统一叙事 | Table 6、S18 | O+B 调整结果依模型/reference 改变方向 | 统一内部排序主张撤回 | 通过 |
| 坐标敏感性是否完整保留 | §6.7、S17 | DeiT-S Q 的 B11−B0 为 -0.004/0.003/0.002，区间均跨零 | 不声称一般参数化稳健性 | 通过 |
| 阴性效用结果是否被弱化 | 摘要、§6.6、§7.1、§8 | H4 四项为负；区域 local map 无稳定实用增益 | 结构分辨率不等于干预预测增益 | 通过 |
| 外推范围是否完整 | §8.2 | ImageNet、patch-16、固定 scalar、规定 replacement、非受控 checkpoint | 不外推训练因果或其他架构 | 通过 |

### 5. 学术写作与叙事结构

| 问题 | 最终稿位置 | 直接证据 | 影响 | 裁决 |
| --- | --- | --- | --- | --- |
| 主文是否像工程日志或审稿回复 | 全文；defensive_writing_audit.md | D1–D14 得分 2/10 | 保持 claim→evidence→boundary | 通过 |
| 中英文是否同步 | 四版本验证 | 章节、21 个 display equations、11 图、表和关键数字一致 | 避免翻译漂移 | 通过 |
| 文献与术语是否可核验 | Table 1、S4.1、references.bib | 六项 2026 文献标明 arXiv 版本/DOI/访问日期，Bib 展开进 PDF | 不显示工程路径 | 通过 |
| LaTeX/PDF 是否可读和匿名 | build/validation/visual reports | 连续两次干净编译；119 页逐页检查；metadata/路径扫描 | 满足匿名预审稿要求 | 通过 |

### 6. 改进路径

| 问题 | 最终稿位置 | 直接证据 | 影响 | 裁决 |
| --- | --- | --- | --- | --- |
| 当前轮是否仍需新模型实验 | v7 protocol、terminal receipt | 核心问题由旧 E4/E5/M3/v6 原始记录 CPU 复算回答；GPU inference=0 | 不扩大实验掩盖阴性结果 | 本轮不需要 |
| 是否仍有阻断或重大修改项 | 本报告、完成门审计 | 定义、H4、分层、四版本、编译、视觉和匿名均有证据 | 仅剩命令回归和 Git 发布审计 | 无科学阻断/重大项 |
| 后续研究最小增量 | §8.2 | 当前设计未控制训练、规模、数据，也未验证其他架构 | 新主张须另立前瞻协议 | 非本轮条件 |

## 十个复审问题

### 1. 结构合法切面与点态微分闭合有什么不同？

结构合法切面是 tensor-use DAG 的图论性质：每条输入到目标 use-path 恰穿一次。点态微分闭合是在固定输入和导数约定下，use-specific Jacobian/cotangent 分组之和等于目标梯度。前者不依赖数值，后者依赖当前点。

### 2. 为什么单点数值闭合不能发现零导数路径遗漏？

在 $s(x)=x+x^2$、$x=0$ 时，平方支路结构存在但导数为零。遗漏它仍得到导数 1，数值误差为零；结构 cut 却不完整。

### 3. 命题 2 对哪些目标和 cut 适用？

只适用于位于 cut 下游、且所有输入到目标结构路径都经过该 cut 的标量目标。最终 logit 可用完整 residual cut；内部目标只能使用其祖先子图中的上游 cut。

### 4. 为什么极小 p 值不受 10,000 次 bootstrap 分辨率限制？

Bootstrap 只估计中位数区间。p 值由非 tie 图像的正负计数做双侧精确二项符号检验，并按冻结 family 作 Holm 调整。

### 5. A 与 V 分开后，原 pooled 相关如何解释？

它只描述 24 个 A/V 混合观测的整体秩关系。ViT-B 的 V-only 明显高于 A-only，DeiT-S normalized-zero 的 V-only 区间跨零；pooled 值不能解释为两者有相同能力。

### 6. 层级控制后，Table 6 的结论是否仍然成立？

不成立为统一规律。A/V O+B 在 ViT-B 为 0.030/-0.018，在 DeiT-S 为 0.089/0.095；residual O+B 为 -0.128/0.042/-0.091/0.142。pooled 正相关只保留为混合 family 描述。

### 7. H4 的两个分数具体是什么？

$M_{jbh}=\sum_i|\mathcal L_{jbh,i}|$。$G^{AG}_{jbh}=|\langle z_{jbh},\nabla_{z_{jbh}}s_j\rangle|$，先对 attn.proj 输入的整个 head slice 点积再取绝对值。二者与 normalized-zero head replacement absolute effect 在 heads 内作 Spearman。

### 8. 区域任务估计什么？

它先以区域替换输入产生目标 head activation，再把该 activation 放入 clean forward，保持其他 heads 和 bypass clean。因此是两阶段指定-head activation replacement，不是总像素效应。

### 9. 三种坐标量分别测量什么，哪些结论敏感？

三者是 $x\odot\nabla_xs$、$\nabla_rs$ 和 $r\odot\nabla_rs$。Unit、K/V、residual TV 的 phase 方向保持；DeiT-S Q 的 B11−B0 点估计变号且三段区间跨零。

### 10. H4 和区域实验为负时，双索引方法还有什么价值？

它使 Q/K/V entropy、routing/value 空间差异、residual output–identity TV 和 cut 合同检查可定义。该价值是 representation 与 correctness；现有证据不支持普遍 finite-intervention ranking gain。

## 风险与终审结论

- 阻断问题：0。
- 重大问题：0。
- 一般问题：匿名 article-style 预审稿不是学校归档模板；指定模板后需重新排版但不改变科学内容。
- 修辞问题：防御性写作评分 2/10；必要 caveats 已集中保留。

在当前匿名预审稿任务范围内，六维复审结论为：**可送审**。
