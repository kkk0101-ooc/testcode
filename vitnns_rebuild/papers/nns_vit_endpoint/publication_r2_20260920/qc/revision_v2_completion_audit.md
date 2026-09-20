# 修改意见 v2 完成审计

## 目标与完成标准

目标是对 修改意见/修改意见v2.md 指定的论文初稿执行送审前重构：同步修改中英文主文；把 NNS 定位为一般计算图上的局部 Jacobian 表示；明确与 Input × Gradient 的数学身份；建立 ViT attention、Q/K/V、multi-head 与 residual 的结构化 differential；重组研究问题、实验、结果和结论；纠正随机基线；补齐指标、复现与文献边界；精简主文图件；保留缺乏直接证据的增强项为未执行范围。

完成判据：修改意见第二十节的十项第一优先级全部有直接稿件或数据证据；双语结构、公式、表格、图序和结论边界一致；既有封存证据未经改写；校验器直接覆盖上述要求。

## Prompt-to-artifact checklist

| 修改意见章节 / 明确要求 | 状态 | 直接产物与核验 |
|---|---|---|
| 一：NNS 提升为一般计算图局部微分语言；承认 Input × Gradient；覆盖 attention/residual；界定能力 | 完成 | 双语摘要、Introduction、Sections 3–4、8；命题 1–2 |
| 二：重设题目，避免“新基础归因/普遍优越” | 完成 | 中文题名“计算图解释与结构路径分析”；英文同义题名；摘要明确 base quantity 非新 |
| 三：一个总问题与 RQ1–RQ4 | 完成 | 双语 Introduction 中编号 RQ1–RQ4；校验器检查四项且拒绝 RQ5 |
| 四：理论、方法、经验、有限干预四层结论 | 完成 | 双语 Abstract 与 Section 8.2 四点结论；经验结论绑定两 checkpoint |
| 五：四项创新点 | 完成 | Introduction 四项 contribution；文献边界将“建立”收紧为“在 NNS 框架下系统化” |
| 六：一般定义、标量等价命题、证明、Euler/非齐次限制、边归因、合法切面及反例 | 完成 | Sections 3.1、4.3；含 G=(V,E)、精确条件、函数值非完整性、多输入/共享边、遗漏 bypass 与重复 branch 反例 |
| 七：routing/value、Q/K/V、multi-head 与 output projection | 完成 | Section 4.2；使用 Q/K/V-induced differential，明确非前向成分，给出 E-MHA 等于各 head 项之和 |
| 八：residual shared cotangent | 完成 | Section 4.2；identity 使用 g_y，并说明使用完整 nabla_u F 会重复记账 |
| 九：RGB/patch mass、entropy、locality、distance、routing/identity scale、Value-only error、绝对范数 | 完成 | Section 4.4；零质量为 NA；Sections 6.3–6.4 报告相对尺度与 absolute mass |
| 十：四组实验、图像为统计单位、closure 先于可视化、真实前向变量干预 | 完成 | Section 5.4 与 Results 6.1；Section 4.5 明确实际 activation replacement；补充材料 S1.1 伪代码 |
| 十一：clean-only/reference-aware 区分；随机 baseline 绝对值顺序；效应量与区间 | 完成 | Section 4.5、7.1、Tables 3–7；补充 S5/S9；20-repeat corrected estimator 直接重算 |
| 十二 1：随机初始化或额外 checkpoint | 完成最低建议 | Supplement v1：两架构各 3 个随机初始化、32 图、12 blocks；替代 trained checkpoint 未执行 |
| 十二 2：抽样敏感性 | 完成现有样本敏感性 | Section 6.9 / Supplement S13：channel、token-stratum、image bootstrap 与 leave-one-out；独立 channel seeds 未执行 |
| 十二 3：目标异质性 | 完成主要项 | Section 6.7 / Supplement S11：GT、top1、margin 与 correct/incorrect；未另分 high/low confidence |
| 十二 4：patch 内细粒度控制 | 部分完成 | Section 8.1 明确 16×16 patch embedding 约束；尚无像素置乱专门实验，不进入结果主张 |
| 十三：模型范围和外推边界 | 完成 | Abstract、Sections 5.1、8.1、8.2 均限定为两个固定 ImageNet 预训练 checkpoint |
| 十四：八章论文结构 | 完成 | 主文依次为 Introduction、Related Work、Jacobian、Structured NNS、Validation、Depth、Intervention、Discussion/Conclusion |
| 十五：摘要重写 | 完成 | 摘要包含问题、等价性、结构分解、两模型结果、有限干预边界与贡献定位 |
| 十六：绪论核心表述 | 完成 | Introduction 区分表示统一与结构扩展，并把结构扩展定义为主要技术增量 |
| 十七：结果语言收紧 | 完成 | 使用 local dependence/differential/relative norm；未使用“真正学到/理解/完整因果贡献/普遍优越” |
| 十八：主文保留约 11 张核心图，重复热图与展示对比移附录；signed/absolute 可查 | 完成 | 主文 Figures 1–11；figure_manifest main_text_figure_map；Supplement S8 链接 11 张移出图；所有 22 张源图保留 |
| 十九：checkpoint、软件、预处理、eval、seed、VJP、时间/显存/存储、代码与伪代码 | 完成 | Supplement S1、S1.1、Table S2、Supplement-experiments protocol 与 evidence JSON |
| 二十：第一优先级 1–10 | 完成 | 对应以上一至十一项；校验器检查题名、RQ、公式、术语、随机估计器、章节、图序和文献 |
| 二十：第二优先级增强项 | 部分完成且不冒充 | Value-only error、absolute norms、sampling sensitivity、random initialization、target heterogeneity、cost 已完成；多幅度替换和替代 trained checkpoint 未执行 |
| 二十：第三优先级写作展示 | 完成 | 主文 11 图、统一术语、探索/历史详情放补充、阈值与连续估计并报、图注收紧 |
| 二十一：NNS 到数学位置、ViT 问题、结构化 NNS、验证、路径规律、边界和总结闭环 | 完成 | Sections 1–8 顺序与四项结论 |
| 二十二：最终定位 | 完成 | Abstract、Introduction 2.5 与 Conclusion 均定位为 NNS 理论边界/对象扩展，不包装 Gradient × Input |
| 命名文件与入口 | 完成 | 当前稿：zh/en manuscript；双语 supplement；supplement experiments；references；README；v2 QC |
| 命令与测试 | 完成条件 | validate_publication_r2.py、verify_publication_r2_methods.py、validate_endpoint_supplement_v1.py --paper；最终输出记录在本文件末尾 |

## 明确未执行的增强项

修改意见把多幅度 replacement、替代 trained checkpoint、独立 channel-seed 重跑、额外输入变换和 CNN 展示列为第二/第三优先级或“如果资源允许”的增强项。当前封存证据不含这些实验。本稿在 Section 8.1 与 Supplement S14 中明确范围，不以 JVP、重采样或其他代理结果替代它们。第一优先级送审项不依赖这些增强项。

## 最终验证记录

- validate_publication_r2.py：PASS，1,707 项检查，0 errors；覆盖 11 张正文图映射、22 张完整图件及哈希、92 张历史补充图、双语章节/公式/表格、链接、v2 题名/RQ/术语、随机基线顺序、镜像稿与 165 个真实显示面板复算。
- verify_publication_r2_methods.py：PASS；9 个 image-model 任务的 Endpoint 与原生图逐项一致，Grad-CAM 最大绝对差不超过 2.80e-9，rollout 不超过 1.05e-17。
- validate_endpoint_supplement_v1.py --paper：PASS，32,091 项检查，0 errors；覆盖 456 个补充任务（64 feature、192 random、200 target）及 source/payload/closure/论文链接。
- verify_endpoint_supplement_statistics.py：PASS，63,448 项独立复算；重算 672 个 M2 检验与 61,440 个 feature pairs。
- pytest tests/test_endpoint_supplement_v1.py -q：3 passed。
- 双语专项检查：章节编号、display equations、61 条表格行、Figures 1–11 路径均一致；git diff --check 通过；正文与 QC 文件无控制字符；防御性套话扫描为 0。
- 源意见 SHA-256：de2447782e90a028c139e7c31114de3fb5560ab5d775fcbd5e20798b5d4f8ffb。

结论：修改意见 v2 的十项送审前第一优先级要求均已完成并有直接证据。未执行项均属于意见明确列出的条件性增强或后续控制，已在正文和本审计中显式标注，没有被代理指标冒充为完成。
