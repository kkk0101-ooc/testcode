# Endpoint EP0：相关工作与写作审计

日期：2026-09-18。[协议冻结](NNS_ViT_Endpoint_Scanner_EP0协议冻结_20260918.md) · [带URL、定位、摘录与获取哈希的文献注册表](../manifests/nns_vit_endpoint/ep0_r1_20260918/literature_registry.json)。

本轮为定向prior-art审计，不以“未搜到”推导新颖性。官方网页的摘要、可定位公式/API源码及本地原NNS正文为证据；未全文核对的条目明确限定为摘要范围。网络搜索初次无有效结果后使用原始来源直读；Conductance的错误PMLR候选URL返回404，采用arXiv1805.12233；NeuronConductance源码超时后读取官方API。

## 1. 对象—先例—技术差异

| 本文对象 | 本轮核对的先例 | 已覆盖内容 | 本文仍可评价的具体内容 |
|---|---|---|---|
| 输入值seed、当前状态传播 | Dou et al., Neural Networks168:484–495,2023，DOI10.1016/j.neunet.2023.09.030；本地英文`3式1–14 | 坐标初始化、线性权重、ReLU当前门、pool winner | 对真实ViT Jacobian的适配与paper-formula CNN回归；原文末端门记号按一致解释处理 |
| 内部scalar/aggregate扫描 | Captum NeuronGradient，commit413e888e1e8f22cf8da1631cc9ca2133ce30a9db，class/attribute selector | 单神经元、slice求和、callable aggregate及输入形状梯度 | ViT固定目标目录、结构重构ID、覆盖与成本；不把hook或selector本身称新算法 |
| Endpoint基础量 | Captum InputXGradient官方API | 输入乘梯度 | 明确G×I身份与坐标原点，不重复命名基础公式为创新 |
| 输入分辨hidden attribution | Dhamdhere et al., How Important Is a Neuron?, arXiv1805.12233；Captum NeuronConductance API | conductance、隐单元流量、输入形状输出、ablation评价 | Endpoint与Path成分的统一记账及有限干预差异；输入分辨率不是独有性质 |
| Path-integrated版本 | Sundararajan et al., ICML2017, PMLR70:3319–3328 | IG与相关公理 | Endpoint成本—质量实测，不宣称IG是新NNS公式 |
| attention/skip传播 | Chefer et al., CVPR2021:782–791；官方abstract | relevance传播与跨层守恒、attention/skip | 真实Jacobian与其Deep Taylor/relevance规则不同；本轮不作全公式等价判断 |
| transformer input/latent attribution | Achtibat et al., ICML2024, PMLR235:135–168；官方abstract | attention-aware LRP与latent attribution、低反向成本 | 真实Jacobian Endpoint与自定义relevance规则分列 |
| attention路径/残差相关先例 | Abnar & Zuidema, ACL2020:4190–4197；AtP*`3.1.1、附录C.2 | rollout/flow；QK饱和、edge patching与残差direct effect | 精确输入坐标成分的合同及C4b评价；不主张首次研究head/QKV/残差 |
| activation Taylor基线 | Kramár et al., AtP*, arXiv2403.00745v1，`2.2式4–5 | clean/noise activation差×clean gradient，注意其loss/sign定义 | $p^{ref}$为mandatory先例基线；本协议统一用F(clean)−F(replace)符号 |
| corruption与效应指标 | Zhang & Nanda, arXiv2309.16042v2，摘要与`3–4定位 | patching的corruption/metric选择影响结论 | 冻结zero/blur、区域预算、signed/absolute区分 |
| Transformer梯度流分析 | Mehri et al., LibraGrad,CVPR2025:67–78；官方abstract | backward路径pruning/scaling、gradient flow imbalance | Canonical Endpoint保留真实导数；不把修改后的backward当其实现 |
| 原NNS filter aggregate | 原NNS`4.2.1激活加权式 | $\sum h_tL_{h_t}$ | energy是该式的代数复现，不作为新energy算法 |

参考材料的发表年与本次API快照年份分开；Captum条目的2026是获取快照年。除固定Captum源码外，其余在线API/网页按获取SHA标识；未将任何外部代码下载后导入或运行。

## 2. 投稿定位决策

冻结为 `REPRODUCTION_UNIFIED_IMPLEMENTATION_AND_SYSTEMATIC_EMPIRICAL_AUDIT`。正文可陈述：列明条件下继承原NNS endpoint原则；实现ViT目标注册和精确结构边；评价实现正确性、成本与有限干预。尚不能陈述：新基础归因公式、首次输入分辨hidden attribution、首次QKV/残差分析、普遍性能优势或新因果机制。

C4b测量的是在固定窗口池、预算和replacement下，成分图是否比总图/梯度/区域Taylor/随机排序提供额外区域选择信息。该实证问题具有独立可检验性，但其结果与新颖程度都不在EP0预先判为成功。

## 3. Academic-defensive-writing-auditor 审计

| Location | Original | Type | Severity | Necessary caveat? | Why it feels defensive | Likely reviewer reaction | Recommended action | Rewrite |
|---|---|---|---|---|---|---|---|---|
| 论文定位 | 拟议“结构化分解作为核心新增” | MIXED/D14 | Major | 是 | 已有head/edge/gradient工作，直接贴新增标签缺少差异 | 要求明确先例和实现增量 | 按审计结果收紧定位 | 原始NNS的ViT适配、统一实现与系统实证审计 |
| 样本量 | 固定400图 | CLEAN | Minor | 是 | 非防御，但不能省略设计依据 | 希望知道精度而非任意整数 | 写具体二项精度与CI出口 | 符号比例最不利95%近似半宽0.049；相关/效应精度由冻结CI宽度判断 |
| 局部干预 | 导数一致预测干预 | MIXED | Major | 是 | 易将定义恒等式包装为效用 | 要求独立有限效应 | EP5-A只归Correctness | EP5-B/C4b用有限replacement与同预算区域评价Utility |
| 后续资源 | 320 GPU-hours可完成 | D10/MIXED | Major | 是 | 无P95测量支撑完成预测 | 要求测量来源 | 删除完成承诺 | 320小时为前瞻性保护上限，EP1/EP2后在cap内测量和调度 |
| 数据隔离 | 全新未见过图像 | D13/MIXED | Major | 是 | 无法证明全历史与全球未见 | 要求排除台账范围 | 限定所核对来源 | 排除本地1000及登记的历史元数据并集1098个ID |

### Paper-level defensive-writing score

2/10（编辑判断）。冻结说明以具体定义、数值和权限为主，未用性能形容词补偿未完成实验。

### Highest-priority removals

1. 删除未由prior-art核对支持的“首次/新公式”表述。
2. 将“能完成”的资源承诺改为有停止机制的hard cap。
3. 将局部导数验证从Utility移到Correctness。

### Necessary caveats to preserve

原NNS门控解释；same-tensor energy；两模型支持范围；公开页面核查深度；部分历史访问排除范围；真实图像payload尚未验证；EP0 PASS不授权后续生产。

### Repeated phrase audit

协议说明与本审计均按“定义—证据—一次范围限定”组织。技术操作约束集中于对应合同，不以审稿人假设问题组织正文。
