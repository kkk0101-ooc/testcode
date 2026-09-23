# Revision v6 匿名稿硕士论文标准终审

## 总评

评审对象为 publication_r2_20260920 的最终中英文匿名稿及本地补充证据。评审按硕士毕业论文的科学要求检查当前投稿稿，不把它冒称已经套用学校模板的正式学位论文。送审判断依据最终定义、原始计算及负结果，不依据执行记录中的 PASS 宣告。

当前稿件形成了清楚但有限的研究贡献。基础测量是已有 Gradient × Input。双索引表示将指定 consumer use 的当前点微分保留到原始输入坐标，并以分支与全局两类重构对象组织审计。经验分析回答固定 targets/checkpoints 上的空间分布及规定替换的排序问题。结构分辨率与有限干预增益没有被混同。

初审阻断项中的对象混用已修正。解析参考和两类必需 mutation 有直接结果。坐标分析完成完整 cohort 并保留 DeiT-S Q 例外。H4 四 checkpoint 反向和 all-family 未获支持均出现在摘要和结论。匿名选择解决身份占位符问题。当前未发现需要改变科学设计或主要结论的未处理阻断/重大问题。

## 六维审查

### 1. 研究问题与科学价值

**问题。** 双索引是否只是给已有梯度量重新命名，还是增加了可回答问题。

**证据位置。** 正文 §3.1 直接给出 Gradient × Input 身份；§4.1 定义 use 与输入两个索引；§6.3/Table 4 比较总图、component scalar 和图组。§2/Table 1 同列 conductance、DecompX、LRP、EAP、Vi-CD 和 Unpack。

**影响。** 单项归因公式没有新颖性。保留指定分项的空间分布，使 Q/K/V entropy 和 output–identity TV 可计算，而单一总图或 scalar 不能恢复各分项。这是适合硕士研究的实现、审计与受限经验贡献，但不应被表述为通用归因理论突破。

**修改建议与判断。** 维持当前两点方法贡献及经验结果分列。当前范围可接受，未要求新增大型 benchmark 来支撑未提出的普遍优越性主张。

### 2. 研究设计与因果逻辑

**问题。** 全局 cut、branch partition、path scale 与有限替换是否仍被相互替代。

**证据位置。** §4.2–4.3 及 Figures 2–4 区分 A_F 与 L_F；§4.6 和 S16 的 A、V、head-output、residual-branch 表列实际前向变量与 slot。stage0_definition_evidence.json 对原数组直接重构，legacy_spatial_source_audit.json 区分 28,800 branch 与 48,000 global closures。

**影响。** Q/K/V 或 routing/value 单独闭合到 attention contribution，identity 纳入后才恢复 Total Endpoint。有限替换改变前向状态，不满足一般加法分解。正文已避免把闭合当成有限效应的因果充分性。共享模型上的层号轨迹也不能识别深度或训练的因果作用。

**修改建议与判断。** 保留“固定 checkpoint 上与 block index 相关”的叙述。S16 充分说明重算范围，当前设计与问题一致。

### 3. 方法学与统计推断

**问题。** 推断单位、目标抽样、多重检验、阈值与输入参数化能否支持主张。

**证据位置。** §5.1–5.3，Tables 4–10，S15–S17；cohort_bindings.json、analysis_identity.json、original_inference_recomputed.json、coordinate_sensitivity_summary.csv。原 100 图 unit study 是 400 图 evaluation 子集，主文已修正。

**影响。** 图像级区间条件于固定目标与 checkpoint，不代表新 neuron/head 总体。28 项 Holm、单元五项、attention 四项、各 residual 两项、事后四项及独立队列四项分开保留。0.10/0.05 是研究者操作门槛，正文没有借阈值微调取得正结果。完整 1,000 个坐标任务与 27,600 个图像级 contrast 完成，DeiT-S Q B11−B0 的符号变化和跨零区间阻止全参数化稳健性表述。

**修改建议与判断。** 继续保持 normalized-input 主范围，并将对象级三表示方向保持单独报告。Table 6 用原确认性区间；Tables 7–10 新区间明确描述性。无未处理重大统计缺口。嵌套重采样只证明对已有抽样层级的敏感性，不能替代新 neuron 总体实验。

### 4. 结果解释与外推边界

**问题。** 有利趋势是否掩盖 H4、模型依赖、坐标原点及训练状态的替代解释。

**证据位置。** 摘要、§6.4–6.7、§7、§8；coordinate_decisions.json；E5 原始任务重算的四个 H4 中位数均负；原 all-family C4a/C4b 均 NOT_SUPPORTED。

**影响。** 随机参数也存在空间扩展，结构几何与输入坐标加权均可贡献观察到的轨迹。不同公开 checkpoint 混合训练方案与规模因素，不能估计训练因果。四 checkpoint 检查由同团队、同代码与指标执行，属于独立队列稳健性而非团队外复现。H4 表明结构分辨率并不自动提升干预排序。

**修改建议与判断。** 当前讨论已经覆盖这些替代解释。没有把阴性结果降为实现失败，结论与证据匹配。

### 5. 学术写作与叙事结构

**问题。** 最终稿是否仍像审稿回复或工程日志，图注是否能独立理解。

**证据位置。** 摘要及引言两点方法贡献；§6.7 控制分析入口；Figures 1–11 图注；defensive_writing_audit.md；中英文公式、表格数值与章节自动比较。

**影响。** 主文删除 NOV 和阶段标签，控制分析集中补充，工程路径移出。图注说明模型/变量、n、聚合、误差带或无误差带、显示规则和分析身份。10 张主表较宽，适合本轮 Markdown 审阅；目标模板排版仍可能需要纵向拆表，但不影响统计含义。

**修改建议与判断。** 稿件按匿名送审准备，作者身份不再是 blocker。模板排版属于后续指定模板后的形式工作。中文个别段落仍使用领域英文术语，但已明确定义，未构成重大写作缺陷。

### 6. 改进路径

**问题。** 是否仍有必须在本轮追加的实质性修改。

**证据位置。** 最终完成清单及直接来源；测试验证器覆盖的是具体合同，未把测试数当作一般正确性证明。

**影响。** 核心概念、最低必要实验和推断范围已经对齐。继续增加模型、选择新原点或搜索子集不会修复本轮剩余的问题，反而会改变研究范围。

**修改建议与判断。** 本轮仅完成最终 package 验证及 Git/远端一致性审计。未来若主张一般参数化稳健、训练机制或干预预测增益，应另立前瞻协议。该未来路线不是本轮送审的必备条件。

## 证据与结论不匹配检查

| 论文位置 | 当前结论 | 直接证据 | 缺口或替代解释 | 允许表述 | 状态 |
| --- | --- | --- | --- | --- | --- |
| §4.2–4.3 | branch 与 global 分别重构 | 原 E3 raw arrays、cut_reconstruction、解析参考 | branch 单独缺 bypass | branch contribution / Total Endpoint 分开 | 已解决 |
| §4.6/S16 | A/V forward replacement | _explicit_attention_runtime、component_rows、原运行 rows | 导数项不是前向 tensor | A-probability 与 V-projection replacement | 已解决 |
| §6.2 | 单元空间轨迹 | 原 200 tasks 的 columnar records | targets/checkpoints 固定 | normalized-input 下固定目标的描述性轨迹 | 已收缩 |
| §6.4/S17 | Q 模型与坐标依赖 | 400 图 Q/K/V、三表示完整结果 | DeiT-S Q B11−B0 跨零且变号 | 不合并为普遍规律 | 已保留 |
| §6.6 | 四 checkpoint 方向检查 | 1,600 task JSON/NPZ、6,400 值复算 | 同团队/代码，非独立训练种子 | 独立图像队列稳健性 | 已解决 |
| §7/Table 6 | 11/12 family 达操作门 | 原 16 families/28 tests 完整重算 | CI lower>0.10 更强，非同一规则 | 广泛但非普遍；all-family 未支持 | 已保留 |
| §7/Tables 7–10 | Taylor 在规定任务更适配 | 原 component/curve 重算 | 位移信息预算不同 | reference-aware 代理排序结果 | 已解决 |
| §8/H4 | 结构信息不自动带来预测增益 | 四 checkpoint H4 全反向 | 图组价值不在该排序增量 | 结构分辨率、审计和分项空间测量 | 已保留 |

## 风险等级

**阻断毕业。** 当前科学稿范围内为零。没有未解决的定义冲突、缺失核心结果或被删除的负结果。

**重大修改。** 当前为零。坐标不一致没有靠改原点或缩 cohort 回避，结论已收缩。解析参考与 mutation 的能力边界写明。

**一般修改。** 若另行要求正式学位模板，需要学校模板和具体排版要求；本轮交付是匿名投稿稿，以硕士标准审查其科学内容。该条件不虚构为已完成的学校归档手续。

**修辞优化。** 可在目标刊物字数约束下继续缩短相关工作和拆分宽表，保持全部数字与负结论。

## 十个答辩问题

### 答辩 1. Q/K/V 分组与全局切面各闭合到什么

Q/K/V 和 heads 分组重构 attention branch contribution A_F。Identity 加完整 branch partition，或两次 residual 的完整输入集合，重构同一目标 Total Endpoint L_F。见 §4.2–4.3、Figure 2、stage0_definition_evidence.json。

### 答辩 2. Routing differential 与 attention probability 有何区别

Routing-induced 是 (dA)V 经当前 cotangent 收缩后的微分组，等于 Q-induced 加 K-induced。A 是实际 softmax 前向概率张量，能作为有限替换变量。二者由不同运算定义。见 §4.2 四类对象表、S16 变量表。

### 答辩 3. Table 6 实际替换什么

Head 替换 attn.proj 输入的指定 64-channel slice；A 替换 A@V 的左输入、shape [1,H,197,197]；V 替换右输入、shape [1,H,197,64]；attention/MLP branch 替换两次 residual add 的 branch 输入。Reference 来自完整 reference forward，其余前缀保持 clean，下游重新计算。见 S16 与原 E4 component rows。

### 答辩 4. 三坐标结论是否一致

主要 phase 方向在三表示中一致。旧 QKV 对比实际为 B11−B0；DeiT-S Q 中位数约为 −0.0038、0.0030、0.0020，三段区间跨零。故不能声称全部规律稳健。见 §6.7、S17 的全 96 行及 coordinate_decisions.json。

### 答辩 5. 主区间推广到哪个总体

推广对象是固定目标、checkpoint 和替换合同下的图像变化。目标 strata/image 重采样另作敏感性，不能据此推广到所有新 neuron/head。见 §5.3、主表 Inference 列和 S15。

### 答辩 6. 四 checkpoint 支持什么

支持四个指定公开参数状态中、独立图像队列上的方向检查。模型规模、训练 recipe 与 pretraining 数据不受随机控制，分析团队和代码也相同，不能识别训练因果或称为团队外独立复现。见 §5.1、§6.6、§8.1 和 checkpoint hashes。

### 答辩 7. 独立参考和 mutation 如何降低共同错误

NumPy 参考从 affine、LayerNorm、softmax Jacobian 和 Q/K/V 显式构造分量，不使用主 scanner 的 component 函数。驱动将逐元素结果与真实 scanner 比较，并用中心有限差分检验固定方向。Wrong-slot 可在总和仍闭合时触发逐 use mismatch，fan-out 合并触发 identity/分量错误。它降低这些故障的风险，不排除未测复杂图的所有错误。见 S17、analytic_reference_arrays.npz、mutation_results.json 和 tests/test_revision_v6.py。

### 答辩 8. H4 阴性时图组收益是什么

收益是指定分项的空间分布和可审计重构，包括 Q/K/V entropy 和 residual output–identity TV。它们不能由总图或单一 scalar 恢复。Head 排序增益未获支持，正文没有以结构收益替代该负结果。见 Table 4、Table 5 和 §8。

### 答辩 9. 等价重参数化改变分项意味着什么

分项是指定实现、切面和内部坐标的描述，不是仅由输入输出函数决定的唯一解释。§4.4 的可逆隐藏坐标变换反例保持总 G×I 而改变分项。故跨实现比较需先固定合同，而不能把分项称为唯一机制份额。

### 答辩 10. 删除内部标签后的核心贡献是什么

将当前点局部微分按具体 use 和原始输入位置共同组织，并把分支/全局重构目标、独立逐分量检查及结构错误检出落实到 ViT。核心用途是可复算的分项空间测量，非新的基础归因公式或普遍干预预测器。见引言两点方法贡献和 §4/§6。

## 最小剩余路线与结论条件

本轮科学稿无剩余实质性修改。提交前仍须完成最新文件的全部发布验证、prompt-to-artifact 审计与 Git/远端一致性检查；这些检查不能由本报告自行替代。若后续删除坐标例外/H4/原 all-family 负结论，或把 branch partition 恢复成完整目标 cut，当前判断即失效。若改为学校正式归档任务，须另提供模板及匿名政策要求。

在上述已明确的匿名稿范围和直接证据下，终审判断如下。

可送审
