# NNS 在 ViT 中的 Patch 与深度研究：接下来执行步骤

日期：2026-09-14
状态：`PROPOSED_NOT_FROZEN`；仅定义后续工作，不启动实验。
研究问题、公式和结论边界以[研究目标](nns_vit_patch_depth_research_goals_20260914.md)为准。

## 1. 当前授权与路线

本次任务只生成两份规划文档。没有签发新实验授权，没有修改 Stage Y 的协议、证据、报告或路由，也没有开始 Y5、F3-Final 或 Stage G。

建议的新研究命名空间为 `NNS_VIT_PATCH_DEPTH_SEMANTICS`，下文以 S0–S6 表示它自己的工作包，不复用 Y4 recovery 编号。

```text
S0 数据与证据盘点，确定可行研究范围
 → S1 隔离开发与 pilot，验证测量和成本
 → S2 正式协议、cohort、probe 方案与预算冻结
 → S3 冻结 backbone 的表示提取及 probe 训练
 → S4 正式 NNS 与匹配遮挡计算
 → S5 CPU-only 统计、独立重算与逐项决策
 → S6 图表、结论及后续交接
```

每阶段开始前验证上一阶段终态；未通过则停留在该阶段。阶段完成不自动授权下一阶段，尤其不能从文档编写直接进入 GPU 生产。

## 2. S0：先确认有没有条件回答问题

### 输入

- [Y4-R5 科学报告](../reports/stage_y_y4_r5_20260914/report.md)。
- [Stage Y component 定义](../manifests/stage_y/component_cut_registry.json)、[Y4-R3 映射](stage_y_y4_r3_attention_cut_recovery_20260914.md)、[Y4-R4 零质量规则](stage_y_y4_r4_zero_mass_na_recovery_20260914.md)。
- [模型与环境绑定](../config/recovery_v2_binding.json)。
- 已有 cohort manifests、source-object 身份、原始资产清单及新标注数据的元数据。

### 操作

1. 记录工作区 HEAD、代码身份和现有协议状态；对实际继承的文件重算 SHA-256／canonical digest，不把报告中的抄录值当成验证结果。
2. 盘点是否保留原生 patch/token/branch 所需数组。区分“磁盘上可读取”“协议允许继承”“target 与新问题相同”，三者不能互相替代。
3. 查明可用的目标分割标注、图像来源、类别映射和使用许可。数据集名称、路径和样本数以实际盘点为准；本文件不声称本机已经有语义 mask。
4. 将候选样本按 source image 分组，与 Stage C/D/E/F/X/Y 已用样本及本研究开发集去重；同源不同文件名、裁剪版本和重复标注不能进入不同 splits。
5. 写出图像预处理与 annotation mask 的共同坐标变换、同类多实例合并和多类别 target 选择规则。
6. 单独检查 CLS-only 读出、block 11 的零 patch-destination contribution、source-patch 聚合是否包含 CLS 路径。

### 数据选择分支

| 盘点结果 | 下一步 |
|---|---|
| 具备类别对应的目标分割 mask | 执行目标／前景最小设计 |
| 另有可靠部件 mask | 可另提部件扩展；不自动加入主 family |
| 只有 bounding box | 只能研究框级定位，先修订研究范围；不能把框内部全部当作目标或部件真值 |
| 只有图像级类别 | 不能完成主设计；先获取或独立制作标注，不能用 NNS/attention 热图伪造 ground truth |
| 标注类别不能映射到冻结分类头 | 停止类别 logit 主研究或前瞻性改题；不能临时用相关类别替代 |

人工标注者应看原图和标注规范，而不看 NNS、probe 或遮挡效果；若需要标注质量抽检，应预先划出独立 QC 样本并记录所有排除。

### 交付和准入

建议交付：`asset_inventory.json`、`annotation_contract.json`、`reuse_and_exclusion_manifest.json`、`feasibility_decision.json`，均属于未来 S0，不在本次任务中生成。

准入条件：至少目标级任务具备可靠标签、明确类别映射和独立验证池；否则记为 `BLOCKED_DATA` 或 `BLOCKED_DEFINITION`。**S0 不解码未授权图像、不推理、不训练、不运行 attribution；只作元数据与源码审计。**

## 3. S1：用隔离开发集验证定义，而不是试着找到显著结果

### 开发集隔离

开发样本可以包含已有 Stage Y 样本，但必须标为 `EXPLORATORY_ONLY`；如果计算标注类别 target，需要重新计算对应数组，不能将 predicted-class attribution 改个名字后继承。

开发集、probe train、probe validation、最终 confirmatory test 在 source-image 级别不相交。任何被查看热图、选 endpoint 或调匹配规则的样本，永久归入开发用途。

### 方法与实现检查

1. 实现 image/mask 同步 resize/crop 和 patch 面积比例；数值分析在原生 14×14 网格，插值热图只用于显示。
2. 保留所有 12 blocks 的 patch 表示、aggregate attention contribution，以及单独记录的 residual components 和 CLS 分量。
3. 校验 `a[b,p] = sum_q T[b,p,q] + C[b,p]`、signed→patch sum→positive 顺序、完整 cut 重构和 baseline accounting；不跨 block 相加冒充单一完整分解。
4. 验证完整 input attribution 的同 target 对照，以及 hook 前后 logits 一致性。数值误差和零质量要有单独状态，不能通过归一化隐藏。
5. 建立 P1 的位置分层 score permutations；先在模拟居中物体、常数 score、纯位置 score 等例子上验证，不会把位置先验误当语义增益。
6. 建立 P2 的等 patch 数、位置分层匹配遮挡生成器。冻结位置分箱、形状约束、允许误差、控制数量和 seeds；若要求不能满足，应返回 typed NA，不能静默退化成普通随机遮挡。
7. 遮挡对照可以与 top-k 有冻结规则允许的重叠，不能为强制不重叠而只留下异常区域；记录重叠率和匹配误差，规则不使用目标 logit 选择控制。
8. pilot 比较候选填充方式的有效性与图像伪影，正式时选定一个 primary；其他方式若保留，只按预先写明的 sensitivity 规则报告，不事后选更显著的算子。
9. 实现固定容量、类别条件的 linear ridge occupancy probe 和同类别条件的位置／低级外观对照。每个目标类别必须有独立 train/validation/test 图像；patch 相关性、类别不均衡和样本权重都进入训练合同，不能沿用“每类只有一张图”的配置训练 probe。
10. 对 label-shuffle、constant-feature、constant-score、zero-positive-mass、缺少控制、裁剪后无目标、预测错误、最后层 patch-gradient 为零等情形执行单元与失败路径测试。

### 成本与统计可行性

- 使用少量、预先登记的 pilot 样本测量 feature extraction、NNS integration、control forward、probe training、CPU reduction 的时间、显存、存储和有效率。
- 先解决终端拓扑与分母退化，再扩大样本量；结构性 n=0 不能靠多算图像解决。
- 根据实质效应阈值、pilot 方差和保守精度／功效模拟制定样本量；阈值须有任务意义依据，不得设在 pilot estimate 下方以便通过。
- 记录同一张图不同层、不同遮挡预算的相关性，按图像级端点估算功效；不能把 196 个 patches 当作 196 个独立样本。
- 定义特征保存策略和足以复核的原始分解数据；如完整数组太大，应在 formal 前明确 audit subset、流式聚合和可独立重算的层级，不能事后声称未保留的数据已经全量复核。

### 准入

P1/P2/D1/D2 全部可定义，关键失败路径已测试，成本和有效样本需求可承受，才能进入 S2。

若需更换 primary branch、target、depth grouping 或 endpoint，必须在 S1 的开发记录中说明并更新两份规划；验证集保持未读。不能把 S1 的正结果当正式证据。

## 4. S2：一次性冻结正式问题和测试集

### 4.1 必须补齐的参数

下表未完成前，状态保持 `PROPOSED_NOT_FROZEN`，不得启动正式生产。

| 字段 | 本轮建议／必须确认内容 | 冻结依据 |
|---|---|---|
| 数据与版本 | `[DATASET_REQUIRED]`、图像和 mask 身份、许可、source 去重规则 | S0 资产审计 |
| 样本量与组成 | `[N_REQUIRED]`、类别分层、各 split 数量和固定 class weights | 精度／功效及预算 |
| primary target | 标注目标类别 raw logit；冻结映射和多目标规则 | 语义可解释性 |
| primary component | 第一 residual 的 all-head aggregate attention，含 CLS source contribution | 拓扑和重构检查 |
| 空间和层组 | 16×16 patches；0–3、4–7、8–11；deep-minus-shallow | 研究目标 |
| P1 对照 | `[POSITION_STRATA_REQUIRED]`、`[PERMUTATION_COUNT_REQUIRED]`、seeds | 位置偏差和有效性测试 |
| P2 操作 | `[PATCH_BUDGETS_REQUIRED]`、`[MASK_OPERATOR_REQUIRED]`、对照数量、形状／位置约束及失败规则 | 匹配可行性 |
| probe 配置 | 固定容量、ridge 搜索网格、train-only 标准化、validation 选择规则、low-level features | 开发合同 |
| 实质效应 | `delta_P1`、`delta_P2`、`delta_D1`、`delta_D2` 均待登记 | 独立科学意义与测量分辨率 |
| 统计方案 | 图像均值、配对 CI、非零阈值检验算法、Holm family、replicate 数、seeds | 预注册 SAP |
| 缺失与分母 | 每 endpoint 最小有效图像数、最大 NA 比例、层／预算完整性规则 | estimand 与功效 |
| 算力与资源 | GPU-hours、wall-hours、CPU-hours、峰值显存、存储、重试上限 | pilot 投影 |
| 展示与复核 | 固定展示抽样、audit subset、保留数组、独立 validator 范围 | 可审计性 |

### 4.2 统计与决策约束

1. 主 family 只有 P1、P2、D1、D2；Holm 控制 family-wise alpha=0.05。P1/P2 为正向，D1/D2 为双侧实质差异。
2. 检验应围绕实质效应阈值定义，而不是仅做零效应检验再称“超过门槛”；双侧 D1/D2 的方向选择也计入检验定义。
3. 以 source image 为配对重采样单位，所有 layers、patch budgets 和 controls 随图像一同重采样。若按类别分层抽样，保持冻结 strata 与权重；不把测试结果外推为新类别泛化。
4. probe test CI 默认条件于已冻结的训练数据、权重和超参数；若要包含训练随机性／训练集抽样不确定性，须预先规定重复训练或嵌套过程及预算。
5. CI、阈值检验与 Holm 的具体组合需先用模拟数据检验错误率和覆盖率，再冻结；不能沿用某个统计代码名字就假定检验正确。
6. 正质量为零、缺少匹配对照、缺失必要 block 等按冻结规则传播 NA；不得填零、补图、只保留好看的 layers 或只选预测正确样本。
7. 达不到有效分母门槛的 endpoint 为 `INVALID`，仍保留在四端点 family 中且不拒绝。仅对有效配对样本的估计必须同时报告全 cohort 的终态分布，不能把估计对象写成无缺失总体。
8. 只预注册 P1+P2 的联合支持及 D1/D2 同方向互证规则；无六项全部成立之类的 omnibus PASS，也不强迫两条主线同时成功。

### 4.3 封存与授权

- 冻结数据 splits、target mapping、模型 identity、代码 commit、全部 registries、预算和展示选择规则。
- 使用 metadata-only 选择与 source-byte hashing；在封存前不查看 formal test 图像、feature、logit、map 或 endpoint。
- 创建新命名空间的 route amendment 和前瞻性执行授权，明确与现有 `STAGE_Y5_RESULTS_TO_PAPER_HANDOFF_ONLY` 的关系；新路由不能靠这份规划隐式覆盖旧路由。
- 使用 closed schemas；校验字段、引用、digest 和 claim truth tables。终态根目录不可原地修补，修复走新版本并保留前驱证据。

建议未来产物：`protocol.json`、`cohort_manifest.json`、`annotation_contract.json`、`metric_registry.json`、`statistical_analysis_plan.json`、`budget_receipt.json`、`protocol_freeze_receipt.json`。

## 5. S3：表示提取与 linear probe

### 操作

1. 通过启动检查后，按冻结模型、预处理和 manifest 提取所有 blocks 的 patch features；feature extraction 使用 eval 模式，backbone 不训练。
2. 分 split 存放 features、patch occupancy labels 和 source IDs。train/validation/test 不允许按 patch 随机混分。
3. 用 train 拟合标准化和 ridge probe；仅用 validation 选择预先规定网格中的超参数。所有层使用相同训练协议、容量和调参预算。
4. 训练位置／低级外观对照；保存全部权重、数据绑定与训练日志。不要把额外语义网络作为低级对照。
5. 封存 probe 权重和选择记录后才计算 test 预测；将 per-image/per-block 误差封存，不在 S3 查看深浅层结论后改变 S4。
6. 单独报告预测错误样本和类别构成，但不因此排除它们；同一图像的真实目标 label 不随模型预测改变。

### 准入

features 覆盖合同通过、split 无泄漏、probe 权重已冻结、所有 test 样本有明确终态。probe 效果不好不是偷偷调参的理由；只要测量有效，就按冻结方案继续保留负结果。

建议未来产物：`feature_inventory.json`、`probe_training_receipt.json`、`probe_weights`、`probe_test_predictions`、`representation_completion.json`。

## 6. S4：正式 NNS 与匹配遮挡生产

### 启动与资源检查

1. 验证 S2/S3 终态、代码 ancestry、全部输入 digest 和正确的 producer identity。
2. 记录 Python 可执行文件 realpath、依赖、GPU 身份、确定性变量；GPU probe 包含矩阵乘法和 backward，不仅做 elementwise 操作。
3. 若启用确定性 CUDA，先固定 `CUBLAS_WORKSPACE_CONFIG`；使用资产绑定所指定环境，不凭路径名称猜当前配置正确。
4. 仅使用已授权且实际空闲的设备；设备数、sharding、资源 cap 和重试规则以 S2 为准，不继承其他阶段的预算 waiver。
5. 每个 source/component/operator unit 采用 pending→terminal 记录，区分 numerical failure、infrastructure failure 和科学 endpoint NA。
6. 验证剩余预算、输出空间和独立复核预算足够；生产和 validator 分别计费并汇总，不把 worker 总 GPU-hours 除以卡数。

### 操作

- 对每张 formal 图像执行固定标注类别 target 的 NNS 分解，保留 12 层 primary component 和冻结 secondary 数据。
- 校验 input completeness、pixel/patch 重构、matrix+CLS accounting、head sum 和 residual pair；不要用显示归一化后的图做数值分析。
- 从冻结 patch scores 生成 P1 permutations 和 P2 top-k/matched controls，记录每个 mask、seed、匹配误差及目标 logit。
- 按冻结方案计算完整 input attribution 对照及遮挡 sensitivity；不扩展未注册 heads、layers、target 或 operator。
- 保存足以重算 P1/P2/D1 的原始 scores、labels、control draws 和 logits；无需在结果出现后再求出一套更有利的排序。
- 只监控完成率、数值合同和资源使用，不按显著性提前停止或追加样本。

### 准入

所有预定样本和单位均有终态；源数组、mask、control 和资源账本可关联并核对。有效性不足时允许完成流程封存，但相应科学端点进入 `INVALID`，不能为了 workflow PASS 删除失败记录。

## 7. S5：CPU-only 分析、独立重算与决策

### 操作

1. 只从已封存的数值数组、标签、预测和 logs 读取；不解码 JPEG、不加载 ViT、不调用 GPU/CUDA，也不新增干预。
2. 独立验证 source hashes、geometry contract、patch scores、CLS inclusion、零质量及终态分母；从原始数值表重算 P1/P2/D1/D2。
3. 按固定 image-level 聚合、CI、阈值检验和 Holm 得到四个端点的 `SUPPORTED`、`NOT_SUPPORTED`、`INCONCLUSIVE` 或 `INVALID` 状态，遵循研究目标中的决策定义；INVALID 项不从 family 中消失。
4. 输出全 12 层轨迹和冻结 supplementary analyses，明确标记 descriptive，不从中回选新的 confirmatory contrast。
5. 比较 P1/P2 联合支持、D1/D2 的方向关系；不把“probe 显著”替代“NNS 显著”，也不把“深度对齐变化”写成“branch 导致了语义形成”。
6. producer 与 independent validator 分开重算端点和决策，不以读 producer 的 PASS 字段作为验证。

### 必须包含的结果表

| 表 | 必要列 |
|---|---|
| endpoint decisions | endpoint、estimand、n_total/n_valid/n_NA/n_failed、estimate、CI、threshold、adjusted p、state |
| depth profiles | image、block、component、target、raw alignment、control-corrected alignment、probe error/gain |
| intervention controls | image、block、budget、operator、selected/control IDs、matching diagnostics、raw logit change |
| validity accounting | image、unit、terminal state、reason、zero-mass/geometry/reconstruction diagnostics |
| claim ledger | claim、required endpoints、observed states、允许措辞、证据路径 |

验收要求：原始数值记录可以追到每个汇总值；workflow integrity 与 scientific outcome 分开。无显著结果时仍交付完整报告，不以换 family、换分母或换方向“救结论”。

## 8. S6：可解释的图表与下一步交接

建议只形成以下核心图表：

1. **Image #1：Patch 证据链。** 原图与真值、同 target 的 NNS primary component、匹配 control masks、遮挡响应；展示选择规则先冻结。
2. **Image #2：全 cohort Patch 结果。** P1/P2 效应与 CI、有效率和失败类型；不是只展示正样本。
3. **Image #3：深度曲线。** 全 12 层 NNS 对齐与 probe 性能分面绘制，附 D1/D2；不同指标不用同一纵轴暗示数值可比。
4. **Image #4：跨层 atlas。** 相同样本和坐标的 shallow/middle/deep component maps、CLS routing 和 annotation；保持预定义色标，不靠逐面板亮度制造深度趋势。

生成图片应与简短的逐图解释一起交付；解释按“主张—证据—边界”组织。保存展示样本 manifest 和图像 SHA-256，负例及不一致例也用固定规则选取。

若 D1/D2 方向一致，下一步可以讨论部件扩展；若不一致，先分析表示与输出贡献为何分离。任一后续分析只要受本轮结果启发，就属于探索，新的正式主张另用隔离验证数据。

## 9. 建议的未来产物位置

以下只是命名建议，本次没有创建这些目录或文件：

```text
manifests/nns_vit_patch_depth/             # 冻结数据、指标、控制及选择规则
config/nns_vit_patch_depth/                # run specs、资源及环境合同
evidence/nns_vit_patch_depth/<stage_root>/ # 权威数值、provenance 和终态
reports/nns_vit_patch_depth_<run_id>/      # 可浏览结果、图像及逐图解释
```

代码和 metadata-only validator 优先复用现有模块，新增的 validator 尽量采用 Python 标准库；任何新实现均需在具体阶段获批后进行。本次不调整 Git 发布策略；提交和 push 按当前轮次的明确授权执行。

## 10. 下一条建议执行请求

> 执行 NNS_VIT_PATCH_DEPTH_SEMANTICS 的 S0，仅做 CPU metadata-only 数据与方法可行性盘点：核对 Y4 封存结论、继承资产、目标／部件标注、类别映射、历史 source 去重和 probe/test 分割可行性，输出可行范围与缺失项。不解码图像、不加载模型、不使用 GPU、不修改 Y4，不开始 S1 或正式实验。若目标级标注不可用，报告 BLOCKED_DATA 并列出需要补充的数据。

当前应先执行这一步，而不是申请完整 GPU 重跑；只有 S0 明确可行、S1 测量有效且 S2 完成冻结，才有条件讨论新的正式支持结论。
