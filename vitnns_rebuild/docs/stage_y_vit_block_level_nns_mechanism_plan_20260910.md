# Stage Y：ViT block-level NNS 机制研究规划

日期：2026-09-10

状态：`PLANNING_DRAFT_BEFORE_Y0_PROTOCOL_FREEZE`

建议阶段 ID：`STAGE_Y_VIT_BLOCK_LEVEL_NNS_MECHANISM`

本文件是新的科学规划，不是冻结协议、执行授权、结果报告或论文
Results。它不修改 Stage D 的 `GO_FIDELITY_SCOPE`、Stage E 的
`MIXED`、X2 的 `NO_GO_BROAD_CLAIM_DEVELOPMENT`、X2-MECH M2 的
`MIXED_OR_INCONCLUSIVE`，也不自动暂停当前
`F3_FINAL_FIGURE_PACKAGE_ONLY`。若 owner 决定执行 Stage Y，必须先在
Y0 签发前瞻性 route amendment，将 F3-Final 明确置为
`PAUSED_PENDING_STAGE_Y`。

## 1. 研究定位

Stage Y 不再尝试证明 NNS 普遍优于 Grad-CAM 或其他显著性方法。它要回答
的是一个更接近 NNS 原始研究动机、也更适合 ViT 架构的问题：

> ViT 把图像区域编码成 patch tokens 后，patch-level readout 是否掩盖了对类别决策有用的
> patch 内部路径特异性？NNS-style input-space complete-cut decomposition
> 能否恢复这种细粒度信息，并揭示它如何沿 Transformer 的深度、heads、
> tokens、attention/residual/MLP 分支以及受控扰动条件发生变化？

Stage Y 的预期论文角色是 **ViT 内部机制研究**，不是 saliency method
leaderboard。Grad-CAM 不进入 Stage Y 的研究矩阵，也不产生任何方法胜负
主张。raw attention 只可作为“注意力权重与 class-logit path contribution
是否等价”的结构参照，不能作为 attribution ground truth。

这里的“掩盖”有严格的操作性含义：常规 patch-level score 把一个原生 patch
内部的输入坐标聚合为一个数，而 NNS input-space decomposition 保留 patch 内
的 4x4-cell 或 pixel allocation。它不预设 patch embedding 在信息论意义上
压缩或不可逆地丢失信息。

## 2. 已有证据与尚缺证据

已有证据回答了以下问题：

1. Stage D：在冻结 ViT-B/16、`(block=0, head=9)`、head-direction
   Value-path 和 fine-4x4 endpoint 下，NNS 相对同目标 matched patch
   baseline 的 intervention fidelity 在独立 500 图 cohort 中为正
   `500/500`，median `delta_fid=2.3081996221714247`，paired
   image-bootstrap 95% CI
   `[2.1110806126093067, 2.445235411755993]`。
2. Stage E：四个预先冻结坐标显示该优势具有 scale 与 layer/head
   dependency，但这只是 post-D10 descriptive evidence。
3. F3-HeadAtlas：一个既有样本覆盖全部 `12 blocks x 12 heads`，但只能
   用作定性 atlas，不能形成 cohort-level 机制结论。
4. X1-R1：NNS-v2 已验证 input identity 和 complete-cut accounting；这是
   方法正确性证据，不是 cohort-level 机制证据。
5. X2/X2-MECH：否定 broad spatial-stability/superiority claim，并给出
   post-hoc multiscale mechanism evidence；它不回答 ViT 内部 block/head/
   token 组织。

尚缺的关键证据是：

- 在新 cohort 上同时覆盖全部 blocks 和 heads，而不是四个坐标或一个样本；
- 以 predicted-class raw logit 为统一 target 的 complete-cut component
  allocation；
- 输入 patch 到中间 token 再到 class logit 的定向贡献矩阵；
- attention、identity residual、MLP 和 post-attention residual 的分支分配；
- patch 内 4x4 信息相对 patch-uniform 表示是否有可干预验证的增量价值；
- 上述组织在 shallow/middle/deep 和受控扰动下如何变化。

## 3. 研究问题与拟检验主张

Stage Y 必须把“研究问题”和“结果主张”分开。以下内容在 Y0 冻结前都只
是待检验命题。

### Y-RQ1：patch 内路径特异性

ViT 的原生 16x16 patch-token summary 是否掩盖了同一 patch 内不同 4x4
cells 对 class-logit path components 的异质贡献？在保持 native-patch
顺序和目标不变时，NNS component map 提供的 patch 内排序是否比
patch-uniform/tie baseline 具有额外 intervention fidelity？

对应候选主张 `Y-CLM-INTRAPATCH`：

> NNS complete-cut component maps recover intervention-relevant within-patch
> heterogeneity that is absent from a patch-uniform representation.

只有预注册的 image-level paired endpoint 和 multiplicity gate 通过后才允许
使用 `support`；否则只能报告 `NOT_SUPPORTED` 或 `INCONCLUSIVE`。不能把
“图更细”直接写成“恢复了有用信息”。

### Y-RQ2：空间支持和集中度随深度如何变化

class-logit attribution 及其 attention/MLP/head components 的空间支持、
正负质量、集中度、有效支持范围和 cancellation 是否从浅层到深层发生系统
变化？

对应候选主张 `Y-CLM-DEPTH-SPATIAL`：

> Class-logit-directed NNS components exhibit a reproducible depth-dependent
> organization of spatial support and concentration.

该主张不预设“深层一定更语义”或“浅层一定更局部”。Y0 必须冻结 exact
two-sided depth contrasts；结果方向由数据决定。

### Y-RQ3：head contribution 是集中还是分散

每个 block 内，class-logit-directed attribution mass 和 head-ablation
response 是由少数 heads 集中承担，还是分布在多个 heads；effective head
count、top-1/top-3 share、HHI 和正负 cancellation 是否随深度变化？

对应候选主张 `Y-CLM-HEAD-ORGANIZATION`：

> The organization of class-logit-directed head contributions changes across
> Transformer depth, with the degree of concentration quantified over all
> heads rather than inferred from selected visual examples.

“承担类别决策”必须同时有 complete-cut allocation 和预注册 head ablation
control 支持。仅凭 attribution mass 时，只能写“分配/关联”，不能写因果
必要性。

### Y-RQ4：输入 patches 如何在计算图中混合

输入 source patches 对各 block 的 destination patch tokens 和 CLS token 的
class-logit path contribution 如何重分配？同 patch 保留、跨 patch mixing、
空间传输距离、destination entropy 和 CLS routing 是否随深度变化？

对应候选主张 `Y-CLM-PATCH-MIXING`：

> NNS complete-cut matrices reveal how input-patch contributions are
> redistributed across patch and CLS-token paths through Transformer depth.

这些矩阵是 signed、target-conditioned、path-integrated contribution，不能称
为 attention probability、信息流真值或因果 transition matrix。

### Y-RQ5：attention/residual/MLP component allocation

Y-RQ5 要研究的是：每个 block 在处理类别相关信息时，主要是在“保留已有
表示”，还是通过 attention 或 MLP “加入新的类别相关更新”，以及这种分工
如何从浅层变化到深层。

这需要分别分析两个 residual additions，而不是把三个模块混成一个比例：

1. 第一处 residual addition 比较 `已有表示的 identity path` 与
   `multi-head attention 新增的更新`；
2. 第二处 residual addition 比较 `attention 后继续保留的 residual path` 与
   `MLP 新增的更新`。

每组比较都同时测量 signed conductance、absolute attribution mass 和
cancellation，分别回答该分支总体促进还是抑制目标类别、参与程度有多大、
以及内部正负贡献是否发生强烈抵消。

对应候选主张 `Y-CLM-BRANCH-ALLOCATION`：

> Across Transformer depth, we characterize two complementary
> class-logit-directed allocations: the allocation between the pre-attention
> identity path and multi-head attention update at the first residual
> addition, and the allocation between the post-attention residual path and
> MLP update at the second residual addition.

如果结果显示可重复的深度差异，才允许进一步写“这两组分配及其正负抵消
结构随 Transformer 深度发生系统变化”；若没有清晰趋势，则必须报告层间与
图像间异质性，不得把 component allocation 解释成模块性能排序或唯一因果
责任。

### Y-RQ6：扰动如何调制上述组织

受控 geometric、photometric 和 noise conditions 是否改变 within-patch
heterogeneity、head concentration、patch mixing 和 branch allocation？

对应候选主张 `Y-CLM-PERTURBATION-MODULATION` 只允许作为 secondary
mechanism claim。它不设置 spatial-robustness PASS，也不重新打开 X2。

## 4. 固定研究范围

### 4.1 模型、target 与深度分组

- 模型：唯一使用 recovery-v2 冻结的 ViT-B/16。
- 输入：`224x224`，形成 `14x14=196` 个原生 patch tokens。
- blocks：`0..11` 全覆盖。
- heads：每个 block 的 `0..11` 全覆盖，共 144 个 block/head 坐标。
- 深度组在结果访问前固定为：
  - shallow：blocks `0..3`；
  - middle：blocks `4..7`；
  - deep：blocks `8..11`。
- primary target：clean predicted-class raw logit；同一图像的 clean class
  index 在所有扰动和 intervention 中固定。
- true-class raw logit 可作为预注册 secondary target；不得与 predicted-class
  rows 合并。runner-up target 不进入最小正式设计。
- 所有 correct 和 misclassified 样本原位保留；不得按预测正确性删样本。

Stage D 的 head-direction Value-path target 与 Stage Y 的 class-logit target
必须分表、分图、分措辞。Stage Y 可以把 `(0,9)` 作为事先存在的桥接坐标
单独展示，但不能把两个 target 的数值池化。

### 4.2 新 cohort

推荐正式 cohort 为 ImageNet-1k validation 中新的 100 张图：

- 与 C10、Stage D、Stage E、F3/F3H、F3C、X1 audit、X2 development 和
  X2-MECH 使用过的 source object IDs 完全不相交；
- 100 个不同的 true classes，每类一张；
- 先用 release metadata、class、object ID 和 file SHA-256 构造 selection
  digest 并冻结样本，再允许 decode、inference 或 attribution；
- 不按 clean correctness、logit、attribution、运行速度或视觉质量筛选；
- 不因数值失败补样；所有 unit 必须终止为 `VALID`、`FAILED` 或 `NA`。

从同一冻结 cohort 按 selection digest 预先指定：

- 10 张 full-array audit subset：保留完整 component pixel maps；
- 20 张 perturbation subset：执行全部冻结扰动；
- 其余样本仍保留 clean 条件的全量数值 summaries 和 matrices。

两个 subset 都不得在读取预测或 attribution 后更换。

### 4.3 扰动面板

推荐 Y0 冻结下列最小、可解释的 conditions：

1. `clean`；
2. `horizontal_flip`；
3. `translate_x_-8` 与 `translate_x_8`；
4. `rotate_-10` 与 `rotate_10`；
5. `brightness_0.9` 与 `brightness_1.1`；
6. `gaussian_noise_0.01_seed_0`。

这里的对称几何变换用于避免只选一个方向。所有 geometric maps 采用与
X2 相容的 forward/inverse coordinate contract，同时保留 full-domain
horizontal flip 作为无裁剪对照。若 Y1 预算试跑显示正式矩阵不可承受，只能
在访问正式 cohort outcome 前通过 resource-only prospective amendment 等比例
缩小 perturbation subset；不能根据初步科学结果删掉困难 transform。

## 5. NNS 计算对象

### 5.1 complete-cut 身份

Stage Y 复用 X1-R1 已验证的 NNS-v2 mathematical identity，而不发明新的
传播规则。对图像 `x`、baseline `x0`、固定 target `f` 和 complete cut
`C_b`：

```text
K_b[i,j] = (x_i - x0_i) * integral_0^1
             (partial f / partial h_b,j)
             (partial h_b,j / partial x_i) d alpha
```

每个 cut 必须重新满足 pixel reconstruction、total reconstruction、endpoint
parity 和 convergence contract。各 block 是同一 total attribution 的替代
decomposition，禁止把 12 个 block 的 maps 相加。

### 5.2 必须提取的 cuts/components

每张图至少覆盖：

1. post-patch-embedding token cut；
2. 每个 block 的 input-token cut；
3. 第一 residual addition：identity edge 与 12 个 native heads 经对应 output
   projection columns 后的 components；
4. attention branch aggregate 与 identity residual；
5. 第二 residual addition：post-attention residual 与 MLP output；
6. block output token cut；
7. final norm 与 classifier-input cut，用于端到端 closure。

Q/K/V、softmax attention 和 MLP hidden-unit views 可作为 audit-only drill-down，
但孤立 Q、K、V 或 attention tensor 不是 complete cut，不能单独命名为完整
class-logit decomposition。

### 5.3 输入 patch 到 destination token 的贡献矩阵

对 block `b` 的 token cut，把 input pixels 按 196 个 source patches 分组，
把 cut components 按 destination token 分组：

```text
T_b[p,q] = sum_{i in source_patch(p)} sum_{d in destination_patch_token(q)}
             K_b[i,(q,d)]

C_b[p]   = sum_{i in source_patch(p)} sum_{d in CLS_token}
             K_b[i,(CLS,d)]
```

其中 `T_b` 是 signed `196x196` patch-to-patch contribution matrix，`C_b`
是 source-patch-to-CLS vector。二者与对应 direct/constant accounting 一起
必须重建该 cut 的 total input-dependent logit delta。

第一 residual addition 还必须分别产生 identity matrix、12 个 per-head
matrices 和 attention-aggregate matrix，并验证 12 个 head matrices 之和等于
attention matrix；第二 residual addition 分别产生 post-attention-residual 与
MLP matrices。这样才能区分“上一层已经形成的路由”与“当前 block 的
attention/MLP component allocation”。这些 layer views 仍然是同一 total
attribution 的替代分解，不能解释成可逐层相加的 Markov transition。

Y1 必须用 factorized JVP/VJP 或等价的 exact grouped computation 证明该矩阵
可计算；禁止物化不可承受的 pixel-by-hidden-coordinate 全矩阵后再聚合。

### 5.4 head 与 branch allocation

每个 head 和 branch 同时保留：

- signed scalar conductance；
- signed 4x4-cell input-space map；
- positive、negative 和 absolute mass；
- L1-normalized allocation share；
- cancellation ratio。

head concentration 使用全部 12 heads 的 absolute map mass 计算 top-1 share、
top-3 share、HHI 和 entropy-derived effective head count。signed conductance
单独报告，不允许用绝对值结果掩盖正负抵消。

## 6. 指标与验证实验

### 6.1 patch 内信息恢复

对每个 native 16x16 patch 的 16 个 4x4 cells 计算：

- within-patch variance 和 normalized total variation；
- top-cell absolute-mass share；
- entropy/effective-cell count；
- positive/negative mass 与 cancellation；
- component-to-total map agreement。

“恢复有用信息”的 primary evidence 必须来自 paired intervention，而不能只
来自非零方差。推荐冻结 hierarchical within-patch intervention：

1. 两个方法使用相同的 native-patch order；
2. NNS 条件在每个 patch 内按 signed/positive component score 排列 4x4 cells；
3. patch-uniform control 在每个 patch 内使用 hash-seeded tie permutations，
   预先固定 32 个 seeds；
4. 两个条件使用相同 masking baseline、percentiles 和 class-logit target；
5. image-level endpoint 为 NNS 相对 tie-ensemble median 的 insertion/deletion
   或 ROAD AUC gain。

Y0 必须冻结 positive/negative endpoint、perturbation operator 和 primary
component family。不得在看到 144 个 heads 的结果后选择最好的 head 作为
primary。建议 primary 使用每个 block 的 aggregate attention-branch map；
per-head endpoint 作为完整 descriptive atlas。

### 6.2 空间支持与深度

按 image、block、component 输出：

- absolute-mass entropy 和 effective spatial support；
- Gini、top-1%、top-10% mass share；
- local connected support 或预注册邻域质量；
- source-patch spatial radius；
- positive/negative displacement 与 cancellation。

所有跨深度比较使用 raw scientific arrays 或同一数值定义，不能使用逐面板
归一化后的 display PNG。

### 6.3 head 组织与 intervention control

除 concentration metrics 外，对每张图、每个 block 执行 12 个 single-head
ablation controls：在冻结位置将该 head 的 pre-concat output 替换为 Y0
定义的 baseline，并记录固定 clean target raw-logit change。

必须检验：

- attribution allocation 与 held-out ablation effect 的 rank association；
- 按 attribution 预先排序后的 top-k 与 bottom-k ablation curves；
- head concentration 随 shallow/middle/deep 的 image-level paired contrast。

ablation 是辅助干预证据，不把单 head 删除等同于自然网络的唯一因果路径。

### 6.4 patch mixing

从每个 `T_b` 和 `C_b` 计算：

- diagonal/self-patch absolute-mass fraction；
- off-diagonal fraction；
- spatial mixing radius；
- destination entropy/effective destination count；
- source concentration 与 destination concentration；
- CLS-routing fraction；
- signed cancellation；
- matrix effective rank（作为 secondary descriptor）。

同一 block 的 raw attention matrix可计算同名结构量，并与 `T_b` 做描述性
association，用于回答“attention weights 是否足以代表 class-logit-directed
contribution”。不得把二者差异直接解释为 attention 错误。

### 6.5 branch allocation

围绕“保留已有表示还是加入新更新”这个问题，对两个 residual additions 分别
报告：

- 第一处：pre-attention identity path vs multi-head attention update；
- 第二处：post-attention residual path vs MLP update；
- signed conductance、L1-normalized absolute mass share、cancellation；
- component sum 对 total map 的 reconstruction residual；
- shallow/middle/deep paired contrasts。

这两组是各自 complete cut 内的分配，不得把 identity、attention 和 MLP
直接合并成一个三项百分比，也不得把较高 attribution share 直接称为较高
分类性能。

### 6.6 perturbation modulation

以 clean 为同图参照，量化：

- within-patch metrics 的变化；
- head share vector 的 rank/cosine agreement；
- effective head count 的变化；
- patch-mixing matrix 经坐标回映后的 agreement；
- attention/residual/MLP allocation 的变化。

geometric conditions 必须报告 valid mask 与 full-domain flip control；所有 clean
prediction flips 原位记录，不删行。该模块只回答机制组织是否被调制，不给出
“鲁棒/不鲁棒”二元认证。

## 7. 统计设计

1. 独立统计单位始终是 image；12 blocks、144 heads、196 patches 和多个
   transforms 都是同图重复测量，不能伪装成独立 `n`。
2. primary summaries 先在 image 内按冻结规则聚合，再进行 cohort inference。
3. shallow/middle/deep 使用 blocks `0..3`、`4..7`、`8..11` 的固定聚合；
   同时报出 12-block trajectory，不能只显示显著的 depth pair。
4. 置信区间使用 image-level paired bootstrap；同一 bootstrap replicate 内必须
   保留一张图的全部 blocks/heads/conditions。
5. 建议冻结两个 confirmatory families：
   - Family A：within-patch refinement gain 的 primary endpoints；
   - Family B：early-vs-late spatial support、effective head count、off-diagonal
     mixing 和 branch allocation 四个 depth contrasts。
6. 每个 family 使用 Holm correction；exact hypotheses、directions、alpha、
   bootstrap replicates、effect-size threshold 和 missing-data rule 必须在 Y0
   outcome access 前写入 machine-readable registry。
7. 144-head、196x196 matrix、middle-depth trajectory、correct/misclassified 和
   perturbation interaction 默认是 estimation/descriptive analyses；若未在 Y0
   注册，不得事后升级为 confirmatory discovery。
8. 每项 claim 独立得到 `SUPPORTED`、`NOT_SUPPORTED`、`INCONCLUSIVE` 或
   `INVALID`；不得用一个 omnibus GO 覆盖所有机制问题。

## 8. 数据保留与展示

为兼顾审计性与存储成本，采用分层保留：

- 全部 100 图：保存 per-image/per-block/per-head scalar summaries、signed
  4x4-cell component maps、aggregate/branch/per-head `196x196`
  patch-mixing matrices、CLS vectors、branch/head allocation 和 terminal
  records；
- 固定 10 图 audit subset：保存完整 signed component maps、positive/negative
  maps、raw attention controls 和 reconstruction arrays；
- 固定 20 图 perturbation subset：保存全部 condition summaries 和足以重新
  验证 inverse-map/metric 的 numeric arrays；
- scientific arrays 与 display arrays 分根保存；PNG/SVG 不作为科学输入；
- 所有 files、arrays、manifests、terminal rows 和 aggregate tables 均生成
  SHA-256、canonical digest 和 whole-root inventory。

展示必须包括：

- 全 12-block trajectories；
- 全 12-head distribution 或完整 atlas 的索引入口；
- 不得只展示 best heads、best patches 或最漂亮的样本；
- 若版面只能容纳少量案例，使用冻结的 audit subset 与固定坐标，完整结果放
  Supplement/interactive data package；
- 每张图明确标注 `class-logit complete-cut contribution`，不得标成 raw
  attention 或 Stage D Value-path attribution。

## 9. 执行阶段

### Y0：route、protocol、claims、cohort 与预算框架冻结

任务 token：`STAGE_Y0_PROTOCOL_AND_BUDGET_FREEZE_ONLY`

1. 签发 route amendment：`F3_FINAL_PAUSED_PENDING_STAGE_Y`。
2. 把本规划转成冻结协议和 machine-readable claim/metric/schema registries。
3. 冻结 cohort selection/exclusion algorithm、audit/perturbation subset rules、
   target、cuts、transforms、统计 families、failure semantics 和 output roots。
4. 定义 Y1 技术 pilot 和预算外推公式，但不得读取正式 Y cohort outcomes。
5. 实现 CPU-only schema validator、access ledger 和 false-pass tests。
6. Y0 PASS 只授权 Y1，不授权 formal cohort inference。

### Y1：实现验证与小规模成本 pilot

任务 token：`STAGE_Y1_IMPLEMENTATION_VALIDATION_AND_COST_PILOT_ONLY`

1. 在 analytic/synthetic fixtures 验证 complete-cut identities、head/branch
   sum、patch-matrix conservation、sign 和 aggregation。
2. 在预先允许的既有 audit images 上验证 native-forward parity、重复性、
   convergence 和 12x12 coverage。
3. 验证 grouped JVP/VJP 实现不物化不可承受的完整 Jacobian。
4. 对 clean 和至少一个 perturbation 运行 cost pilot，记录 H20-GPU-hours、
   peak memory、calls、bytes 和 wall time。
5. 按冻结公式计算 Y2/Y3 预算与硬上限。超出上限时必须在正式 cohort
   outcome access 前停下，签发 resource-only prospective amendment；不得用
   pilot 科学结果缩小研究范围。
6. Y1 PASS 只授权 cohort sealing 和 Y2 clean production。

### Y2：新 cohort 封存与 clean formal production

任务 token：`STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY`

1. metadata-only 生成 100 图 registry、exclusion proof 和 selection digests。
2. 冻结 audit/perturbation subsets 后才允许 decode 和 model access。
3. 对全部 100 图生产 clean class-logit complete-cut evidence、144-head
   allocation、12 个 patch-mixing matrices、branch summaries 和 intervention
   endpoints。
4. 每个 image/block/head 写 atomic typed terminal；失败不得替换。
5. source-recomputing reconcile 后封存 Y2；此阶段不做跨图结果选择或论文
   解释。

### Y3：冻结扰动下的 formal production

任务 token：`STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY`

1. 只对预先冻结的 20 图 subset 和全部冻结 transforms 执行。
2. clean target class 固定；保留 prediction flip、valid mask 和 inverse-map
   provenance。
3. 生产与 Y2 同 schema 的 head/patch/branch summaries 和必要 numeric arrays。
4. 不计算或公布 broad stability PASS，不修改 X2 terminal。
5. source validation、whole-root sealing 和 resource accounting 通过后结束。

### Y4：CPU-only 机制分析与 claim decision

任务 token：`STAGE_Y4_CPU_ONLY_MECHANISM_ANALYSIS_AND_DECISION`

1. 不读取 JPEG、不加载模型、不访问 CUDA/GPU。
2. 从 Y2/Y3 sealed numeric evidence 重算全部 primary/secondary metrics。
3. 以 image 为单位完成 paired bootstrap 和 preregistered multiplicity control。
4. 输出 depth trajectories、all-head concentration、patch mixing、branch
   allocation、within-patch intervention 和 perturbation modulation。
5. 对六项 claims 分别签发 `SUPPORTED/NOT_SUPPORTED/INCONCLUSIVE/INVALID`。
6. false-pass audit 必须主动拒绝：best-head selection、pseudo-replication、
   Value-path/class-logit 混用、raw-attention 等同 contribution、failure
   omission、display-array 回流和 post-hoc confirmatory promotion。

### Y5：results-to-paper handoff 与路线恢复

任务 token：`STAGE_Y5_RESULTS_TO_PAPER_HANDOFF_ONLY`

1. 新建不可覆盖的 Stage Y results-to-paper handoff，绑定 refs、commits、
   inventories、terminal decisions 和允许/禁止措辞。
2. 预先列出的六项 claim 逐项进入正文、Discussion、Supplement 或不进入
   论文；负结果不得隐藏。
3. 更新 Stage F/F3-Final composition，但不得重算 Stage Y 科学结果。
4. 恢复路线：`F3_FINAL_FIGURE_PACKAGE_ONLY -> STAGE_G_WRITING_G0_ONLY`。
5. G1 才正式冻结合并后的 scientific story 和 Evidence Matrix。

## 10. 工程与科学验收边界

每个阶段的 engineering PASS 与 scientific claim outcome 必须分开。

最低工程验收包括：

- checkpoint、preprocessing、model state、target 和 baseline identity 全部绑定；
- native forward parity 与 endpoint parity 通过；
- complete-cut pixel/total residual 通过冻结阈值；
- head sums、branch sums、patch-matrix sums 和 scalar/map sums 相互闭合；
- 100 图 registry 和所有 terminal denominators 完整；
- 重复运行 digest 一致；
- CPU/GPU 边界、预算、wall time、memory 和 call counts 有正式记录；
- Git commit、tracking ref、remote ref、archive ref 和 LFS 状态一致。

以下情况必须 fail closed：

- 正式结果访问后改变 cohort、depth groups、primary components 或 transforms；
- 因某 head/convergence 失败而静默删行或换样本；
- 从 144 heads 中挑最好结果作为预注册 primary；
- 把 single-sample atlas 当 cohort evidence；
- 把 raw attention、display heatmap 或 Grad-CAM 当成 complete-cut component；
- 将各 layer 的替代 decompositions 相加；
- 把 class-logit attribution completeness 写成语义上的“完整解释”；
- 把 perturbation modulation 写成普遍空间鲁棒性；
- 用 Stage Y 覆盖 Stage D/E/X2/X2-MECH 的历史结论。

## 11. 结果分支与允许措辞

Stage Y 不预设结果必须正向。Y4 至少准备以下结果分支：

1. **within-patch gain supported**：可以说 patch-token summary 忽略了
   intervention-relevant within-patch heterogeneity，但必须限定在模型、cohort、
   target、cuts 和 endpoint。
2. **heterogeneity visible but intervention gain absent**：只能说 NNS 恢复了
   数值结构，不能说该结构提高 fidelity。
3. **head allocation concentrated**：可以量化少数 heads 承担较高
   class-logit-directed attribution/ablation response；不能写成唯一因果 heads。
4. **head allocation distributed**：将“分布式组织”作为机制发现，不把它写成
   方法失败。
5. **mixing changes with depth**：允许描述预注册指标的方向和 effect sizes；
   不用“信息流真值”措辞。
6. **no reproducible depth trend**：报告稳定的个体差异或无趋势，不从 atlas
   中事后挑故事。
7. **perturbation-dependent reorganization**：只作为 bounded secondary
   mechanism result；X2 broad NO-GO 保持不变。

## 12. 可能形成的论文主线

若相应 claims 获得支持，Stage Y 可把当前论文主线扩展为：

> NNS 在 ViT 中不仅提供输入坐标级的数值完备分解，还能把最终 class-logit
> 归因解析为跨深度、head、token 和 residual/attention/MLP 分支的可审计
> components。其 patch 内细粒度信息是否具有干预价值、head 组织是集中还是
> 分散、patch contribution 如何随深度混合，均由新的 cohort-level evidence
> 决定。该框架用于理解 Transformer 内部机制，不主张普遍取代区域显著性
> 方法。

最终措辞必须由 Y4 决策和 Y5 handoff 决定。当前规划阶段只能提出问题，
不能提前写成已证明结论。

## 13. 预期产物与目录

建议后续建立：

```text
docs/stage_y_vit_block_level_nns_mechanism_protocol_20260910.md
docs/stage_y_results_to_paper_handoff_20260910.md
config/stage_y/
manifests/stage_y/
evidence/stage_y/y0_protocol_20260910/
evidence/stage_y/y1_validation_pilot_20260910/
evidence/stage_y/y2_clean_formal_20260910/
evidence/stage_y/y3_perturbation_formal_20260910/
evidence/stage_y/y4_mechanism_analysis_20260910/
reports/stage_y_y0_20260910/
reports/stage_y_y1_20260910/
reports/stage_y_y2_20260910/
reports/stage_y_y3_20260910/
reports/stage_y_y4_20260910/
```

所有日期型正式 roots 在首次创建后不可复用；BLOCKED/FAIL 后只能用新的
prospective amendment 和新 root。Y0 应冻结最终路径、schemas、archive refs
和 completion tokens。

## 14. 推荐实际顺序

```text
当前 M2 results-to-paper handoff（已完成）
-> owner 决定插入 Stage Y
-> Y0 route/protocol/budget freeze
-> Y1 implementation validation + cost pilot
-> Y2 new-cohort clean evidence production
-> Y3 perturbation mechanism production
-> Y4 CPU-only analysis + per-claim decisions
-> Y5 results-to-paper handoff
-> updated F3-Final
-> Stage G / G0-G7
```

下一步不是直接启动 100 图 GPU workload，而是执行
`STAGE_Y0_PROTOCOL_AND_BUDGET_FREEZE_ONLY`。Y0 应把本规划中仍是“建议”的
内容变成不可随结果改变的正式合同，并先解决计算可行性、预算和失败语义。
