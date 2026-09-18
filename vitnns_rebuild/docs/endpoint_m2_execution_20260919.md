# M2 补充冻结执行协议

依据《阶段M_endpointnnsvit.md》§3.1–3.8、§11.5、§13必做3与§14，执行 RQ4 跨层全 head 扩展，止于 M2。状态为 supplementary frozen study，原 C4a/C4b omnibus 均保持 NOT_SUPPORTED。M1 已重新验证全部封存文件、800任务与2688检验。

两模型使用原共享400图、ground-truth logit、0-based blocks 2/5/10；ViT-B/16每层12 heads，DeiT-S每层6 heads。共2400 image-model-block任务、21600 image-model-block-head条件。normalized-zero与原raw-RGB blur（kernel31、sigma10）保持不变。图像、目标、模型、预处理从M1原始任务表绑定继承，不按结果选择。

## 区域、预测量及效应

主要空间单元为14×14网格的16×16 patch，预算为1、10、20、49 patch；ceil(196×比例)给出实际面积1/196、10/196、20/196、49/196。原E5实际为4×4网格、每格56×56像素，并非4×4像素区域；保留该几何的1/2/4格曲线作补充，面积为6.25%/12.5%/25%。采用非连通top-K集合，K=1覆盖单patch；2×2 square、连通搜索及固定形状面积搜索是§3.5可选比较，本轮不新增。

对指定head的clean cotangent g_h=∇_{z_h}F，E_h=x⊙J_{z_h,x}ᵀg_h。所有head图均包含真实softmax Jacobian。同一head/region的八种规则为：

1. Endpoint absolute：Σ_{i∈R}|E_{h,i}|。
2. Endpoint signed：|Σ_{i∈R}E_{h,i}|；同时保留未取绝对值的signed sum。outcome是效应大小，因此采用signed aggregate的绝对值排序，与§7.4一致。
3. Regional Taylor：|g_hᵀ(z_h(x)−z_h(x_R))|，x_R是单候选区域替换后的输入。
4. Gradient-only：Σ_{i∈R}|(J_{z_h,x}ᵀg_h)_i|，保持同一个head。
5. Activation displacement：||z_h(x)−z_h(x_R)||₂。
6. Random：每image/model/block/head/reference/geometry固定hash种子，20个均匀随机排列；各排列独立计算absolute effect/AUC后平均，不先平均signed effect。
7. Center：按候选区域中心至图像中心距离排序。
8. Attention：指定head的clean CLS-to-patch softmax权重，patch内均匀分配，仅用于本轮attention对象；对56像素网格按像素交叠聚合。

排序均以region行优先ID打破平局；top-K使用单区域分数，预算增加时嵌套。首先选择区域，再计算有限效应。

**主要outcome**是§3.6的完整输入替换效应绝对值 |F(x)−F(x_R)|。另保留旧RQ4的单head隔离outcome：由x_R取得z_h，然后在clean前向中仅替换此head，其他heads及identity均保持clean。两种outcome分别报告，不混合统计。signed effect逐点保存。

候选最优只在实际评估集合中定义：K=1穷举全部196/16个区域，K>1使用八方法（含20个random重复）选择的去重集合；不得称为所有组合中的全局oracle。报告candidate-best regret=max_{R∈C_K}|ΔF(R)|−|ΔF(R_method)|，并逐项保存C_K及有限logit。

## 统计与实现合同

每张图先平均random重复，再平均同一block全部heads。报告每model/reference/block/geometry/outcome/budget的中位效应、random差值、candidate-best regret、effect-budget curve及以最大实际面积归一化的梯形AUC。逐head汇总保留，用于描述层级异质性；另报告图像内平均三个blocks的描述性汇总。两模型独立，不以pooled替代模型级结论。

bootstrap单位为图像，10000次，95% percentile CI。复用经过独立检验的经验CDF中位数bootstrap算法，key包含M2命名空间。配对符号检验比较Endpoint absolute与其余七方法（含signed），分别检验两个方向；所有model/reference/block/geometry/outcome/七baseline/两个方向的AUC检验统一Holm，共672项。逐budget的配对差值及CI为描述性；不能以不显著宣称等效。逐head推断不新增检验。无图像排除，不接受缺图、NaN/Inf或不全head。

实现冻结后发布commit，确认远端包含，再访问正式图像。生产float32、汇总float64；TF32/autocast关闭，确定性开启，Torch导入前CUBLAS_WORKSPACE_CONFIG=:4096:8。动态SQLite原子队列，每任务一次尝试，失败保留根；运行资源信息记录，不把时长当作科学结论。H20≤8卡、free≥32768MiB、util≤10%，记录已有PID并获取协作锁。

每个任务检查head VJP求和闭合、clean noop；每head/reference/geometry的第0候选，比较优化suffix与原full clean-prefix隔离实现。abs tolerance=1e-4、relative=1e-4。合成模型测试覆盖多个blocks、所有head slice及独立有限差分，不用正式结果调阈值。

## Prompt-to-artifact 验收清单

| 要求 | 交付与验收 |
|---|---|
| M1前置通过 | predecessor_validation.json及M1封存SHA、逐任务/推断复算输出 |
| §3.1/3.2跨early/middle/late | 精确2400任务、blocks={2,5,10}、分block统计 |
| §3.3两模型全部heads | 每任务12/6全head、21600个head条件 |
| §3.4两reference与冻结blur | 任务绑定、reference tensor SHA、kernel31 sigma10 |
| §3.5四patch预算与原几何 | 1/10/20/49及legacy1/2/4、逐点selection/area、独立掩码检查 |
| §3.6八方法、同内部成分 | 分数、随机种子、候选logit、逐点排序复算 |
| §3.6中位effect/random lift/regret | summary.csv、paired.csv、candidate逐点复算 |
| §3.6分block/head、curve/AUC | head_summary.csv、图像级汇总、曲线PNG/PDF及数字 |
| §3.7图像统计与统一absolute | 10000bootstrap、672 sign/Holm测试独立复算、signed单独保留 |
| §3.8层级结论边界 | 结果文档逐model/reference/block/outcome限定，未通过不视为等效 |
| §11.5/§13必做3/§14冻结 | 配置、source/task digest、发布commit、最终completion_audit及terminal |
| 原结论及同步论文边界 | 新双语M2补充稿+evidence matrix，不改写M1封存绑定的主稿 |
| 本轮不适用 | M3 token/channel抽样，M4–M6，Path IG，新blur强度、mixed-effects及可选连通区域 |

最终仅在所有任务、来源、方法、候选、指标、图像统计及文档证据核验通过后封存ENDPOINT_M2_PASS_EXECUTION_AND_AUDIT，next_authorization=STOP_AFTER_M2。科学结论依观测签发，不把execution PASS写成Endpoint superiority。
