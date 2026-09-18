# M1 补充冻结协议与验收清单

本轮仅执行《阶段M_endpointnnsvit.md》的 M1，以及与 M1 同步的 RQ3 表述修正和原 registry 覆盖表。原 E6 omnibus C4a/C4b 保持 `NOT_SUPPORTED`。新增实验为 supplementary frozen study；新增统计方案在原研究之后制定，不能并入原确认性结论。

机器配置为 `config/nns_vit_endpoint/endpoint_m1.json`。两模型各使用同一批 400 张图像、ground-truth logit、0-based blocks 0–11、normalized-zero 与 raw-RGB blur（kernel 31、sigma 10）。所有成分比较共享目标、reference 与成分列表。

E4 已保存大部分逐成分 predictor 和有限效应，直接继承并重算统计。E3 保存所有图像的 identity/carry/Q/K Endpoint mass。仅新增两条 skip edge 的有限替换和基线，以及原 E4 后 336 张缺失的 Q/K。第一条 identity 替换为 `r1=u_ref+attention(norm1(u_clean))`；第二条为 `r2=r1_ref+mlp(norm2(r1_clean))`。替换 skip 时不替换 sibling branch 的输入。identity1/identity2 合并为 residual_identity family；attention/MLP 为 residual_branch。value 与 V 是同一 V projection edge 的两种命名，分别呈现但不当作独立证据。

所有图像都保留。NA 仅适用于常数向量、零 oracle effect、无可比较 pair 或不足四项的 Spearman；记录原因和有效图像数，缺图时不作推断。Q/K/V 三项和 residual 两项比较用 pairwise/top1，不报告连续 Spearman。跨 block、early/middle/late 分开报告；同 block 全 head 排序逐层报告，同时先在图像内平均再推断。pooled 结果先合并同一 image ID 的两模型指标，仅为描述性补充。

每 comparison set 固定 1,000 次 random permutation，保存种子、Spearman 分布摘要/直方图及决策指标的随机均值。Capture 使用 K=1、3（小集合取 min(K,n)）；NDCG 使用非负 absolute-effect linear gain。bootstrap 为图像级 10,000 次，95% percentile CI；配对差值使用单侧 exact sign test，所有 model/reference/family/stratum × baseline 的 Spearman 比较统一 Holm。Endpoint 大于 0 与大于 0.10 为独立检验，另做统一 Holm；不以大于 0 替代实际阈值检验。其余决策指标报告图像级 CI 和与随机的差值。

| 原要求 | 实施/交付物 | 验收证据 |
|---|---|---|
| §2.1 RQ3a / RQ3b | 分开报告绝对关联与相对基线增量 | inference、paired comparisons、结果文档 |
| §2.2 同成分、同目标、同 reference；绝对效应与 mass | 继承 E4，缺失 edge 严格同定义；mean mass=m/150528 | 来源 SHA/canonical digest、效应重算与 parity |
| §2.3 六类 baseline | activation/gradient/actgrad/displacement/Taylor/random | 每一 comparison set 的七种 predictor 完整性 |
| §2.3 random≥1000、固定种子、分布与 topK lift | 配置种子+逐 set 分布统计 | random summary、Capture difference |
| §2.4 Spearman、Capture@1/@3、Regret@1 | 外加 NDCG、pairwise、top1 | 单元测试与逐 image 指标复算 |
| §2.5 A 同 block 全 heads | 每模型12层及图像内平均 | heads=12/6 的逐 set 覆盖 |
| §2.5 B/C 跨层与 early/middle/late | 独立 strata | blocks 0–3/4–7/8–11 的 ID 集合 |
| §2.5 D 模型独立及 pooled descriptive | 模型级统计+同图像 pooled | 400 image IDs 和 model 标签 |
| §2.5 E 八类 family、小集合处理 | Q/K 扩展到400图；两条 skip edge；value/V别名 | family counts、NA规则与pairwise/top1 |
| §2.6 图像单位、中位数、95% CI、paired sign、Holm | CPU统计模块 | 独立指标/推断复算及测试 |
| §2.6 0 与0.10区分、如实报告Taylor | 两种 absolute tests 与 paired tests | 报告不把负 utility 当 execution failure |
| §9 / §13 必做5 RQ3表述与覆盖率 | 中英文稿、evidence matrix、registry coverage | 11/12、广泛但非普遍、omnibus保持 |
| §14 独立冻结文件 | JSON+实现/源快照 | 冻结时间、commit、SHA256 |
| 本次scope外 | M2–M6、token/channel新增抽样、空间预算均不执行 | next_authorization=STOP_AFTER_M1 |

建议性 calibration curve、intervention curve 和 mixed-effects model 不纳入 M1 主执行；文档没有将它们列为 M1 必需交付。M3 的 token/channel 列表及 M2 空间预算在本次不适用。
