# NNS-ViT Endpoint Scanner：EP0 协议冻结

日期：2026-09-18。前置版本为 commit `a0386ee9a1aea8f78fd8dfc39d8725d7639c8a63` 的 [规划 R2](NNS_ViT_Endpoint_Scanner论文思路与规划与步骤_20260918.md)。用户本轮指令为“执行 EP0”。本文和下列机器可读包是后续 Endpoint 研究的冻结依据；R2 的 `TBD_EP0` 是历史待决表，以本冻结包的具体值为准。

```text
EP0_PROTOCOL_FROZEN
PRODUCTION_EXPERIMENTS_NOT_YET_AUTHORIZED
EP1_NOT_STARTED
```

入口：[协议](../manifests/nns_vit_endpoint/ep0_r1_20260918/protocol.json) · [完整目录](../manifests/nns_vit_endpoint/ep0_r1_20260918/README.md) · [验证报告](../reports/nns_vit_endpoint_ep0_r1_20260918/validation_report.json) · [terminal receipt](../reports/nns_vit_endpoint_ep0_r1_20260918/terminal_receipt.json) · [相关工作审计](NNS_ViT_Endpoint_Scanner_EP0相关工作审计_20260918.md)。

## 1. 冻结范围与论文定位

主定义仍为 $L_s(x)=x\odot\nabla_xs(x)$，输入是模型实际归一化 RGB 张量。基础量是 G×I；单元、sum、energy、head 和固定类别结构分解使用同一真实 Jacobian。本文定位为 **原始 NNS 的 ViT 适配、统一实现与系统实证审计**。

相关工作核查覆盖原 NNS、IG、Conductance、Captum 的 NeuronGradient/InputXGradient/NeuronConductance、Chefer、AttnLRP、attention rollout、AtP*、activation patching 方法学和 LibraGrad，共 12 条记录。内部目标选择、输入分辨 conductance、activation-displacement Taylor 以及 attention/skip 分析已有先例；本协议不将其作为新的基础公式。C4b 的空间信息增益仍待有限干预评价。

Identity、Correctness、Utility 三层分别产出。EP5-A 的局部方向导数只检验 Correctness；C4a 是整成分关联，C4b 是同预算空间评价增量。任何实验优势主张都不在 EP0 赋值。

## 2. 模型与数据

| 项目 | 冻结值 |
|---|---|
| 主模型 | `vit_base_patch16_224.augreg2_in21k_ft_in1k`；12 blocks，12 heads，D=768 |
| 第二模型 | `deit_small_patch16_224.fb_in1k`，非蒸馏；12 blocks，6 heads，D=384 |
| 共同输入 | RGB；短边 resize 248，bicubic/antialias；center crop 224；patch16 |
| 主模型 mean/std | [0.5,0.5,0.5] / [0.5,0.5,0.5] |
| 第二模型 mean/std | [0.485,0.456,0.406] / [0.229,0.224,0.225] |
| 环境 | Torch 2.2.2+cu121、timm 0.9.16、torchvision 0.17.2+cu121、NumPy 1.26.4、Pillow 10.2.0 |
| 后端 | standard timm Block/Attention，canonical unfused，eval；dropout/drop path 关闭 |
| 开发 / 校准 / 评价 | **64 / 64 / 400** 张，两个模型共用全部 ID；源图像与类别在三集合间互斥 |
| 标签与目标 | 官方 validation WNID 表 → 已核对的 1000 类序；主 ground-truth logit，错分图保留；次级 frozen clean top-1 |

模型文件字节 SHA-256 已实际流式重算：

```text
ViT-B/16
489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d
DeiT-S/16
cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f
```

EP0 未反序列化 checkpoint。EP1 获授权后须检查完整 key/shape、架构、原生前向和 tensor identity。只支持通过注册合同的两个 block 实现。

[cohort_registry.json](../manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json) 保存全部 528 个 ID。选择仅使用类别、文件名和固定 SHA-256 namespace：先排除本地已有 1000 个 validation 文件名及 Stage C/D/X/Y、旧 Endpoint 20 图等元数据的并集，共 **1098 个已知访问 ID**；有候选的类别剩 **990 类**。按 class hash 取 528 类，各类取 image hash 最小的非排除对象，依顺序分为 64/64/400。

旧 Endpoint 20 图与三个新集合交集均为 0。排除审计限定在登记的历史来源，不声称 ImageNet 图像在全球从未被看过。无候选的 10 类在排序前移除；不存在按模型效果替换图像的规则。

本轮读取标签文本与历史样本元数据、列举文件名，只 stat validation tar。正式图像未打开，尚未抽取、哈希或解码；每张图的 payload SHA 在后续获授权的 materialization 中绑定。文件缺失、重复 payload 或模型不兼容时失败并保留现场，不自动换图。完整 ID 及来源足以冻结选择，但不等于证明图像 payload 已可用。

## 3. 目标、分解与存储尺度

[目标注册表](../manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json) 枚举 **ViT-B/16 457 个目标、DeiT-S/16 385 个目标**。

- 每个 block：token 0/1/98/196 × channel 0/D÷4/D÷2/D−1 的单元图；四个 channel 的 patch sum 和 energy；全部 head 的 CLS query、head 内 channel0。
- Block 0/5/11：norm1、attention projection、norm2、MLP fc1/GELU/fc2 的 channel0 patch sum。
- Patch embedding 四个 channel sum、final CLS channel0、ground-truth logit、frozen clean top-1 logit。

Head 方向预先固定；energy 使用同一个 post-block 激活，$\frac12\sum_{t=1}^{196}h_{t,c}^2$，排除 CLS。Margin 本轮不计划。

结构分解覆盖全部 12 blocks、同一个 ground-truth logit。Head-only、Q+K+V、routing+Value 重构固定最终目标的 attention 成分 $A_F$；加 identity 后、在所有输入到目标路径经过该 add 的条件下重构 $L_F$；第二个 add 用 carry+MLP。Bias 参与真实下游状态，直接像素 seed 为零。Hook 点、bypass 和 reconstruction ID 均已登记。

主质量为原始 `[c,h,w]` 上 $m_k=\sum|E_{k,c,h,w}|$，signed sum 为 $a_k=\sum E$。先绝对值后 RGB/空间求和；显示图不参与统计。所有评价图保存 class maps 与目标/成分摘要，前 16 个冻结评价 ID 保存完整 atlas 和结构图，前 8 个生成固定样例图册；近零和失败样本保留索引。

## 4. 评价与干预合同

### 4.1 输入评价与成本

Endpoint/G×I 同一行、full-target IG 和 raw gradient 为核心比较。Grad-CAM 仅作适用类别目标的描述性参考；SmoothGrad-IG/Expected Gradients 本轮不计划。

IG 用 normalized-zero、直线路径、Gauss–Legendre。EP2 在固定 64 张校准图和两个类别目标上从 M=32/64/128/256/512 中选择每模型统一 M；比较 2M 图的 relative L2≤0.02、completeness residual≤0.01，每目标至少 61/64 通过且全部有限。近零 target delta 用绝对残差≤1e−5。最大候选仍失败则 `EP2_NUMERICAL_CALIBRATION_FAILED`，不新增积分阶数、不根据正式样本单独调步数。

Fidelity 同时报告 normalized-zero 和 RGB blur 两种 replacement。Blur 固定 31×31、sigma10、reflect，在 clean 224 RGB crop 上一次生成，然后归一化。Cell4 为3136格、patch16为196格；absolute RGB mass 降序、ties按row-major ID。面积网格0:0.05:1，格数四舍五入，横轴用实际面积。Deletion从clean替换，insertion从相应reference恢复；主 ordinate为固定类别raw logit、梯形AUC，方向分别报告。Normalized AUC仅次级，$|F(x)-F(b)|<10^{-6}$ 记NA而保留raw曲线。

Flip以ground-truth类别为主，同报 $|F_n(x)-F_n(Tx)|$、预测变化、inverse-warp后的signed pixel/absolute cell Spearman。全400图是主集合；delta≤0.1/0.5子集仅描述并报告覆盖。RGB noise sigma0.01/0.03，各3固定seed，clip到[0,1]。内部单元flip因语义对应不明确不纳入本轮。

时间分别测 K=1/16/64/256/完整目标表、固定M和成分C：单目标反向、共享前向多目标反向、batched VJP（若实现）、cut增量、完整atlas摊销。预热3次、重复10次，CUDA同步计时；报告加载/I/O是否计入、P50/P95/max、targets/sec、images/sec、peak memory、bytes/target，不用旧0.01492比值代表全Scanner。

### 4.2 EP5-A 与 EP5-B

EP5-A 令 $v_k=J_{z_k,x}x$，测试 $z_k-\eta v_k$，$\eta$=0.01/0.003/0.001/0.0003，其他切面边固定clean。允许误差为固定eta中最小absolute误差≤$10^{-4}+0.01|a_k|$，全部结果有限且完整误差曲线保留。它只检查hook/方向/cotangent。

EP5-B 使用两种同图reference forward，不使用其他图像donor。对head、routing、Value、attention/MLP输入边执行有限替换，$e_k=F(clean)-F(replace)$。主为 $m_k$ 对 $|e_k|$；$a_k$ 对 $e_k$ 为signed次级。Mandatory comparisons包含 $p_k^{ref}=\langle g_k,z_k-z_k^{ref}\rangle$、output norm、gradient norm、gradient×activation及随机。Q/K单独替换、zeroing和联合routing/value交互在前64个评价ID上作次级分析。

C4b **启用为确认性评价**：固定block5的head 0/H÷2/H−1，16个不重叠56×56候选窗口。比较选前1/2/4格，即面积6.25%/12.5%/25%；各方法用同一窗口池。Endpoint区域分数为成分absolute mass；基线是总G×I、raw gradient、regional Taylor、20次固定随机排序。

Regional Taylor计算候选单窗口replacement的activation位移与clean cotangent的内积绝对值，额外forward成本计入。有限效果采用同一head切面隔离干预：$F(clean)-F(z_h(x_R),z_{-h}(clean),identity(clean))$；候选union的真实效果在排序冻结后测量。每图先对3个head取平均，再计算0–25%面积上的absolute effect AUC÷0.25；与每个基线求图像级配对差。完整输入扰动效果单独报告，不当作某个head的独立效应。

## 5. 确认性统计与出口

[统计计划](../manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json) 共 **16个family、28个检验ID**：

| 族 | 数量 | 主统计量 | 最小实际效应 | CI最大宽度 |
|---|---:|---|---:|---:|
| C4a | 2模型×2reference×3结构族=12 | 每图成分Spearman后取图像中位数 | 相关0.10 | 0.30 |
| C4b | 2模型×2reference=4 | 每图相对4个mandatory baseline的效果分数差，中位数 | raw-logit AUC÷0.25增量0.05 | 0.50 |

每模型固定400张评价图；每family至少360张共同有效图，sign test至少300个非零图像差。样本量按固定精度设计：400个独立类/图像的二项符号比例，在最不利p=0.5时95%近似半宽约0.049。这个计算不是相关或logit统计的功效保证；其精度由上述CI宽度门实际判定。没有运行功效模拟或结果后扩样。

实际效应阈值是本轮在新评价outcome不可见时固定的研究标准：相关0.10要求至少达到弱正排序关联；0.05 raw-logit空间效果增量要求可测量的正增益，并在两种replacement和每个mandatory基线上一致通过。它们不是从旧pilot拟合出的最优阈值，也不是通用领域常数。CI宽度0.30/0.50是本研究精度出口；较宽区间触发INCONCLUSIVE，不靠显著性单独判支持。

CI：95%双侧percentile image-cluster bootstrap，10,000次；保留同图不同方法/reference/model依赖，NumPy1.26.4 PCG64，SHA256导出的family seed，linear quantile。检验采用**双侧精确符号检验**：零差阈值1e−12；有效正负符号在零假设下独立等概率，统计单位为图像，不要求head可交换。28个ID统一Holm、alpha0.05；CI用于精度和展示，不宣称它是28项同时置信区间。

Spearman精确ties用平均秩，排序ties按稳定ID；至少8个成分，常量向量相关NA。主mass保留零；near-zero mass阈值1e−12；signed方向效果阈值1e−6；非有限raw结果触发相应数值INVALID。比较使用mandatory方法的共同有效集，未完成任务不得通过只分析有利子集获得支持。

确定性顺序：合同/cohort/hook/数值违约→**INVALID**；覆盖不全、有效图<360、非零图<300、统计不可估或CI过宽→**INCONCLUSIVE**；point estimate达实际阈值、方向正、相关Holm准则全部通过→**SUPPORTED**；数据充分且精确但条件未通过→**NOT_SUPPORTED**。C4b须同时优于4个mandatory baseline；C4a关联通过不表示优于Taylor。整体两模型结论须所有相应族通过，子族状态完整公开。

## 6. EP1 合同与资源

[实现合同](../manifests/nns_vit_endpoint/ep0_r1_20260918/implementation_contract.json) 固定float64 atol1e−9/rtol1e−7、float32 atol1e−5/rtol1e−4；近零L2阈值分别1e−12/1e−6。逐项检查absolute与relative分支，所有原始值有限。Fused/unfused logit容差另列float64 1e−8/1e−7、float32 1e−4/1e−4。不能暴露结构边的fused实现仅作前向对照。

EP1必须完成CNN paper-formula回归、energy、all-block目标、residual错误总梯度反例、非零head/routing遗漏反例、QKV/branch-total范围、no-op forward、有限差分/JVP、单卡/多卡一致、过期claim拒绝、崩溃恢复及资源测量。EP2只能从冻结batch/alpha候选中选择，不得改样本、目标、检验、阈值或总cap。

**能多卡并行就多卡并行。** 1–8张合格H20，每卡常驻worker，全局SQLite WAL动态队列；至少32768MiB空闲、利用率≤10%，记录既有PID，不占用他人进程。Lease180s、heartbeat30s、递增fencing token防止过期worker提交；每任务immutable结果和exact coverage。Seed来自task ID，非卡号。

| 阶段 | aggregate H20-GPU-hours cap | wall-hours cap | 累计保留GiB cap |
|---|---:|---:|---:|
| EP0 | 0 | 6 | 2 |
| EP1 | 8 | 12 | 16 |
| EP2 | 16 | 24 | 40 |
| EP3 | 64 | 72 | 168 |
| EP4 | 96 | 96 | 192 |
| EP5-A | 8 | 12 | 200 |
| EP5-B | 128 | 120 | 248 |
| EP6 | 0 | 24 | 256 |

累计GPU cap320小时；每进程GPU24GiB、CPU RSS32GiB；每task1800秒、最多2次attempt，磁盘至少50GiB空闲。预算是前瞻性保护上限，不是测量预测；后续阶段仍需授权。Claim前预留在途开销，触cap停止派发并 `PAUSED_RESOURCE_CAP`，保留完成集/待跑清单；无自动扩额或压缩科学工作量。

任何Torch/CUDA导入前设 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；TF32/autocast关闭，float32生产，float64用于CPU fixtures。记录解释器realpath、CUDA/包版本、UUID、模型/数据hash。EP0未查询或初始化CUDA。

## 7. EP0验证与阶段交接

EP0验证器只使用Python标准库，检查封闭且值绑定的schema、17项冻结内容、cohort独立重算、模型/源文件hash、主张/资源/权限一致，并运行16个元数据突变反例：生产越权、重复cohort、top1偷换主目标、mass尺度变化、bypass/Q遗漏、EP5-A当效用、删Taylor/Holm、降阈值、改单CI、删blur、静态分片、增预算、开放payload、未决字段。

这些是**协议验证器测试**，不是EP1模型测试。Source bindings记录22个本地元数据/代码/checkpoint来源，公开文献另记录获取hash与摘录。本轮没有读取新图像/mask、导入Torch/timm/PIL、反序列化模型、做forward/backward、运行GPU或生成新科学指标。

复核命令（仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python \
  vitnns_rebuild/scripts/validate_endpoint_ep0.py
```

封存结果允许把 **EP1_IMPLEMENTATION_AND_REGRESSION** 作为下一步请求；`next_authorization` 仍要求用户明确下达EP1。EP0 PASS不授权EP1执行，更不授权正式outcome access。历史数学图包保持原commit及原hash，本轮不改写。
