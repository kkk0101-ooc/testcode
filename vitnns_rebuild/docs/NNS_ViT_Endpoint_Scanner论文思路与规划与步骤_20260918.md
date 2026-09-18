# NNS-ViT Endpoint Scanner：论文思路、规划与执行步骤

日期：2026-09-18。本文规划一篇以**终点内部扫描与结构化分解**为主线的独立论文。当前已完成方法分析、单图实现验证和数学示意图；下文 EP0–EP6 是待执行计划，阶段名只用于本论文，不替代仓库已有阶段授权。

修订 R2：已纳入 EP0 前的目标、分解、干预和统计澄清要求；具体 cohort 数量、统计阈值、模型绑定与资源上限仍须在 EP0 填写并冻结。当前状态为：

```text
PLAN_READY_FOR_EP0_WITH_REQUIRED_CLARIFICATIONS
PRODUCTION_EXPERIMENTS_NOT_YET_AUTHORIZED
```

本状态表示规划可作为 EP0 工作底稿，不表示 EP0 已 PASS。本轮交付为文档修订与历史来源复核，不访问正式评价数据、不运行 EP1/EP5，也不启动 GPU 生产。第 6.1 节任一 `TBD_EP0` 未解决时，不能签发 `EP0_PROTOCOL_FROZEN`。

入口：[数学分析与方法对比](NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md) · [实现与用法](endpoint_nns_vit_scanner_20260918.md) · [真实单图 Image #1–#9](endpoint_nns_vit_scanner_figures_20260918.md) · [本轮数学图及说明](../reports/nns_vit_endpoint_math_comparison_r1_20260918/figure_analysis.md) · [图册](../reports/nns_vit_endpoint_math_comparison_r1_20260918/gallery.html)。

## 1. 论文定位与研究问题

建议标题：**NNS-ViT: Endpoint Scanning and Structured Decomposition of Vision Transformers**。

中文：**NNS-ViT：视觉 Transformer 内部响应的终点扫描与结构化分解**。

中心问题：NNS 的终点 learning-image 原则能否被忠实适配为面向 ViT 内部目标的结构化扫描框架，其输入分辨成分能否闭合，并在有限干预评价中提供行为预测信息？

证据分三层：**Identity** 是 $L_s=x\odot\nabla s$ 的方法身份；**Correctness** 是成分闭合、hook 和局部微分合同；**Utility** 是有限 replacement 与空间定向干预的评价。局部方向导数检查属于 Correctness。三层分别报告，闭合或局部合同通过不替代 Utility 证据；有限干预相关也不等于完整因果机制。

核心量固定为

$$
L_s(x)=x\odot\nabla_xs(x),\qquad
x=\operatorname{Normalize}(\operatorname{Crop}(\operatorname{Resize}(I))).
$$

它等于内部标量目标上的 Gradient × Input。技术内容是目标注册、ViT 真实算子适配、结构分解及其评价。数学身份由链式法则建立；解释是否有用由跨图实验回答。

对原始 NNS-CNN 的继承范围是输入值初始化、当前点局部传播、保留输入坐标和逐内部单元扫描。条件是使用与网络一致的门控/局部 Jacobian、包含所有路径，并比较原始 signed 图。原论文聚合、阈值和显示不自动属于这个等价定理；末端 ReLU 门记号及官方实现可得性问题见数学分析第 3.5 节。

## 2. 方法身份与贡献清单

| 对象 | 核心量 | 求值状态 | 论文角色 |
|---|---|---|---|
| 原始 NNS-CNN 单神经元图 | 列明条件下为 $x\odot\nabla s_{\rm CNN}(x)$ | 当前权重、ReLU 门、池化选择 | 原理来源及回归对象 |
| NNS-ViT Endpoint Scanner | $x\odot\nabla s_{\rm ViT}(x)$ | 完整 ViT 当前 Jacobian | 内部扫描主方法 |
| 此前 internal-target Path IG | $(x-x_0)\odot\int_0^1\nabla s(x_\alpha)d\alpha$ | 沿参考路径变化 | 成本、fidelity、稳定性比较 |
| 此前 fixed-attention Value-path IG | 对固定 $A(x)$ 的条件目标做 IG | clean routing 固定 | 既有方法的特定范围 |
| baseline-relative endpoint extension | $(x-x_0)\odot\nabla s(x)$ | 同终点、不同 multiplier | 单独的 seed 分析 |

$x_\alpha=x_0+\alpha(x-x_0)$。对于 baseline-relative endpoint 与 IG，路径 Jacobian 恒定是相等的充分条件；充要条件是每个 $x_i\ne x_{0i}$ 的坐标满足终点梯度等于路径平均梯度。Canonical Endpoint 与一般 baseline IG 还存在 seed 差异。CNN 与 ViT 的架构不同，不决定采用 endpoint 还是 path。

贡献按证据状态组织：

1. **形式化适配。** 统一推导 NNS endpoint 与 G×I 的关系，给出 ViT 完整局部 Jacobian 合同。
2. **内部扫描接口。** 注册单元、聚合、head 和类别目标，保留输入坐标；补齐 energy target 后再声明对应原 NNS filter aggregate。
3. **结构分解。** 对 residual/head 计算边和 attention 微分路径给出成分图、闭合条件与实现验证。
4. **实验评价，待完成。** 量化扫描成本、输入归因质量及稳定性，评价结构成分预测匹配干预的能力。

前三项仍需相关工作核对以确定新增程度。G×I、链式法则和一阶加法本身不作为新算法贡献。干预预测优势须在 EP5-B 的有限行为评价中得到支持后再写进摘要；EP5-A 的局部合同通过只计入实现正确性。

## 3. 方法部分的数学骨架

### 3.1 初始化与传播命题

$$
L_{x_p}[i]=x_i\mathbf1(i=p),\qquad \mathcal L(x;x)=\operatorname{diag}(x).
$$

对 $y=f(u_1,\ldots,u_r)$，在真实输入状态上传播

$$
L_{y_j}[i]=\sum_{a,k}\left.\frac{\partial y_j}{\partial (u_a)_k}\right|_x L_{(u_a)_k}[i].
$$

在可微点，或非光滑算子采用固定一致的次梯度约定时，沿有向无环图归纳得到

$$
\mathcal L(s;x)=J_{s,x}(x)\operatorname{diag}(x),\qquad
L_s[i]=x_i\frac{\partial s(x)}{\partial x_i}.
$$

证明放正文一个命题，算子条件放附录。显式逐层传播与完整 VJP 是同一数学量的计算方式。

### 3.2 目标注册

| 对象 | 精确标量/成分 | 当前状态及下一步 |
|---|---|---|
| token-channel 单元 | $s=h_{l,t,c}$ | 已实现；记录 pre/post operator、token、channel |
| patch-channel sum | $s=\sum_{t\in\mathcal T}h_{l,t,c}$ | 已实现；明确 $\mathcal T$ 是否含 CLS |
| patch-channel energy | $s=\tfrac12\sum_{t\in\mathcal T}h_{l,t,c}^2$ | 待实现；固定张量和 token 集合 |
| head 内部目标 | $s=u^TO_{l,h,t}$，如 $u=e_c$ | 已实现固定通道；一般方向须预先冻结 |
| logit / margin | $s=\ell_y$ 或 $\ell_y-\ell_r$ | 已实现；跨图冻结类别规则 |
| 对最终目标的结构成分 | 同一 $F$ 的 $E^C_{k,i}$ | 已实现标准 timm block 的有限单图验证 |

原 NNS filter aggregate 对应

$$
\sum_t h_{l,t,c}L_{h_{l,t,c}}
=x\odot\nabla_x\left(\frac12\sum_t h_{l,t,c}^2\right).
$$

权重和单元图必须使用同一激活标量。原文未清晰绑定 pre-ReLU/post-ReLU 时，在复现中分别注册并报告具体张量。Channel sum 是不同标量目标。

**主类别设计固定为 ground-truth logit**：对带标签图像 $(x_n,y_n)$，$F_n(u)=\ell_{y_n}(u)$，保留错误分类图，correctness 仅作预定描述性分层。跨图估计的是“按真实标签构造目标”的规则表现，并非所有图共享同一个固定类别函数。若增设固定类别 $c$ 的分析，则 $F(u)=\ell_c(u)$，其 cohort、类别与指标单独注册。

**次级设计为 frozen clean top-1**：$\hat y_n=\arg\max_y\ell_y(x_n)$、$F_n^{top1}(u)=\ell_{\hat y_n}(u)$。它估计选择 clean top-1 后解释该 logit 的目标构造规则；clean 选类后在所有扰动、翻转、reference 和方法中固定，ties 使用预定类别索引顺序。不按结果在主/次目标间择优。

EP0 固定单元、sum、energy 和 head 的精确张量位置及 token 集合，内部方向仅在 discovery 数据上确定。单元自身图与其对最终类别的成分图分栏报告。Margin 的两类别构造规则如启用，另列为次级目标。

### 3.3 算子对应

| CNN 原规则 | ViT 适配 | 合同 |
|---|---|---|
| affine/conv | patch embedding、QKV/MLP/output projection | 真实线性 Jacobian |
| ReLU 当前门 | GELU 当前导数 | 不把 GELU 二值化 |
| MaxPool 当前 winner | attention 输入相关混合 | 都使用当前 Jacobian；softmax 不是 winner 选择 |
| 无直接对应归一化项 | LayerNorm | 均值、方差及通道耦合全部保留 |
| skip/add | 两次 residual add | 输入边传播，防止重复计数 |

LayerNorm 使用 PyTorch 总体方差：$\sigma^2=D^{-1}\sum_j(u_j-\mu)^2+\epsilon$，$\hat u_j=(u_j-\mu)/\sigma$，

$$
\frac{\partial y_i}{\partial u_j}=\frac{\gamma_i}{\sigma}
\left(\delta_{ij}-\frac1D-\frac{\hat u_i\hat u_j}{D}\right).
$$

$\beta$ 的局部输入导数为零，但改变下游状态。CLS、position 和 bias 同样参与真实前向，没有直接像素 seed。$s(x)-\sum_iL_s[i]$ 是局部仿射截距，不是完整模型常数贡献账本。

### 3.4 Attention 完整微分

$$
Q=UW_Q,\quad K=UW_K,\quad V=UW_V,\quad
S=QK^T/\sqrt{d_h}+B,\quad A=\operatorname{softmax}(S),\quad O=AV,
$$

$$
dO=(dA)V+A(dV),\qquad
dA=J_{\rm softmax}(S)\left[(dQ)K^T/\sqrt{d_h}+Q(dK)^T/\sqrt{d_h}+dB\right].
$$

Softmax Jacobian 按 query row 作用。固定位置 bias 满足 $dB=0$；输入相关 bias 增加成分。完整传播还包括 concat、$W_O$、其 bias 的下游作用及后续模块。

**Attention backend 硬合同：**结构分解的 canonical implementation 使用显式 **unfused attention**。模型设为 `eval()`，验证 attention dropout、projection dropout、stochastic depth 均关闭。注册并绑定以下张量/边：Q/K/V projection 后、逐 row softmax 后、$AV$ 后且 concat 前、output projection 前后、两个 residual add 各自的两个输入边和输出。

EP1 对同输入、同权重、同 dtype 的 fused/unfused 原生前向做冻结容差内 parity，并检查无 hooks、no-op hooks 和正式 hooks 的前向一致。Fused kernel 无法暴露所需边时，仅可用作前向/性能对照，不能声称支持其结构分解；若环境无法运行 fused 对照，记录该项缺失并限制已验证 backend 范围。论文模型范围为“已注册且通过合同的 ViT block”，不泛化到所有 ViT。

### 3.5 Structured endpoint decomposition

对标量 $F$ 和向量/张量成分 $z_k$，

$$
E^C_{k,i}=x_i\left\langle\nabla_{z_k}F(x),\frac{\partial z_k(x)}{\partial x_i}\right\rangle.
$$

内积遍历所有成分坐标。若每条输入到目标路径恰好穿过一个注册边，无遗漏、无重复，且共享源使用显式输入边，则

$$
\sum_k E^C_{k,i}=x_i\frac{\partial F(x)}{\partial x_i}.
$$

对 $r=u+a(u)$，且通向 $F$ 的所有输入路径经过 $r$，定义 $g_r=\nabla_rF$：

$$
E_{\rm id}=x\odot J_u^Tg_r,\quad E_{\rm attn}=x\odot J_a^Tg_r,\quad
E_{\rm id}+E_{\rm attn}=x\odot\nabla_xF(r(x)).
$$

VJP 的下游 cotangent 是固定 seed。$\nabla_uF$ 已含 identity 和 branch，不能再作为 identity 加上 branch。

记 $A_F=E_{\rm attn}^{F}$ 为固定最终目标 $F$ 在第一次 add 的 attention 输入边上定义的像素成分；$L_F=x\odot\nabla_xF$ 为完整目标图。$A_F$ 包含真实下游 cotangent，并不等于另选一个 attention 内部标量目标的 learning image。每个 decomposition registry 必须写明 `target_id / block / hook_edges / reconstruction_id / required_bypass`：

| 注册分解 | 成分 | 精确重构对象与条件 |
|---|---|---|
| Head-only | 所有 projection 前 head 成分 | $A_F$；各 head 经 concat/$W_O$ 和同一 downstream cotangent |
| Identity + all heads | 第一次 add 的 bypass + 全 head | $L_F$；通向 $F$ 的每条输入路径经过该 add |
| Routing + value | routing + Value | $A_F$；routing = Q + K (+ 输入相关 B) |
| Q + K + V (+B) | 全 attention 微分路径 | $A_F$；包含所有输入相关项 |
| Identity + routing + value | bypass + routing + Value | $L_F$；第一次 add 的完整路径条件成立 |
| Identity + attention | bypass + $A_F$ | $L_F$；与 head 展开属于同一 add 的不同划分 |
| Carry + MLP | 第二次 add 的 carry + MLP 边 | $L_F$；所有输入到 $F$ 的路径经过第二次 add |

Projection bias 的直接输入导数为零，但其真实前向值影响 downstream cotangent。注册额外旁路的模型须补齐该边后才能用总图闭合；head-only/branch-only 不能拿来与完整类别图比较后宣称闭合。若目标就是选定 residual 输出的一个标量坐标，则相同规则应用于该明确注册的标量目标。

不同 block、切面不能混加；routing 也不能与 Q/K 重复相加。此前 path 成分使用对称形式

$$
K^C_{k,i}=\Delta x_i\int_0^1
\left\langle\nabla_{z_k}F(x_\alpha),\frac{\partial z_k(x_\alpha)}{\partial x_i}\right\rangle d\alpha.
$$

完整切面条件须沿路径成立。它是 input-resolved path conductance，Endpoint 是单点 Jacobian decomposition。

### 3.6 坐标与完整性

若 $x=(r-\mu)/\sigma$，则 $x_i\partial_{x_i}s=(r_i-\mu_i)\partial_{r_i}s$。论文以 coordinate-origin dependence 描述该性质。Canonical Endpoint 没有 reference forward；raw-RGB seed 和 normalized-input seed 是不同分析。IG 的 affine-coordinate consistency 要求 baseline 同步变换并使用同一物理路径。一般 $\sum_iL_s[i]\ne s(x)$，也没有 baseline-difference completeness。

### 3.7 Component mass 的唯一主尺度

主成分数组为 normalized model-input 坐标上的原始 signed RGB 张量 $E_{n,k,c,h,w}$。在 RGB 求和、降采样、clipping 和显示归一化**之前**计算：

$$
a_{n,k}=\sum_{c,h,w}E_{n,k,c,h,w},\qquad
m_{n,k}=\sum_{c,h,w}|E_{n,k,c,h,w}|.
$$

空间 signed 像素图 $S_{h,w}=\sum_cE_{c,h,w}$ 与 absolute 像素质量 $M_{h,w}=\sum_c|E_{c,h,w}|$ 分别保存；通常 $|S_{h,w}|\ne M_{h,w}$。4×4 cell/native 16×16 patch 的主 absolute score 是组内 $M$ 之和，sum 后取 absolute 是另一派生量，不能冒充主 mass。显示图可降采样，但不进入 $a_k,m_k$ 主统计。

## 4. 当前证据与缺口

| 项目 | 已有结果 | 论文用途 |
|---|---|---|
| 单图 40 目标，ViT-B/16 | 一次共享扫描前向；CPU float64 | 示例与目标覆盖 |
| 类别 G×I parity | 最大误差 0 | 同一 autograd 图的不同调用路径一致 |
| 同目标结构闭合 | max-abs $4.44\times10^{-16}$；relative L2 $3.14\times10^{-15}$ | hooks/cotangents/记账验证 |
| 40 目标耗时 | 4.32 秒，CPU、4 threads | 单次记录，不充当 GPU 吞吐 |
| 20 图 4×4 fidelity 中位数 | Endpoint 0.03639；Path 0.10143 | 历史探索性比较 |
| 20 图 flip Spearman 中位数 | Endpoint 0.34104；Path 0.47416 | 历史稳定性结果 |
| Endpoint/Path 时间比中位数 | 0.01492 | 限该 pilot，不外推固定加速比 |

来源：[Scanner report](../reports/endpoint_nns_vit_scan_r3_20260918/scan_report.json)；[20 图 summary](../reports/nns_vit_endpoint_level4_r1_20img_20260917/summary.json)；[20 图决策](endpoint_nns_vit_level4_decision_20260917.md)。

20 图 fidelity、flip、runtime 使用 normalized-zero 条件，endpoint 为 $x\odot\nabla s(x)$；reference gate 比较 $x\odot\nabla s(x)$ 与 $(x+1)\odot\nabla s(x)$，属于两个 multiplier 的比较，不是 canonical 方法内置 reference sensitivity。原决策 `NO_GO_CURRENT_ENDPOINT_PROTOCOL` 保留。新论文的扫描定位需要新的实验问题与预定指标。

待补：paper-formula CNN 回归、energy、全 block/多输入/多精度验证、独立数据评价、多卡生产入口、吞吐测量、干预评价和新颖性核对。现有 `run_endpoint_vit_scan.py` 固定 CPU；设置多张 `CUDA_VISIBLE_DEVICES` 不会使它自动并行。

## 5. 四组实验

### A：原理继承与正确性

1. 在明确算子的 CNN 上实现 paper-formula 传播，与独立 autograd G×I 比较；包含 affine bias、ReLU 打开/关闭、唯一 MaxPool winner、residual。边界/tie 单列固定约定。
2. ViT 覆盖全部 block、单元、sum、energy、head、logit、margin，检查 hook 前后原生前向一致。
3. 同报 max-absolute 和 relative L2；近零图使用预定绝对容差。Float64 验算，生产精度另行校准。
4. 选定目标用有限差分/JVP 检查方向导数；闭合覆盖两个 residual、head+bypass、Q/K/V。

验收为实现符合定义。官方 NNS 代码若不可得，命名为 paper-formula regression，不写成官方实现逐位复现。

### B：内部扫描与 atlas

固定层、token 集合和方向，生成浅/中/深层 atlas。记录激活、signed sum、absolute mass、集中程度和图间相似度。以全注册目标分布为统计主体，展示样本按冻结规则抽取，公开普通及近零/失败样本。

EP0 的 atlas registry 必须固定 layer/token 集合、channel/head 选择规则、示例图像抽样 seed、near-zero 判据及展示顺序。主展示为一个 signed map；positive/negative/absolute 派生图若展示，分别命名。每图归一化函数、共享或逐图色标、clipping 参数和降采样规则一并冻结。正文精选案例标明选择规则，并同时提供全目标索引、未筛选分布、固定抽样图以及失败/近零案例；颜色大小只在约定色标内比较。

分别报告单元图、energy 图、head 内部图和固定类别结构图。跨图保持目标语义；不同模型同编号 channel 不自动语义对齐。冗余主张预定义相似度、零质量处理和每图汇总。Atlas 展示现象，干预评价其行为意义。

默认措辞为 **registered internal scanning**。C2 只有完成 EP0 注册的模型/层/目标规模后才可称 scalable；40 目标示例不称 full-neuron scanning。规模表同时给出 targets/sec、images/sec、peak memory、stored bytes/target、目标覆盖率和完整 atlas 的摊销成本。

### C：Endpoint 与 Path IG 的成本—质量

主比较使用 canonical Endpoint 与 $x_0=0$ 的完整 internal-target IG，保持 seed 相同。多参考 IG、baseline-relative endpoint 单列。固定目标、输入、精度、batch、后处理，IG 步数先校准收敛。

成本分别报告 $T_{end}(1)$、$T_{IG}(1,M)$、$T_{end}(K)$、$T_{IG}(K,M)$、$T_{cut}(K,C)$，其中 $M$ 是积分节点数、$K$ 是目标数、$C$ 是注册成分数。分清单目标 backward、共享 forward 的多目标 backward、batched VJP（若实现）、结构分解额外成本和端到端 atlas 成本；记录预热、I/O、数据类型和实际调度。历史比率 0.01492 只绑定旧 pilot。

**Insertion/deletion 冻结合同。** 主排序取原始 RGB attribution 的组内 absolute mass（第 3.7 节）；signed-positive 排序作为单独次级条件。固定方法间同一目标、同一 cell 网格与预算、ties 的 cell-index 顺序。224×224 且无额外边缘裁切时，4×4 是 56×56=3136 cells，16×16 是 14×14=196 patches；其他输入尺寸须另注册。正贡献排序须说明先 RGB signed sum、再空间组 sum、再取正部的顺序。

主 replacement 条件同时包含 normalized-zero 与由 clean 图像按冻结核生成的 blur。EP0 填写 blur 所在 RGB/normalized 空间、kernel、sigma、边界规则；blur 图只构造一次，不随删除进度重算。两条件都报告，不按效果挑选一种。Deletion 从 clean $x$ 逐组替换为同位置 replacement；insertion 从对应 replacement 起点逐组恢复 clean。Mask 覆盖所选组的全部 RGB 坐标。Noise/inpainting 等 OOD 稳健性条件如启用须独立冻结随机数或模型与参数。

AUC 横轴为已替换/恢复的输入面积比例 $q\in[0,1]$，登记具体 $q$ 网格并用梯形积分。主曲线 ordinate 使用冻结目标的 raw logit 或相对 clean 的 signed effect，EP0 在两者中固定一种并写出公式；使用 absolute 排序时不预设所有 deletion 都降低 logit。若另报告除以 $F(x)-F(x_{base})$ 的 normalized AUC，则预设近零阈值，该派生指标分母不合格记 NA，raw 曲线仍保留。Legacy combined fidelity 与新 AUC 分表报告。

**Flip 稳定性以类别目标为主。** 保持 $F_n$ 的类别规则，报告 $d_n^{flip}=|F_n(x_n)-F_n(Tx_n)|$、clean/flipped prediction 是否改变、以及 inverse-warp 后的图相似度。全 cohort 是主报告；另按 EP0 冻结的 $d_n^{flip}$ 阈值/分层展示目标变化受控子集及覆盖率，不据结果挑选阈值。它们联合反映模型和解释器变化，不能单独归因给解释器。

$|F_n(x)-F_n(T^{-1}T(x))|$ 对精确水平翻转通常为零，只是往返变换检查，不衡量模型翻转等变性。内部 token-channel 的翻转对应仅作次级研究，登记 token 置换、方向和位置编码作用；同坐标/同 channel 不保证同语义。Noise 强度、裁剪空间、seed 和目标响应变化同步登记。

核心输入基线：raw gradient、G×I（与 Endpoint 同一行）、IG。增强组依研究问题加入 SmoothGrad-IG、多参考 IG/Expected Gradients 和目标适用的 Grad-CAM 类方法，记录适用范围。

### D：结构图与匹配内部干预

固定 $F$、block、输入边，定义

$$
a_{n,k}=\sum_iE_{n,k,i},\quad m_{n,k}=\sum_i|E_{n,k,i}|,\quad
e_{n,k}=F(x_n)-F(x_n;\operatorname{replace}(z_k,z_k^{\rm ref})).
$$

$e>0$ 表示替换后目标下降。检验 $a_k$ 对 signed effect、$m_k$ 对 absolute effect 的经验关系。

**微分方向必须对齐。** $a_k=\langle g_k,J_{z_k,x}x\rangle$；reference replacement 的一阶预测为

$$
p_k^{\rm ref}=\langle g_k,z_k(x)-z_k^{\rm ref}\rangle.
$$

通常 $J_{z_k,x}x\ne z_k-z_k^{\rm ref}$，所以 $a_k$ 不自动是替换干预的 Taylor 预测。把 $p_k^{\rm ref}$ 作为强基线，另评价成分图提供的空间信息。

**EP5-A：局部微分合同（Correctness）。** 令 $v_k=J_{z_k,x}x$，在注册边上置 $z_k(\eta)=z_k-\eta v_k$，其他切面边固定 clean，重算下游：

$$
\lim_{\eta\to0}\frac{F(x)-F(z_k-\eta v_k)}{\eta}=a_k.
$$

冻结递减 $\eta$ 集合，报告近似误差及数值噪声。该关系来自微分定义；通过只说明 hook、cotangent、方向和 component 实现符合合同，不能计为外部解释价值或 C4 支持。EP1 先用 fixtures 检查，EP5-A 可在获授权的评价样本上复核，仍按 correctness 报告。

**EP5-B：有限反事实评价（Utility）。** 主预测量和检验问题固定为：

| 子主张 | 主预测量与对象 | 对照与结论范围 |
|---|---|---|
| C4a：整成分有限替换的关联 | 主为 $m_k$ 对 $\lvert e_k\rvert$；$a_k$ 对 $e_k$ 是预注册的 signed 次级分析 | 报告 $\lvert p_k^{ref}\rvert$ 对 $\lvert e_k\rvert$、$p_k^{ref}$ 对 $e_k$ 及简单基线；C4a 关联通过不意味着优于基线 |
| C4b：输入分辨图的额外空间信息 | 原始 $E_{k,i}$ 选择空间区域，评价同预算有限效应或效果曲线 | 必须相对总 G×I、原始梯度、随机区域，以及 activation-space Taylor 基线的相应空间化版本比较 |

$p_k^{ref}$ 若优于 $a_k$，说明该 activation displacement 与 reference replacement 更匹配；C4a 的预测结果按实报告，数学身份和闭合结论不随之失效。C4b 另检验空间信息，不能从 $a_k,m_k$ 两个标量的相关自动推出完整空间图有增益。

**C4b 空间定向干预候选合同，须在 EP0 决定是否作为确认性 family。** 用 clean 成分图的 RGB-level absolute mass 选择固定面积预算 $R_k(q)$，得到 $x^{R,b}=x\odot(1-\mathbf1_R)+b\odot\mathbf1_R$。预算 $q$、网格、replacement $b$、ties 及随机重复数在 EP0 固定。先测完整前向的 $e_F(R)=F(x)-F(x^{R,b})$，再测结构隔离的有限效果

$$
e_{k,R}=F(x)-F\bigl(z_k(x^{R,b}),z_{-k}(x)\bigr),
$$

其中 $z_k,z_{-k}$ 必须属于同一个明确注册的完整切面，下游从该切面重算。成分统计如 output norm/固定方向响应差另列，不用任意张量求和替代原目标。$e_F$ 含全部路径的变化，不能归给单一 $k$；$e_{k,R}$ 是混合状态有限干预，仍受 reference/OOD 影响。

用于该区域评价的对应 Taylor 基线是 $p_{k,R}=\langle g_k,z_k(x)-z_k(x^{R,b})\rangle$。它与整成分 $p_k^{ref}$ 分别登记。若用候选区域的 $p_{k,R}$ 排序，所有方法使用同一候选区域集合与面积，额外前向成本单列；不得读取有限效果 $e_{k,R}$ 后选区域。跨图训练的组合预测器仅在开发集拟合、评价前冻结。C4b 未实施或仅探索时，只报告 C4a，不声明空间信息增益。

| 干预族 | 位置与操作 | 控制 |
|---|---|---|
| head | concat 前或等价投影前切片替换，重算投影/下游 | bypass clean |
| routing | $O=AV$ 处替换 $A$，$V$ clean | 同模型、同 token 布局 |
| value | 替换 $V$，$A$ clean | 同一 reference 配对规则 |
| Q/K | 各自替换后重算 softmax | 与直接替换 $A$ 分列 |
| attention/MLP branch | 加法器 branch 输入边替换 | identity/carry clean |

Reference 来自预注册参考输入或配对样本；EP0 固定来源、配对规则、是否允许同类 donor、donor 所属划分及一次/多次采样汇总，禁止根据效果选择 donor。Zeroing 是把注册 activation edge 置零，区别于 normalized-zero 输入产生的 reference activation，另设稳健性条件。干预 reference 不改变 Endpoint 定义。有限 routing/value 联合效应可含交互，不能由微分闭合推导 $e_{R,V}=e_R+e_V$；报告联合效应和交互残差。

结构基线至少包括 output norm、gradient norm、gradient×activation、$p_k^{\rm ref}$、随机排序；attention 另加 CLS attention/entropy/rollout。若 leave-one-head-out effect 定义被预测的 $e_k$，它是评价标签/代价参照，不再作为免费 predictor 排名。

## 6. 统计、样本与主张判定

独立单位为图像。每图先在注册成分集合 $\mathcal K_n$ 内计算相关、top-k enrichment、方向一致率，再按图像汇总：

$$
\rho_n=\operatorname{Spearman}_{k\in\mathcal K_n}(m_{n,k},|e_{n,k}|).
$$

Routing/value 只有两个成分时，单 block Spearman 是退化排序。预定跨 block 成分集合后每图汇总，或采用成对排序正确率和效果误差。零方差相关记 NA；全零成分保留在覆盖/质量计数，不能只因归因为零就剔除图像。

Paired image-level bootstrap 给出方法差的区间；报告 median、有效样本数及分布。Head、block、重复随机种子不是独立样本。主模型和第二模型分开报告；联合汇总若启用则按图像 cluster bootstrap，保留同图不同模型的依赖。描述性 atlas 与确认性主张分开。

已有 20 图作为历史探索材料；开发集、校准集、评价集按源图像身份互不重叠，并登记它们与历史 20 图的交集。EP0 固定数量、抽样 seed、源数据权限与分配算法，可用元数据确定注册图像；不能读取正式评价 outcome 选样。样本量由最小有意义差异、区间精度或预先功效设计确定；本轮不执行功效模拟。主模型/第二模型分别绑定 checkpoint、preprocessing 和样本数，泛化主张需要第二模型实测。EP2 只按预定算法校准计算参数，不改变这些统计选择。

| 主张 | 证据 | 写法 |
|---|---|---|
| C1：原理适配与实现正确 | 命题 + CNN/ViT parity/闭合 | 明确算子和模型范围 |
| C2：注册规模的扫描能力 | 注册目标规模 + 四项吞吐/存储指标 | 达到冻结规模后才用 scalable |
| C3：成本—质量关系 | 同目标配对实验 | 报告实际收益与损失 |
| C4a：整成分有限替换关联 | $m_k$ 对 $\lvert e_k\rvert$，signed 关系单列 | 只报告关联；不由此声明优于基线或空间增益 |
| C4b：空间信息增益 | 同预算区域干预，强制基线的配对增量 | 只有本项支持才声明输入分辨图增加预测信息 |

输出 `SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / INVALID` 及对应指标。实现正确和科学主张分别判定；历史负结果进入讨论，结论范围由新证据决定。

### 6.1 EP0 统计冻结登记表

以下为待填入机器可读 protocol 的要求，`TBD_EP0` 表示尚未决定、不能签发冻结状态。数值约定可在本底稿中提出，但只有 EP0 校验后才成为运行合同；正文不能把默认建议写成已完成预注册。

| 字段 | 本底稿的规则/建议 | EP0 必须落盘的值 |
|---|---|---|
| `n_discovery / n_calibration / n_evaluation` | 三个互斥集合；列与旧 20 图交集 | 各自整数、样本来源、ID 清单/抽样算法、理由：`TBD_EP0` |
| `models[].n_evaluation` | 主模型、第二模型分别登记 | 模型数、每模型 N、共享图像关系、checkpoint/preprocessing digest：`TBD_EP0` |
| `families[]` | C4a 整成分、C4b 空间增量分别注册；signed、replacement、第二模型若进入确认性判断都需枚举 | 完整 family 数量、成员、primary/secondary：`TBD_EP0` |
| `primary_statistic` | C4a 默认图像内 absolute Spearman 再取图像中位数；C4b 默认图像内同预算绝对效果曲线 AUC 的方法差，再取中位数 | 每 family 的精确公式、方向、聚合轴：`TBD_EP0` |
| `mandatory_baselines` | C4a 同报 $\lvert p_k^{ref}\rvert$、norm/gradient、随机；C4b 比较区域 Taylor、总 G×I、raw gradient、随机 | 每 family 必须完成的具体基线/候选区成本：`TBD_EP0` |
| `delta_min` | 相关与效果/AUC 使用各自单位；不共用无量纲阈值 | 每 family 最小有意义效应/增量及理由：`TBD_EP0` |
| `ci_level / bootstrap_replicates` | 建议双侧 95% image-cluster percentile CI，10,000 次配对重采样 | 方法、水平、次数、固定 seed、分层规则 |
| `test / alpha / alternative` | 建议 $\alpha=0.05$、双侧；不从普通 bootstrap CI 反推未定义的 p 值 | C4a null 的成分置换/交换性条件、C4b 图像配对方法交换检验或另一有依据方法、置换次数/seed：`TBD_EP0` |
| `holm_scope` | 对所有用于确认性支持判断的检验 ID 统一列清单；禁止逐模型/效果有利时拆 family | Holm 清单、总检验数、排序/tie 约定：`TBD_EP0` |
| `n_valid_min / ci_width_max` | 按 family、模型与方法配对交集判断；同时报告各法总覆盖率 | 最少有效图像整数、最大区间宽度：`TBD_EP0` |
| `near_zero` | 对 direction、normalized AUC、closure 单独定义；不能随结果改变 | 每 dtype/指标的绝对或相对阈值与单位：`TBD_EP0` |
| `ties / NA / nonfinite` | Spearman 用平均秩；ranking ties 用稳定注册 ID；常量向量相关 NA，非有限原始结果按数值合同处理 | 最少有效成分数、NA reason codes、配对有效集规则 |
| `exit_rules` | 第 6.2 节的有序判定；缺少阈值时不能运行确认性判定 | 每 claim/family 的阈值映射、基线差检验 ID、必需子主张集合 |

主排序的 absolute mass 包含零值；near-zero 只影响已注册的方向/归一化等指标，不改变 cohort。图像内 Spearman 若有效成分太少或任一向量恒定则记 NA；少于最低图像数属于 INCONCLUSIVE。禁止填 0 代替 NA、按方法各自有利的有效子集比较，或把 head 数当样本量。

### 6.2 确定性科学出口

每个启用的 claim/family 按以下顺序执行，资源状态另记：

1. **INVALID**：cohort/权限、hook、数值或 protocol 合同被破坏，影响该主张；报告精确失败项。
2. **INCONCLUSIVE**：合同有效，但配对有效图像数低于 `n_valid_min`、必需统计量无法估计或 CI 宽于 `ci_width_max`。缺样原因和区间原样报告。
3. **SUPPORTED**：数据与精度条件满足，主统计量达到 `delta_min`，Holm-adjusted 统计准则通过且方向正确。对 C4b 这种“增益”主张，还必须对每个 mandatory baseline 的配对改进为正，达到登记的实际阈值并通过其列入 Holm 的检验。
4. **NOT_SUPPORTED**：执行有效、数据量及精度足够，但上述支持条件未全部满足。它和 INCONCLUSIVE 分开。

C4a 只判断有限替换关联；即使 SUPPORTED，也不宣称优于 $p_k^{ref}$。若论文使用整体“结构图增加行为预测信息”表述，必须列出所依赖的 C4b family 并全部满足其强制基线条件，不能用 C4a 或 EP5-A 替代。子 family 结果逐项公开，未启用的研究项记 `NOT_PLANNED` 而非伪造一个支持/失败结论。

当仅因资源 hard cap 停止而数据不足时，执行状态为 `PAUSED_RESOURCE_CAP`，科学结论按现有数据/精度落入 INCONCLUSIVE；资源超限本身不是科学 INVALID 或 NOT_SUPPORTED。

## 7. 多卡并行与资源安排：执行时必须落实

**做实验时，能多卡并行就多卡并行跑，优先使用可用 GPU 缩短总耗时。** 独立图像、模型和干预任务采用任务并行；纯 CPU 数学绘图/统计保持 CPU。Wall time、GPU-hours 是资源与优化指标，不作为科学有效性失败门。

EP1 先实现最小系统：固定 task manifest、一卡一 worker、SQLite 事务或原子文件 claim、每任务不可变结果、exact coverage audit、crash recovery、由 task ID 导出的 seed。复用仓库已有满足合同的队列能力；不在协议冻结前构建通用复杂调度器。优先级为 **数学/hook 正确性 → 任务语义 → 单卡基准 → 多卡吞吐优化**。

### 7.1 调度合同

1. 启动前记录 GPU UUID、空闲显存、利用率、既有 PID，按配置阈值选择可用设备，不干扰其他作业。每张可用卡一个 worker、常驻模型，弹性使用 1–N 卡，不为凑齐固定卡数等待。
2. 采用**全局动态任务队列与原子 claim**，不做静态 `index % K` 分片。Claim 含 task ID、worker、device、attempt、lease/heartbeat；lease 失效才回收任务，结果原子写入，无重复计数。
3. 优先任务粒度为“图像 × 模型 × 目标批 × 实验条件”，复用同图前向和预处理。Endpoint/IG 配对在同设备同配置执行；图像少时再拆目标批/独立干预。各 worker 重建自己的计算图，不跨进程共享活的 autograd graph。
4. IG 默认整条路径一个任务；按 alpha 分块须固定节点、权重、随机数和归约顺序，并核对单 worker 数值一致性。
5. Seed 由稳定 task ID 导出，不由卡号或完成顺序决定。任何 Torch/CUDA 导入前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；记录 `sys.executable` realpath、包版本、模型/数据哈希、dtype。
6. 按目标分块控制显存和存储。原始 signed RGB 图按协议范围保存，其余保存摘要/派生图与来源；绝对值或显示归一化图不替代原始归因。
7. OOM 按预定顺序缩小目标/图像 batch；不能暗改 IG 步数、精度、目标或模型。仅重试失败任务，保留首次失败和最终状态。

Claim 回收必须防止旧 worker 在失效后提交：每次领取有递增 attempt/fencing token，只有当前有效 token 可完成该 task；过期结果保留诊断但不计入正式完成集。单卡/多卡 exact coverage 同时检查任务遗漏、重复提交和旧 attempt 覆盖。

### 7.2 资源测量与并行验收

单任务 latency 在同设备可控负载下，预热并用 CUDA 同步界定起止；注明模型加载、I/O、分解和序列化是否计入。多卡吞吐另测，不能用 8 卡总吞吐与 1 卡单图 latency 比较算法加速。

记录总 wall time、worker 活跃时间、GPU-hours、峰值显存、利用率、任务长尾，以及单卡任务耗时/显存/输出字节的 P50、P95 和最大值。GPU-hours 按实际 worker 占用时间汇总，不假设所有卡全程忙碌。小批校准瓶颈后调整任务粒度/I/O，记录配置变化。

生产入口验收：1 卡/多卡任务集合一致，无遗漏/重复，seed 稳定；按 task ID 排序后指标在预定容差内一致；崩溃可续跑。正式测量使用冻结配置。

**资源 hard cap 是运行保护合同。** EP0 必须填写每阶段 wall-hours、aggregate GPU-hours、每任务时限/重试次数、总磁盘字节与最低空闲空间，以及在途任务的预留额度，当前均为 `TBD_EP0`。EP0 先依据已有可核查记录或明确资源配额设置 cap，EP1 测得单卡 P95 后在 cap 内细化任务预算；EP2 只在 EP0 允许的校准范围内确定 batch/粒度，不能提高已冻结总 cap。调度器在新 claim 前检查剩余额度，不足则停止派发；在途任务按预留时间完成或在预定安全点终止并落盘。

触及 cap 保存已完成结果、未完成清单和资源日志，记录 `PAUSED_RESOURCE_CAP`；不以已完成的有利子集替代全 cohort，也不自动扩大 cap。续跑或提高上限需要明确的前瞻性资源授权，保留原 cap 和停止记录。Wall time/GPU-hours 继续作为报告指标；科学出口按第 6.2 节判断。

### 7.3 当前代码与未来入口

[run_endpoint_vit_scan.py](../scripts/run_endpoint_vit_scan.py) 是 CPU 演示；[endpoint_vit.py](../src/vitnns/endpoint_vit.py) 是核心 API。EP1 需新增 GPU worker/队列入口并验证上述合同，目前尚未实现该生产入口。

设计伪代码（不是当前可运行 CLI）：

```text
read frozen protocol and registry
qualify and lock available GPUs; start one worker per device
worker: configure determinism -> load model once -> claim task
        -> compute paired methods / registered interventions
        -> validate -> atomic result write -> complete claim
coordinator: reclaim expired leases -> verify exact task coverage
             -> aggregate by image -> report resources and claims
```

## 8. 分阶段执行步骤

| 阶段 | 工作 | 产出 | 完成条件 | 并行部分 |
|---|---|---|---|---|
| EP0 冻结协议 | 固定目标、重构对象、预测量、统计出口、cohort、权限、资源 cap | protocol、registry、novelty 表、参数冻结表 | `TBD_EP0` 清零；不因此获得正式 outcome access | 文献/元数据/CPU 草图 |
| EP1 补实现 | energy、最小队列、unfused edge hooks、遗漏/重复记账反例 | 代码、回归、backend 与 worker parity | 第 8.2 节逐项通过；未支持 backend 明示 | 获授权工程小样本多卡；CPU fixtures |
| EP2 校准封存 | 仅用独立校准集确定容差/IG 步数/batch/粒度的最终值 | 校准报告、配置与 exact task manifest | 使用 EP0 预定算法/候选集，不改变统计、cohort、目标或总 cap | 获授权多卡校准任务 |
| EP3 扫描 | 冻结模型/输入/目标 | atlas、原始图、分解、资源日志 | 全覆盖、逐项校验 | 图像/模型/目标批动态并行 |
| EP4 输入评价 | 同目标方法与扰动 | fidelity、稳定性、成本 | 指标/NA 完整，配对一致 | 图像/扰动多卡 |
| EP5-A 局部合同 | 固定边、cotangent、$\eta$ 和 $J_{z_k,x}x$ | 方向导数误差与数值记录 | 仅支持 correctness，不计为 C4 科学证据 | 获授权独立样本/方向多卡 |
| EP5-B 有限反事实 | reference replacement；若登记则执行空间定向评价 | C4a/C4b、mandatory baseline 比较 | 操作/预算/预测量对应，无效果后选区 | 获授权独立图像/干预多卡 |
| EP6 统计写作 | 汇总 EP3–EP5 | claim decisions、正文/附录、复现包 | 图像级统计，数字可追溯 | CPU 统计/绘图 |

依赖：EP0 → EP1 → EP2 → EP3；之后 EP4、EP5-A、EP5-B 在资源及明确阶段授权允许时并行，所有启用任务完成后进入 EP6。EP1 已含局部合同 fixtures；EP5-B 在相应模型/边的合同通过后才能执行。实现违约先修复；科学结果为负则按原计划报告、收窄主张。

下一阶段路径为 `EP0_PROTOCOL_FREEZE → EP1_IMPLEMENTATION_AND_REGRESSION`；进入各阶段仍须核对当时有效的授权、模型/输入读权限和先决报告。本轮没有签发 EP0 PASS 或生产授权。

EP0 先固定 EP1 工程 fixtures 使用的 float64/float32、backend parity 及局部合同容差；若 EP2 需要按校准集调整生产数值参数，EP0 必须同时冻结调整算法、候选范围、失败出口与资源范围。不得在 EP1 测试失败后用 EP2 名义追认放宽容差。

### 8.1 EP0 必须冻结的清单

- [ ] 主 ground-truth logit、次级 frozen clean top-1 和错分图保留规则。
- [ ] 单元、sum、energy、head 的张量位置、token 集合和冻结方向。
- [ ] 每种 decomposition 的重构对象、target、edge、bypass 和输入相关 B。
- [ ] Canonical Endpoint / baseline-relative extension / Path IG 的名称和输入 seed。
- [ ] 原始 normalized RGB absolute mass、signed sum 与空间/显示聚合顺序。
- [ ] Atlas 的层、token、channel/head、固定抽样、near-zero 和归一化合同。
- [ ] C2 目标规模、成本分解、targets/sec、images/sec、peak memory、bytes/target。
- [ ] Fidelity 的 ranking、两种 replacement、insertion 起点、AUC 和 near-zero 处理。
- [ ] Flip/noise 的冻结类别、模型响应差、预测改变与受控分层。
- [ ] Reference donor/replacement、zeroing、联合效应与 OOD 条件。
- [ ] EP5-A correctness / EP5-B utility 分离；C4a/C4b 是否启用及主要预测量。
- [ ] 开发/校准/评价 cohort、各模型样本量、历史 20 图交集。
- [ ] 第 6.1 节所有数值、检验 family、mandatory baselines、NA 和确定性出口。
- [ ] 模型/checkpoint/preprocessing、unfused backend、fused parity 范围。
- [ ] 第 9 节 novelty 对照及支持的贡献叙事。
- [ ] 各阶段资源 hard cap 与暂停/恢复规则。
- [ ] 正式 outcome access 的授权来源、允许路径、阶段和禁止提前读取项。

### 8.2 EP1 必须通过的清单

- [ ] Paper-formula CNN 与独立 G×I parity，包括关闭门与非边界 winner。
- [ ] Energy target 实现，激活权重和 learning-image 目标绑定同一张量。
- [ ] All-block ViT parity，覆盖已注册的各目标种类。
- [ ] Residual false-pass fixture：故意用 $\nabla_uF$ 代替输入边 $g_r$，验证器必须拒绝重复计数。
- [ ] Head omission fixture：构造非零被删 head，验证器必须拒绝遗漏与错误 bypass。
- [ ] Routing/value omission fixture：构造非零 routing，验证器必须拒绝只传 Value。
- [ ] Q/K/V (+ 输入相关 B) 闭合与 branch/total 两种 reconstruction ID 校验。
- [ ] Fused/unfused forward parity；不能暴露边的 fused backend 不计结构支持。
- [ ] No-hook / no-op-hook / decomposition-hook forward parity。
- [ ] 有限差分/JVP、EP5-A 局部方向合同和 near-zero fixture。
- [ ] Float64/float32 容差与 absolute/relative 分支分别验证。
- [ ] Single-/multi-worker task/seed/result equivalence。
- [ ] Crash/lease 回收、过期 attempt 拒绝、immutable results、exact task coverage。
- [ ] 单卡 P95/最大资源与不同 K/M/C 的成本基准。
- [ ] 可用多卡吞吐与同任务覆盖验证，记录实际设备数及资源 cap。

Omission fixtures 必须先证实被删分量显著非零；不能仅在天然零 Query（如某些首层 CLS 目标）上测试后宣称捕捉遗漏。所有负例预期是明确的失败码，不能仅靠同一路径自比较得到零误差。

## 9. 文章结构、图表与写作顺序

| 章节 | 内容 | 图表 |
|---|---|---|
| Introduction | 内部观察问题、NNS 继承、ViT 结构 | 三方法身份图、真实输入到内部图总览 |
| Related Work | NNS、G×I/IG、conductance、Transformer attribution、head importance、activation/path patching | 目标/seed/路径/切面对照 |
| Formal Relationship | 命题、CNN 条件、Path 区分、坐标 | CNN 门控反例、Endpoint/IG 同目标图 |
| Endpoint Scanner | 注册、算子、实现、成本 | 上排前向/下排同名 learning image |
| Structured Decomposition | head/residual cuts、Q/K/V、闭合 | attention/residual 数学图 |
| Experiments | A–D 四组 | parity 表、atlas、成本质量曲线、干预比较 |
| Discussion / Limitations | 实测范围、局部饱和、坐标依赖、有限干预偏移 | 敏感性/失败样本 |

EP0 核对相关工作原始来源、公式、代码及目标，形成 `claim → source → technical difference` 表；只将已核对文献加入参考文献。本轮不填写未经检索确认的新颖性或接收预期。

| 本文对象 | 需检索的最近工作类别 | 必须逐项核对的差异 |
|---|---|---|
| 内部 Endpoint Scanner | internal G×I、saliency hooks、目标 atlas | 是否已有同样的目标注册、批量 atlas 与原始坐标输出 |
| Head edge decomposition | head attribution、gradient×activation | 是否保留输入坐标、含 downstream cotangent/bypass 并逐像素闭合 |
| Routing/value、Q/K/V | attention Jacobian decomposition | 是否已有相同的输入分辨微分图及 backend 合同 |
| Residual edge cut | residual attribution、LRP | 输入边防重复记账是否已提出；传播规则是否为同一个 Jacobian |
| Energy target | feature visualization、NNS aggregate | 是否只是原公式直接复现、张量位置是否一致 |
| 有限干预评价 | activation/path patching、spatial intervention | 目标、位移方向、预算、baseline 与新评价是否已覆盖 |

每行记录完整文献信息、原始 URL/页码/公式、代码版本、已覆盖项和仍未覆盖项；不能只检查公式是否相同。若 prior art 覆盖主要分解，论文定位转为“原始 NNS 的 ViT 复现、统一实现与系统实证审计”，相应收紧标题、贡献与摘要，不保留未成立的强方法创新叙事。

先写定义/实现，再写协议；实验后写 Results/Discussion；最后由实际主张写 Abstract/Introduction。干预若不优于简单基线，保留内部扫描、方法分析与成本质量定位；EP5-A 的高一致性不能补作 Utility 结果。

摘要起草句：

> 本文将 Neural Network Scanner 的终点 learning-image 原则适配到通过注册合同的视觉 Transformer block。对于内部标量目标 $s(x)$，NNS-ViT 在模型实际输入坐标上计算 $L_s(x)=x\odot\nabla_xs(x)$，通过统一目标注册以及 heads、routing/value、Q/K/V 和 residual 输入边的结构化分解组织内部扫描。成分在列明条件下重构所注册的分支或总目标归因。实验设计分别检查局部微分合同，并评价扫描成本、输入归因质量和有限内部干预。

最后一句在正式摘要中替换为 EP6 的具体结果与区间。

## 10. 本轮数学图与复现

[新图包](../reports/nns_vit_endpoint_math_comparison_r1_20260918/README.md) 承接 [旧 Path 图册](../reports/nns_vit_forward_learning_image_comparison_20260917/gallery.html) 的 $[1,1;3,3]$ 输入和 patch → attention → residual 解析链，增加 Endpoint、CNN 门控、Q/K/V、residual 边、目标聚合及 LN/GELU。

新图是解析示意，不是新增训练模型实验。每幅提供 PNG/PDF/SVG，图内英文便于论文排版，`Image #1` 起的中文说明记录公式、读图顺序、来源与范围。真实模型图沿用已有单图示例，正式结果图在 EP6 由全队列数据生成。

仓库根目录复现：

```bash
CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python \
  vitnns_rebuild/scripts/render_endpoint_math_comparison.py \
  --output-dir vitnns_rebuild/reports/nns_vit_endpoint_math_comparison_reproduction
```

输出目录必须不存在。脚本复算解析 Jacobian、有限差分、积分收敛、分解闭合与原图数值一致性，保存数值数据、SHA-256、验证记录及图册。

**历史图包来源版本。** R1 图包在 commit `f78c73eee4455e42ea0c8f3f4f786c6746878e7d` 生成，绑定的是本规划的 R1 字节。本 R2 修订不改写旧图包、原始数值、图像或 source hashes；直接用旧 `--verify` 对当前工作区做 source 检查会按合同报规划文件漂移。应以该 commit 中的源字节复核历史包，并把当前规划视为后续版本。复核记录见 [EP0 前修订与来源验证](NNS_ViT_Endpoint_Scanner_EP0前修订记录_20260918.md)。旧图只支持其解析公式和显示范围；EP0–EP5-B 新实验要求以本文 R2 为准。
