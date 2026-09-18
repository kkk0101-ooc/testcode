# NNS-ViT Endpoint Scanner：论文思路、规划与执行步骤

日期：2026-09-18。本文规划一篇以**终点内部扫描与结构化分解**为主线的独立论文。当前已完成方法分析、单图实现验证和数学示意图；下文 EP0–EP6 是待执行计划，阶段名只用于本论文，不替代仓库已有阶段授权。

入口：[数学分析与方法对比](NNS_ViT_Endpoint_Scanner数学分析与方法对比_20260918.md) · [实现与用法](endpoint_nns_vit_scanner_20260918.md) · [真实单图 Image #1–#9](endpoint_nns_vit_scanner_figures_20260918.md) · [本轮数学图及说明](../reports/nns_vit_endpoint_math_comparison_r1_20260918/figure_analysis.md) · [图册](../reports/nns_vit_endpoint_math_comparison_r1_20260918/gallery.html)。

## 1. 论文定位与研究问题

建议标题：**NNS-ViT: Endpoint Scanning and Structured Decomposition of Vision Transformers**。

中文：**NNS-ViT：视觉 Transformer 内部响应的终点扫描与结构化分解**。

中心问题：能否把 NNS 的逐神经元、保留输入坐标的终点观察方式推广到 ViT，并用能够闭合的结构成分描述 head、routing/value 和 residual 对同一目标的局部输入依赖？文章以“内部响应 → 输入坐标贡献 → 匹配干预评价”为证据链。

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

前三项仍需相关工作核对以确定新增程度。G×I、链式法则和一阶加法本身不作为新算法贡献。干预预测优势须在 EP5 得到结果后再写进摘要。

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

类别研究预先规定 ground-truth class 或 clean top-1；若用 clean top-1，在同图所有扰动、翻转和方法之间保持类别不变。内部方向在 discovery 数据上冻结。单元自身的图与其对最终类别的成分图分栏报告。

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

Softmax Jacobian 按 query row 作用。固定位置 bias 满足 $dB=0$；输入相关 bias 增加成分。完整传播还包括 concat、$W_O$、其 bias 的下游作用及后续模块。固定 `eval()`，记录 fused/unfused attention 与精度。

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

| 类型 | 成分 | 重构范围 |
|---|---|---|
| computational edge cuts | identity + attention；carry + MLP | 各自独立重构总目标图 |
| multi-head edge cut | identity + 所有 head slices | 含 bypass、projection、下游 |
| attention differential paths | identity + Q + K + V (+ input-dependent B) | 固定 block 和目标 |
| attention differential paths | identity + routing + V | routing = Q + K (+ B) |

不同 block、切面不能混加；routing 也不能与 Q/K 重复相加。此前 path 成分使用对称形式

$$
K^C_{k,i}=\Delta x_i\int_0^1
\left\langle\nabla_{z_k}F(x_\alpha),\frac{\partial z_k(x_\alpha)}{\partial x_i}\right\rangle d\alpha.
$$

完整切面条件须沿路径成立。它是 input-resolved path conductance，Endpoint 是单点 Jacobian decomposition。

### 3.6 坐标与完整性

若 $x=(r-\mu)/\sigma$，则 $x_i\partial_{x_i}s=(r_i-\mu_i)\partial_{r_i}s$。论文以 coordinate-origin dependence 描述该性质。Canonical Endpoint 没有 reference forward；raw-RGB seed 和 normalized-input seed 是不同分析。IG 的 affine-coordinate consistency 要求 baseline 同步变换并使用同一物理路径。一般 $\sum_iL_s[i]\ne s(x)$，也没有 baseline-difference completeness。

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

分别报告单元图、energy 图、head 内部图和固定类别结构图。跨图保持目标语义；不同模型同编号 channel 不自动语义对齐。冗余主张预定义相似度、零质量处理和每图汇总。Atlas 展示现象，干预评价其行为意义。

### C：Endpoint 与 Path IG 的成本—质量

主比较使用 canonical Endpoint 与 $x_0=0$ 的完整 internal-target IG，保持 seed 相同。多参考 IG、baseline-relative endpoint 单列。固定目标、输入、精度、batch、后处理，IG 步数先校准收敛。

测量每图每目标 latency、峰值显存、每卡吞吐和队列 wall time；4×4/native 16×16 insertion/deletion；flip/noise 稳定性。EP0 固定 fidelity 定义、替换值、排序和方向；legacy combined fidelity 与新 AUC 不混为同一指标。

Flip 先 inverse-warp 图；空间单元同时变换对应 token 索引并说明位置编码影响，类别目标保持 clean 类别。噪声强度和裁剪空间固定。

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

局部一致性子实验令 $v_k=J_{z_k,x}x$，在注册边上置 $z_k(\eta)=z_k-\eta v_k$，其他切面边固定 clean，重算下游：

$$
\lim_{\eta\to0}\frac{F(x)-F(z_k-\eta v_k)}{\eta}=a_k.
$$

冻结递减 $\eta$ 集合，报告近似误差及数值噪声。它检验 Endpoint 对齐的方向导数；完整 reference replacement 则评价有限变化，两者分开报告。

| 干预族 | 位置与操作 | 控制 |
|---|---|---|
| head | concat 前或等价投影前切片替换，重算投影/下游 | bypass clean |
| routing | $O=AV$ 处替换 $A$，$V$ clean | 同模型、同 token 布局 |
| value | 替换 $V$，$A$ clean | 同一 reference 配对规则 |
| Q/K | 各自替换后重算 softmax | 与直接替换 $A$ 分列 |
| attention/MLP branch | 加法器 branch 输入边替换 | identity/carry clean |

Reference 来自预注册参考输入或配对样本；zeroing 另设稳健性条件。干预 reference 不改变 Endpoint 定义。有限 routing/value 联合效应可含交互，不能由微分闭合推导 $e_{R,V}=e_R+e_V$；报告联合效应和交互残差。

结构基线至少包括 output norm、gradient norm、gradient×activation、$p_k^{\rm ref}$、随机排序；attention 另加 CLS attention/entropy/rollout。若 leave-one-head-out effect 定义被预测的 $e_k$，它是评价标签/代价参照，不再作为免费 predictor 排名。

## 6. 统计、样本与主张判定

独立单位为图像。每图先在注册成分集合 $\mathcal K_n$ 内计算相关、top-k enrichment、方向一致率，再按图像汇总：

$$
\rho_n=\operatorname{Spearman}_{k\in\mathcal K_n}(m_{n,k},|e_{n,k}|).
$$

Routing/value 只有两个成分时，单 block Spearman 是退化排序。预定跨 block 成分集合后每图汇总，或采用成对排序正确率和效果误差。零方差/零质量相关记 NA，报告有效图像数；方向一致率预定 near-zero 阈值。

Paired image-level bootstrap 给出方法差的区间；报告 median、有效样本数及分布。Head、block、重复随机种子不是独立样本。跨模型分开报告或按图像 cluster bootstrap；多主张/多族比较预先登记校正方式（如 Holm）。描述性 atlas 与确认性主张分开。

已有 20 图作为历史探索材料；另设开发/校准集与不重叠评价集。样本量由冻结的最小有意义差异、区间精度或功效设计确定，EP0 写入具体数量和理由；不根据正式评价结果扩样或降标准。主模型使用现有绑定 checkpoint，第二模型在资产/算子合同核对后预先选定；泛化主张需要其真实结果。

| 主张 | 证据 | 写法 |
|---|---|---|
| C1：原理适配与实现正确 | 命题 + CNN/ViT parity/闭合 | 明确算子和模型范围 |
| C2：可扩展扫描 | 全注册覆盖 + runtime/memory | 报告规模；有限目标不称全神经元 |
| C3：成本—质量关系 | 同目标配对实验 | 报告实际收益与损失 |
| C4：结构图预测干预 | 相对简单基线的预定指标差与区间 | 支持则写具体能力，否则保留观察定位 |

输出 `SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / INVALID` 及对应指标。实现正确和科学主张分别判定；历史负结果进入讨论，结论范围由新证据决定。

## 7. 多卡并行与资源安排：执行时必须落实

**做实验时，能多卡并行就多卡并行跑，优先使用可用 GPU 缩短总耗时。** 独立图像、模型和干预任务采用任务并行；纯 CPU 数学绘图/统计保持 CPU。Wall time、GPU-hours 是资源与优化指标，不作为科学有效性失败门。

### 7.1 调度合同

1. 启动前记录 GPU UUID、空闲显存、利用率、既有 PID，按配置阈值选择可用设备，不干扰其他作业。每张可用卡一个 worker、常驻模型，弹性使用 1–N 卡，不为凑齐固定卡数等待。
2. 采用**全局动态任务队列与原子 claim**，不做静态 `index % K` 分片。Claim 含 task ID、worker、device、attempt、lease/heartbeat；lease 失效才回收任务，结果原子写入，无重复计数。
3. 优先任务粒度为“图像 × 模型 × 目标批 × 实验条件”，复用同图前向和预处理。Endpoint/IG 配对在同设备同配置执行；图像少时再拆目标批/独立干预。各 worker 重建自己的计算图，不跨进程共享活的 autograd graph。
4. IG 默认整条路径一个任务；按 alpha 分块须固定节点、权重、随机数和归约顺序，并核对单 worker 数值一致性。
5. Seed 由稳定 task ID 导出，不由卡号或完成顺序决定。任何 Torch/CUDA 导入前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；记录 `sys.executable` realpath、包版本、模型/数据哈希、dtype。
6. 按目标分块控制显存和存储。原始 signed RGB 图按协议范围保存，其余保存摘要/派生图与来源；绝对值或显示归一化图不替代原始归因。
7. OOM 按预定顺序缩小目标/图像 batch；不能暗改 IG 步数、精度、目标或模型。仅重试失败任务，保留首次失败和最终状态。

### 7.2 资源测量与并行验收

单任务 latency 在同设备可控负载下，预热并用 CUDA 同步界定起止；注明模型加载、I/O、分解和序列化是否计入。多卡吞吐另测，不能用 8 卡总吞吐与 1 卡单图 latency 比较算法加速。

记录总 wall time、worker 活跃时间、GPU-hours、峰值显存、利用率、任务长尾。GPU-hours 按实际 worker 占用时间汇总，不假设所有卡全程忙碌。小批校准瓶颈后调整任务粒度/I/O，记录配置变化。

生产入口验收：1 卡/多卡任务集合一致，无遗漏/重复，seed 稳定；按 task ID 排序后指标在预定容差内一致；崩溃可续跑。正式测量使用冻结配置。

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
| EP0 冻结协议 | 核对相关工作/资产/既有证据；冻结目标、划分、指标、主张、容差、调度 | protocol、目标/干预 registry、证据表 | 参数齐全；正式数据访问服从相应既有授权 | 文献/元数据/CPU 草图 |
| EP1 补实现 | energy、GPU worker/队列、批量策略、edge hooks | 代码、回归、1卡/多卡一致性记录 | 原始图 parity、闭合、恢复测试通过 | 独立小样本多卡；CPU 算子测试 |
| EP2 校准封存 | 独立开发集校准容差、IG 步数、batch、任务粒度 | 校准报告、最终配置/任务清单 | 评价选择在正式结果前确定 | 多卡校准任务 |
| EP3 扫描 | 冻结模型/输入/目标 | atlas、原始图、分解、资源日志 | 全覆盖、逐项校验 | 图像/模型/目标批动态并行 |
| EP4 输入评价 | 同目标方法与扰动 | fidelity、稳定性、成本 | 指标/NA 完整，配对一致 | 图像/扰动多卡 |
| EP5 内部干预 | 固定目标/边/reference | effect、局部检验、基线比较 | 操作落在注册边，成分/效果对应 | 独立图像/干预多卡 |
| EP6 统计写作 | 汇总 EP3–EP5 | claim decisions、正文/附录、复现包 | 图像级统计，数字可追溯 | CPU 统计/绘图 |

依赖：EP0 → EP1 → EP2 → EP3；之后 EP4/EP5 在资源与冻结协议允许时并行；两者完成后进入 EP6。实现违约先修复；科学结果为负则按原计划报告、收窄主张。

建议第一轮实施 EP0–EP1：确认 novelty/协议，补 energy、CNN 回归和多卡入口。正式实验由 EP2 前冻结的 registry 决定。本文交付不启动这些 GPU 阶段。

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

先写定义/实现，再写协议；实验后写 Results/Discussion；最后由实际主张写 Abstract/Introduction。干预若不优于简单基线，保留内部扫描、方法分析与成本质量定位。

摘要起草句：

> 本文将 Neural Network Scanner 的终点 learning-image 原则适配到视觉 Transformer。对于内部标量目标 $s(x)$，NNS-ViT 在模型实际输入坐标上计算 $L_s(x)=x\odot\nabla_xs(x)$，通过统一目标注册以及 heads、routing/value、Q/K/V 和 residual 输入边的结构化分解组织内部扫描。成分在列明条件下重构同一目标的终点归因。实验报告扫描成本、输入归因质量和匹配内部干预评价。

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

输出目录必须不存在。脚本复算解析 Jacobian、有限差分、积分收敛、分解闭合与原图数值一致性，保存数值数据、SHA-256、验证记录及图册。独立复核命令见图包 README。
