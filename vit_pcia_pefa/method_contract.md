# Endpoint–PCIA–PEFA 方法合同

- 状态：`FROZEN_STEP1_20260921`
- 冻结日：2026-09-21（PRC）
- 定位：`PORT_RESOLVED_ENDPOINT_FORMATION_SYSTEMATIZATION_CANDIDATE`
- 下一授权：`STEP2_REQUIRES_EXPLICIT_USER_REQUEST`

本文冻结步骤 2 及其后续可引用的方法对象。任何定义、路径、切面、精度、容差、样本角色或失败规则的变化都必须形成新的、前瞻性的修订，不能覆盖本合同。

## 1. 主张边界与步骤 1 决策

1. Endpoint 是参考点相对的 Gradient × Input，不是新基础归因。
2. PCIA 是 conductance/path-IG 在真实消费端口和合法切面上的系统化特例，不承担基础新颖性。
3. $T$ 是对端口 Endpoint 分量 $Q_{e,i}^b$ 做普通 path IG 的直接实例，也不承担基础新颖性。
4. $R$ 是加权 Hessian 路径项，$\Omega$ 是端口梯度通道的反对称导数；二阶导数、HVP、元归因或旋度运算本身均不新。
5. 冻结检索范围内尚未发现对以下联合对象的完整等价覆盖：原模型真实消费端口上的 $e\times i\times j$ Endpoint 形成张量；$R$ 对 Endpoint–IG 逐坐标缺口的端口分配；以及单端口 $\Omega$ 与完整合法切面零和的共同合同。
6. 因而步骤 1 的出口是 `GO_STEP2_COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE`，不是 `PEFA_FOUNDATIONAL_CANDIDATE`。若后续找到 $R$ 和 $\Omega$ 的完整先例，签发 `NO_GO_FOUNDATIONAL_NOVELTY` 并转为 Endpoint 系统化扩展。

允许的简短贡献表述是：

> 候选框架把已知的 Endpoint 与路径归因组合到原模型真实使用端口上，并检验 Endpoint–IG 差分的端口分辨曲率运输以及合法切面上的路由曲率相消。

禁止使用“全新基础归因”“首次使用路径积分/二阶导数/边归因”“已证明全球无先例”等措辞。

## 2. 数学对象

设 $F:\mathbb R^d\to\mathbb R$，固定参考输入 $b$，输入为 $x$，路径 $\gamma:[0,1]\to\mathbb R^d$ 分段 $C^1$ 且 $\gamma(0)=b,\gamma(1)=x$。

### 2.1 Endpoint

\[
E_i^b(x)=(x_i-b_i)\partial_iF(x).
\]

### 2.2 真实消费端口

端口 $e$ 是 producer value 在某次执行中送入特定 consumer 的一个 use，而不是共享 producer tensor 的名字。设该 use 收到的值为 $z_e$，则

\[
D_{e,i}(u)=\left[J_{z_e,u}(u)^\mathsf T\frac{\partial F}{\partial z_e}\right]_i,
\qquad
Q_{e,i}^b(u)=(u_i-b_i)D_{e,i}(u).
\]

同一 producer 的 fan-out 必须拆成不同 $e$；每个 $\partial F/\partial z_e$ 都是该 consumer use 的下游余切，禁止先在 producer 上聚合后复制。

### 2.3 PCIA 一阶项

\[
A_{e,i}^{\gamma}=\int_0^1\dot\gamma_i(t)D_{e,i}(\gamma(t))dt.
\]

### 2.4 PEFA 组合对象

\[
T_{e,i\leftarrow j}^{\gamma}
=\int_0^1\dot\gamma_j(t)
\partial_jQ_{e,i}^b(\gamma(t))dt,
\]

\[
R_{e,i\leftarrow j}^{\gamma}
=\int_0^1(\gamma_i(t)-b_i)\dot\gamma_j(t)
\partial_jD_{e,i}(\gamma(t))dt,
\]

\[
T_{e,i\leftarrow j}^{\gamma}=\delta_{ij}A_{e,i}^{\gamma}+R_{e,i\leftarrow j}^{\gamma}.
\]

### 2.5 端口路由曲率

\[
\Omega_{e,ij}(u)=\partial_iD_{e,j}(u)-\partial_jD_{e,i}(u).
\]

$\Omega$ 是结构诊断量，不自动等于路径依赖的输出效应或因果干预效应。

## 3. 成立条件与守恒对象

### 3.1 图和光滑性

- 从 $b$ 到 $x$ 的拓扑、consumer use、tensor shape、目标和参数必须固定；不允许数据依赖的拓扑切换。
- $F$ 与端口映射在积分节点处至少二阶可微。对 ReLU 等分段光滑 fixture，只接受几乎处处导数；Gauss–Legendre 节点若命中 kink，任务以 `FAIL_NONSMOOTH_NODE_HIT` 终止，不能抖动节点。
- 模型保持 `eval()`；dropout、drop path 和随机算子关闭。参数、buffer、位置编码、归一化统计和 bias 都是固定模型状态。
- 一阶 $D$ 的 VJP 在单次求值中使用指定 use 的余切；计算 $\partial_jD_{e,i}$ 时必须 `create_graph=True` 并保留 Jacobian 与下游余切对 $u$ 的全部依赖。对 cotangent 或中间值使用 `detach`/`stop_gradient` 属 `FAIL_SECOND_ORDER_GRAPH_TRUNCATED`。

### 3.2 合法切面

切面 $C$ 合法，当且仅当冻结图中每条从所有变化根到 $F$ 的有效有向路径与 $C$ 恰好相交一次。必须以路径枚举或独立图检查证明，不能因数值近似闭合就反推合法。合法时：

\[
\sum_{e\in C}D_{e,i}=\partial_iF,\qquad
\sum_{e\in C}Q_{e,i}^b(x)=E_i^b(x),
\]

\[
\sum_eA_{e,i}^{\gamma}=\operatorname{IG}_i^{\gamma}(F),\qquad
\sum_i\sum_eA_{e,i}^{\gamma}=F(x)-F(b),
\]

\[
\sum_jT_{e,i\leftarrow j}^{\gamma}=Q_{e,i}^b(x),\qquad
\sum_e\sum_jT_{e,i\leftarrow j}^{\gamma}=E_i^b(x),
\]

\[
\sum_e\sum_jR_{e,i\leftarrow j}^{\gamma}
=E_i^b(x)-\operatorname{IG}_i^{\gamma}(F),\qquad
\sum_e\Omega_{e,ij}=0.
\]

每条等式先在未聚合、signed、normalized RGB `[3,224,224]` 坐标或 CPU fixture 的原始坐标上验证。patch/head/layer/空间图只可在通过后派生。

## 4. 冻结的根、坐标、目标、参考与路径

### 4.1 变化根变量

唯一变化根是模型实际接收的 normalized RGB 张量 $u\in\mathbb R^{3\times224\times224}$。模型参数、buffers、位置编码、类别索引、预处理常量和所有随机状态固定；它们的路径导数为零。内部 activation 是 $u$ 的派生量，不是独立插值根。EAP 风格的 activation-space clean/corrupted 路径不是本合同的 $\gamma$。

### 4.2 输入坐标

$i,j$ 采用 C-major 的 `[channel,row,column]` 标量坐标，线性索引为 `(c*224+h)*224+w`。主计算保存 signed tensor，不先做绝对值、正部、通道和、patch pooling 或逐图归一化。normalized RGB 到 raw RGB 的仿射变换只用于解释，不改变主坐标。

### 4.3 标量目标 $F$

- 主目标：对每个注册图像与模型的官方 ground-truth class raw logit。误分类图保留。
- 次级目标：同一 clean forward 冻结的 top-1 class raw logit；不可替代主目标。
- 不使用 softmax probability、margin、loss 或后验挑选的 neuron 作为步骤 2–5 主目标。
- CPU fixture 的 $F$ 在步骤 2 必须至少包含线性、双线性、平滑饱和、fan-out、residual 和多输入 DAG；解析参数与输入在创建实现前写入 fixture registry，之后不得按结果调整。

### 4.4 参考 $b$ 与路径 $\gamma$

- ViT 主参考固定为 normalized-zero：$b=0$，物理上对应每模型预处理的 RGB mean。
- 主路径固定为直线 $\gamma(t)=b+t(x-b)$，$\dot\gamma=x-b$。
- 不把 blur、其他图像 donor、activation replacement 或逐样本优化路径混入主恒等式。它们若用于以后干预，只能作为另行冻结的评价条件。
- 路径不得按样本、模型收敛结果或效应方向更改。

## 5. 端口 ID 与冻结切面族

### 5.1 端口 ID

每个端口必须序列化为以下封闭元组：

```text
(model_id, graph_revision, block_index, producer_tensor_id,
 consumer_operator_id, input_slot, invocation_index, slice_spec)
```

- `producer_tensor_id` 是一次 forward 中的 SSA 风格 tensor identity，不是易碰撞的模块显示名；
- `consumer_operator_id` 含完整模块/算子路径与 operator kind；
- `input_slot` 从 0 开始，区分 add、matmul、linear 等多输入；
- `invocation_index` 从 0 开始，区分模块复用；
- `slice_spec` 只允许显式、互斥、完备的 token/head/channel slice；无切片写 `ALL`。

任何字段缺失、fan-out use 合并或动态调用次数变化均为 `FAIL_PORT_ID_AMBIGUOUS`。

### 5.2 必须实现和证明的切面族

| ID | 端口集合 | 重构对象 | 备注 |
|---|---|---|---|
| `RESIDUAL1_FULL` | `u -> add1.slot0`，`attention_out -> add1.slot1` | 完整 $F$ | 两条 use 均要记录 |
| `HEADS_WITH_BYPASS_FULL` | residual identity use，加 `attn.proj.slot0` 的全部互斥 head slices | 完整 $F$ | 少/重一个 head 都失败 |
| `QKV_WITH_BYPASS_FULL` | residual identity use，加 Q、K、V 各自真实 consumer use | 完整 $F$ | Q/K 经过 score/softmax，V 经过 AV；不得把 routing 与 Q、K 重复相加 |
| `ROUTING_VALUE_WITH_BYPASS_FULL` | residual identity，`routing=Q+K`，Value | 完整 $F$ | 与上一行是替代切面，不可跨行相加 |
| `RESIDUAL2_FULL` | `r1 -> add2.slot0`，`mlp_out -> add2.slot1` | 完整 $F$ | 覆盖 MLP branch |
| `MLP_INTERNAL_WITH_CARRY_FULL` | residual carry use，加冻结 MLP 内部的完备 use-port cut | 完整 $F$ | 具体 fc1/GELU/fc2 use 必须在步骤 4 registry 闭合后启用 |

attention-only 的 `HEADS_BRANCH`、`QKV_BRANCH` 或 `ROUTING_VALUE_BRANCH` 只能重构 attention branch $A_F$，不得标记为全局合法切面。不同 block、不同切面或替代分解不可互相求和。

## 6. 积分规则与数值合同

### 6.1 固定求积

- 规则：闭区间 `[0,1]` 上 Gauss–Legendre，使用 `numpy.polynomial.legendre.leggauss` 生成 `[-1,1]` 节点后按 $t=(s+1)/2,w'=w/2$ 映射；节点、权重、顺序在首次运行前写入 registry 并 SHA-256 绑定。
- CPU 步骤 2 主结果：固定 64 节点；另以固定 128 节点作预先声明的收敛诊断。64 节点 gate 不能因失败改成逐 fixture 自适应。
- ViT 步骤 4–5 主结果：固定 512 节点；固定 256 节点仅作收敛诊断，不参与挑选样本或改写 512 节点结果。既有 Endpoint EP2 已表明 ViT-B 的 path-IG 在 512 节点下未通过旧校准门，因此该模型上的失败风险是已知边界，不得放宽阈值规避。
- 任何新节点数、Monte Carlo、trapezoid 或自适应 quadrature 都需前瞻性新修订。

### 6.2 CPU 解析 fixture 数据

步骤 2 的主输入和参数现在一并冻结；名称、顺序、$b$、$x$ 或系数均不得按测试结果修改。所有向量按下列书写顺序索引，路径均为 $b+t(x-b)$。

| ID | $b$ | $x$ | 冻结标量图 | 必须检查的结构 |
|---|---|---|---|---|
| `LINEAR3` | $(0,0,0)$ | $(1.25,-0.5,2)$ | $F=1.5u_1-2u_2+0.25u_3$ | $R=0,\Omega=0$ |
| `BILINEAR2` | $(0,0)$ | $(2,-3)$ | $F=u_1u_2+0.2u_1$ | 非零 off-diagonal $T/R$；解析 Endpoint–IG 差 |
| `SATURATING3` | $(0.1,-0.2,0.3)$ | $(1.2,0.7,-0.4)$ | $F=\tanh(0.7u_1-1.1u_2+0.5u_3+0.2)$ | 平滑非线性与 quadrature 收敛 |
| `FANOUT2` | $(0,0)$ | $(1.1,-0.8)$ | $h=0.6u_1-0.4u_2$；$a=2h$；$c=\tanh(h)$；$F=a+0.7c$ | `h->a` 与 `h->c` 是不同 use；合并、漏项、重复均失败 |
| `RESIDUAL2` | $(0,0)$ | $(0.9,-1.3)$ | $h=\tanh(Wu+q)$；$r=u+Bh$；$F=v^Tr$，其中 $W=[[0.8,-0.3],[0.2,0.9]],q=(0.1,-0.2),B=[[0.7,0.1],[-0.4,0.6]],v=(1.2,-0.5)$ | identity/branch 完整切面 |
| `MULTIINPUT3` | $(0,0,0)$ | $(0.8,-1.1,0.6)$ | $a=u_1+2u_2$；$c=0.5u_2-u_3$；$m=a\,c$；$F=m+0.3a-0.2c$ | multi-input slot 0/1 不可交换 |

另冻结两个只用于负控制的一维非光滑 fixture，均使用 $b=(-1),x=(1)$：

- `RELU_SAFE`：$F(u)=\operatorname{ReLU}(u_1)$。64 点 Gauss–Legendre 不含 $t=0.5$；最近两节点为 $t_{31}=0.48782485366828776$ 与 $t_{32}=0.5121751463317122$（零起索引），故所有节点距 kink 的输入距离至少 `0.024350292663424478`。
- `RELU_HIT`：$F(u)=\operatorname{ReLU}(u_1-u_*)$，其中 $u_*=-1+2t_{31}=-0.024350292663424478$。实现必须用同一份 float64 节点数组和表达式 `-1.0 + 2.0*t[31]` 构造 $u_*$；在求导前检测该节点命中并签发 `FAIL_NONSMOOTH_NODE_HIT`，不得采用框架在零点约定的次梯度绕过。

两组函数、节点索引、十进制常数及构造表达式均在步骤 1 冻结；步骤 2 只可把实际节点/权重的 float64 bytes 写入 registry 并验证上述值，不再选择系数或命中节点。

### 6.3 精度

- CPU fixture：模型、自动微分、节点、权重与累加全部 float64。
- ViT：冻结 checkpoint 的 float32 forward/一阶和二阶自动微分；TF32 与 autocast 关闭；所有积分与残差 reduction 转 float64，用 `math.fsum` 或确定性等价实现。
- 不显式物化完整 $d\times d$ Hessian；只允许 JVP/VJP/HVP 与流式 patch-group 计算。该实现约束不能改变数学对象。

### 6.4 闭合残差与通过条件

对 observed $o$ 和 reference $r$ 同时报：

\[
r_{\max}=\max|o-r|,\qquad
r_{\mathrm{rel}}=\frac{\lVert o-r\rVert_2}{\max(\lVert r\rVert_2,\epsilon)}.
\]

| 域 | `atol` | `rtol` | `near_zero` |
|---|---:|---:|---:|
| CPU float64 | $10^{-9}$ | $10^{-7}$ | $10^{-12}$ |
| ViT float32 autodiff + float64 reduction | $10^{-5}$ | $10^{-4}$ | $10^{-6}$ |

每个原始坐标必须满足 `abs_error <= atol + rtol*abs(reference)`；非近零 reference 还须 `relative_l2 <= rtol`，近零则须 `max_abs <= atol`。所有值必须有限。五类主等式和每个适用负控制全部通过才是 PASS；平均误差、展示图相似或聚合后闭合不能替代。

## 7. 模型与数据冻结

### 7.1 模型

沿用且只允许以下两个已绑定的标准 timm unfused `eval()` 模型：

| ID | timm 名称 | checkpoint SHA-256 | 角色 |
|---|---|---|---|
| `vit_b16` | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | `489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d` | 主模型 |
| `deit_s16` | `deit_small_patch16_224.fb_in1k` | `cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f` | 复现模型 |

支持范围只到这两个注册实现；不外推到所有 ViT、fused attention、SwiGLU LLM 或含动态控制流的图。

### 7.2 数据

- 容器固定为 `vitnns_rebuild/assets/imagenet/ILSVRC2012_img_val.tar`，SHA-256 `c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0`。
- 成员与角色固定为 EP0 `cohort_registry.json` 的 528 行：64 discovery、64 calibration、400 evaluation；模型共享相同图像行，不允许缺失时替换。
- 步骤 2 不读图像。步骤 4 只可用 discovery split 做实现和成本诊断。步骤 5 pilot 只可从 calibration split 按 `split_order` 前缀选择，具体前缀长度须在步骤 5 outcome 读取前另行冻结。evaluation split 保留给通过 pilot 后的步骤 6。
- 旧 Endpoint 已使用 calibration/evaluation 数据；这不构成 PEFA 结果，但属于已知历史暴露。不得根据旧 C4a/C4b 结果筛图、换目标、改阈值或声称全新盲测。

预处理严格沿用 EP0：RGB；短边 resize 248，bicubic 且 antialias；center crop 224；各模型使用其 registry 中 mean/std；不做 EXIF transpose。

## 8. 随机性与可复现性

- CPU 解析 fixture 默认无随机性；如负控制需要随机张量，seed 为 `int.from_bytes(SHA256("ENDPOINT_PCIA_PEFA|step|task_id")[:8], "big")`。
- ViT seed 同一规则且与 worker/GPU 数无关；记录 Python、NumPy、Torch CPU/CUDA RNG state。
- 在导入 Torch/CUDA 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；启用 deterministic algorithms；TF32、autocast、dropout、drop path 关闭。
- 记录 `sys.executable` realpath、包版本、CUDA/driver、GPU UUID、git commit、dirty status、模型与输入 hashes。
- 动态全局任务队列使用 lease/heartbeat/fencing token；不使用 `index % K` 静态分片。

## 9. 资源边界

预算是停止/收缩门，不是科学失败，也不自动授权后续步骤。

| 步骤 | GPU 权限 | aggregate H20-hours cap | wall-hours cap | 新增保留 GiB cap |
|---|---:|---:|---:|---:|
| 1 查新与合同 | 0 | 0 | 8 | 0.25 |
| 2 CPU reference | 0 | 0 | 12 | 2 |
| 3 理论链 | 0 | 0 | 12 | 1 |
| 4 ViT 适配/成本门 | 明确授权后 1–8 H20 | 16 | 24 | 32 |
| 5 冻结 pilot | 明确授权后 1–8 H20 | 64 | 72 | 128 |
| 6 正式生产 | 未冻结 | 未冻结 | 未冻结 | 未冻结 |

H20 资格：空闲显存至少 32768 MiB、利用率不高于 10%，记录已有 PID，不终止他人进程。每 worker GPU memory cap 24 GiB、CPU RSS cap 32 GiB、每 task 1800 秒、最多 2 次 attempt、开跑前磁盘至少 50 GiB。触 cap 时停止新 claim 并签发 `PAUSED_RESOURCE_CAP`。

## 10. 失败类型与不可变性

| Code | 条件 | 后果 |
|---|---|---|
| `NO_GO_FOUNDATIONAL_NOVELTY` | $R$ 与 $\Omega$ 的完整合同均被现有工作覆盖 | 转为 Endpoint 系统化扩展，不制造新名 |
| `FAIL_SOURCE_BINDING` | commit、文件、模型、数据或 registry hash 不一致且无前瞻修订 | 当前 root 终止 |
| `INVALID_OUTCOME_BLINDNESS` | outcome 后改定义、样本、阈值、节点或目标 | 当前科学结论无效 |
| `FAIL_PORT_ID_AMBIGUOUS` | fan-out、slot、invocation 或 slice 无法唯一标识 | 不计算端口量 |
| `FAIL_CUT_NOT_LEGAL` | 漏路径、重复路径或无法证明一次相交 | 该切面不可用 |
| `FAIL_SECOND_ORDER_GRAPH_TRUNCATED` | 二阶依赖被 detach/stop-gradient 截断 | 实现失败 |
| `FAIL_NONSMOOTH_NODE_HIT` | 求积节点命中未定义二阶导点 | fixture/task 失败，不抖动节点 |
| `FAIL_NONFINITE` | 任一原始量、梯度、HVP 或残差非有限 | task 失败 |
| `FAIL_QUADRATURE_CLOSURE` | 任一冻结恒等式超容差 | 当前阶段失败 |
| `FAIL_NEGATIVE_CONTROL` | 任一负控制未触发指定失败 | 当前阶段失败 |
| `FAIL_PORT_FIDELITY` | 只有不可审计聚合近似可闭合 | ViT 适配失败 |
| `NO_GO_COST` | 冻结实现无法在预算内完成 | 收缩坐标分辨率或转数学/系统论文 |
| `NOT_SUPPORTED_EMPIRICAL_VALUE` | 闭合但无基线增量或干预预测力 | 不升格为主线方法 |
| `PAUSED_RESOURCE_CAP` | 资源 cap 即将或已经触发 | 保留完成集，等待显式修订 |

失败 receipt 不可改写；恢复必须新建 root 并引用原失败。环境/资产失败不得表述为科学不成立。

## 11. 步骤 2 入口与出口

步骤 1 只开放步骤 2 的申请资格：

```text
STEP1_STATUS=PASS_CONTRACT_FROZEN
NOVELTY_POSITION=COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE
PCIA_ROLE=CONDUCTANCE_EQUIVALENT_PORT_BASELINE
T_ROLE=PATH_IG_OF_PORT_ENDPOINT_COMPONENT
NEXT_AUTHORIZATION=STEP2_REQUIRES_EXPLICIT_USER_REQUEST
STEP3_TO_STEP6_NOT_AUTHORIZED
```

步骤 2 必须独立实现解析 fixture、五类恒等式和遗漏/重复 residual、fan-out 聚合、input-slot 交换、串联切面重复、错误 detach、仅取 Hessian 对角等负控制。步骤 1 PASS 不代表任何数学、实现、端口忠实性、成本或实证价值已经 PASS。
