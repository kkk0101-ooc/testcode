# Endpoint–PCIA–PEFA 步骤 3 门禁决定

- 状态：`PASS_THEORY_AND_SYMBOLIC_CONSISTENCY`
- 日期：2026-09-21（PRC）
- 范围：理论推导与 CPU 符号核验；未读取 JPEG、未加载 ViT、未使用 GPU。

## 决定

```text
STEP3_STATUS=PASS_THEORY_AND_SYMBOLIC_CONSISTENCY
THEORY_SCOPE=FROZEN_C2_GRAPH_AND_PIECEWISE_C1_PATH
CORE_IDENTITIES_PROVED=4/4
REPRESENTATION_CASES_RESOLVED=7/7
SYMBOLIC_FIXTURES_PASS=6/6
CONJECTURES_USED_AS_CONCLUSIONS=0
NO_GO_GENERALITY=false
SCIENTIFIC_EFFECT_CLAIMS=UNTESTED
STEP4_REQUIRES_EXPLICIT_USER_REQUEST
STEP4_TO_STEP6_NOT_STARTED=true
```

步骤 3 的出口门满足。`theory_endpoint_pcia_pefa.md` 在固定有限 DAG、分离 use-port、合法完整切面、路径邻域 $C^2$ 和分段 $C^1$ 路径的范围内证明：逐端口/逐 Endpoint 坐标闭合、$T=\delta A+R$、Endpoint–IG 逐坐标差分恒等式、完整切面 $\Omega$ 零和。核心证明不要求直线路径；只有加权 Hessian 的直线简式要求 $\gamma(t)=b+t(x-b)$。

这组图限制是端口分解可定义所需的范围，不阻断冻结的 smooth CPU fixtures 或计划中的固定 `eval()` ViT 图，因此 `NO_GO_GENERALITY=false`。跨 kink、动态图切换或无法唯一分离端口的情况明确排除；不能用几乎处处 Hessian、数值闭合或聚合展示替代这些前提。

项目定位保持 `COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE`。PCIA 是 conductance 特化，$T$ 是对端口 Endpoint 分量做普通 path IG，$R$ 属于加权 Hessian 路径项，$\Omega$ 是标准反对称导数。候选增量限于“真实使用端口 × 来源坐标 × Endpoint 坐标”的联合合同及其切面关系，实证效用保持未检验。

## 理论结论摘要

| ID | 结论 | 证明位置 | 状态 |
|---|---|---|---|
| TH1 | $\sum_jT_{e,i\leftarrow j}=Q_{e,i}^b(x)$，合法切面再对 $e$ 求和得到 $E_i^b(x)$ | 理论文档定理 1 | `PROVED` |
| TH2 | $T_{e,i\leftarrow j}=\delta_{ij}A_{e,i}+R_{e,i\leftarrow j}$ | 理论文档定理 2 | `PROVED` |
| TH3 | $E_i^b-\operatorname{IG}_i=\sum_{e,j}R_{e,i\leftarrow j}$ | 理论文档定理 3 | `PROVED` |
| TH4 | $\sum_e\Omega_{e,ij}=0$ | 理论文档定理 4 | `PROVED` |
| TH5 | 仿射图/常量端口场有 $R=\Omega=0$ | 理论文档命题 5 | `PROVED` |
| TH6 | 一维输入没有反对称曲率 | 理论文档命题 6 | `PROVED` |
| TH7 | 完整模型梯度场的总 $\Omega$ 为零 | 理论文档命题 7 | `PROVED` |
| EMP1 | $R/\Omega$ 能否预测 Endpoint 变化或端口干预 | 步骤 5 才能检验 | `UNTESTED` |

## 等价与表征边界

- PCIA 对标量隐藏 use 数值等价于输入分辨 conductance；端口注册和合法切面是系统合同，不构成新积分算子。
- 固定 $e,i$ 时，$T$ 精确等于对 $Q_{e,i}^b$ 做普通 path IG，因此属于普通元归因实例。
- 双线性反例给出 PEFA 与 Integrated Hessians 的不同权重和守恒终点，说明二者通常不数值相等；该区别不把二阶运算变成新算子。
- EAP-IG 的路径位于 clean–corrupted 内部激活空间并近似 patch metric；本合同路径位于输入根空间，二者通常不等价。
- $A/T/R$ 只对同一有向几何路径的保向重参数化不变；路径选择和参考点会改变坐标分配。
- 逐坐标仿射变换保留 $E/Q/A/T/R$ 的同索引数值，置换只重标；一般坐标混合不保留逐坐标归因。
- 同一 use 内、带精确逆 consumer 改写的可逆 $C^2$ 端口坐标变化保留根空间端口场及其派生量。
- 邻域内满足 $D_e=\sum_kD_{e_k}$ 的并行细化只保证细化项之和恢复；单项没有规范唯一性。
- 跨图重写只保留函数级总量，不保留端口级对象；新图必须重建端口 registry。

## 独立符号一致性

`run_step3_symbolic.py` 不通过步骤 2 的解析函数构造公式。它独立用 SymPy 重建六个冻结标量图和 use-port 场，直接验证四类符号恒等式，在 $t\in\{0,0.37,1\}$ 比较逐点导数，并以 60 位 mpmath 积分比较十类完整路径量。步骤 2 的 `analytic_fixtures.py` 仅在所有符号量建好后作为比较目标。

机器收据 `step3_symbolic_validation.json` 还验证七种表征情形：路径重参数化、路径选择、参考点、输入坐标、端口内可逆重参数化、并行细化和跨图重写。

## Prompt-to-artifact 审计

| 显式要求 | 具体证据 | 状态 |
|---|---|---|
| 执行 `执行规划.md` 步骤 3 | 本表逐项覆盖步骤 3 的工作、产物和门槛 | PASS |
| 正式证明逐端口闭合 | `theory_endpoint_pcia_pefa.md` 引理 2、定理 1，固定 $e,i$ 后沿路径使用微积分基本定理 | PASS |
| 正式证明逐 Endpoint 坐标闭合 | 理论文档引理 1、定理 1，合法切面和得到 $E_i^b$ | PASS |
| 正式证明 $T=\delta A+R$ | 理论文档引理 3、定理 2，逐 $e,i,j$ 乘积法则 | PASS |
| 正式证明 Endpoint–IG 差分恒等式 | 理论文档定理 3，逐坐标保留端口和来源索引 | PASS |
| 正式证明完整切面 $\Omega$ 零和 | 理论文档定理 4，邻域切面恒等式加混合偏导对称 | PASS |
| 明确光滑性条件 | 理论文档 1.1：路径邻域 $C^2$；非光滑反例和范围收缩另列 | PASS |
| 明确图冻结条件 | 理论文档 1.1：拓扑、use、slot、invocation、shape、目标固定 | PASS |
| 明确端口分离条件 | 理论文档 1.1/1.2：consumer use 独立余切，fan-out 不合并 | PASS |
| 明确路径条件 | 理论文档 1.1：像位于光滑邻域、分段 $C^1$、变化根完备 | PASS |
| 区分路径重参数化 | 理论文档 5.1 给出保向换元证明；符号收据含 nonlinear reparameterization 检查 | PASS |
| 区分路径选择 | 理论文档 5.2 给出双线性直线/折线路径反例；机器收据复核 | PASS |
| 区分参考点 | 理论文档 5.2 给出 $F(u)=u^2$ 反例；机器收据复核 | PASS |
| 区分输入坐标变换 | 理论文档 5.3 证明逐坐标仿射协变并给出一般混合反例；机器收据复核 | PASS |
| 区分端口内可逆重参数化 | 理论文档 5.4 给出链式法则抵消条件；机器收据用非对角可逆矩阵复核 | PASS |
| 区分并行细化 | 理论文档 5.5 给出邻域点态加法条件和五类求和关系；机器收据复核 | PASS |
| 区分跨图重写 | 理论文档 5.6 给出乘法单端口/双 slot 反例；机器收据复核 | PASS |
| 线性模型中 $R=0$ | 理论文档命题 5 对仿射图/常量端口场证明，并补充仅输出线性时端口项可相消的精确边界 | PASS |
| 单输入维度无反对称曲率 | 理论文档命题 6 | PASS |
| 完整模型梯度场总 $\Omega=0$ | 理论文档命题 7 | PASS |
| PEFA 退化为普通元归因的条件 | 理论文档 4.2：基础解释、根坐标、参考、路径和端口身份条件明确 | PASS |
| $R$ 的解释边界 | 理论文档 1.3/3：沿路径累计的 Endpoint 形成二阶项，不将二阶算子称新 | PASS |
| $\Omega$ 的解释边界 | 理论文档 1.3/3：点态端口场反对称诊断，不称 Riemann 曲率 | PASS |
| $R/\Omega$ 不是天然因果效应 | 理论文档反例 3 用零曲率但非零 intervention effect 区分 | PASS |
| 定义—引理—定理—反例—适用边界完整链 | `theory_endpoint_pcia_pefa.md` 第 1–7 节依次包含全部五类结构 | PASS |
| 数值观察与理论命题分栏 | 理论文档第 8 节是唯一数值观察区，并明确不参与证明 | PASS |
| 独立符号推导与步骤 2 fixture 一致 | `run_step3_symbolic.py`、`step3_symbolic_validation.json`：6/6 fixture，18 个点，60 次路径量比较 | PASS |
| 证明不完整内容降级 conjecture | 核心式均已证明；干预预测和增量效用列为 `UNTESTED_EMPIRICAL_CLAIM`，未当作理论结论 | PASS |
| 审计核心图限制是否可接受 | 固定图和光滑条件覆盖冻结 smooth fixtures 与固定 eval 图；`NO_GO_GENERALITY=false`，跨 kink/动态图显式排除 | PASS |
| 不越权进入步骤 4 | 产物只含 Markdown、SymPy/mpmath CPU 脚本和 JSON；无 JPEG、模型或 GPU 访问 | PASS |
| 独立构造与 CPU 边界 | `validate_step3.py` 解析 runner AST，禁止 Torch/timm/PIL/cv2，并确认 `build_fixture` 不引用步骤 2 解析实现 | PASS |
| 防御性写作审计 | 主文采用命题—证明—一次性边界；必要限制集中于 7.2，无 reviewer-facing prebuttal 或宣传性形容词 | PASS |
| 防御性写作审计产物 | `step3_defensive_writing_audit.md` 按 D1–D14 记录局部改写、保留边界和重复短语检查 | PASS |

## 复现与验证

在仓库根执行：

```bash
CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python vit_pcia_pefa/run_step3_symbolic.py
CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python vit_pcia_pefa/validate_step3.py
```

`validate_step3.py` 不把顶层 PASS 当作充分证据。它逐 fixture 检查四类符号恒等式、三个路径点、十类路径量和七类表征检查，重算理论、决策、前置收据及源码哈希，并检查本表对步骤 3 要求的覆盖。

## 下一门禁

步骤 4 必须由用户显式授权。当前决定不授权 ViT 适配、模型/JPEG 读取、GPU 使用或成本测量。
