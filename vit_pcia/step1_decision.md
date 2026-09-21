# Endpoint–PCIA–PEFA 步骤 1 门禁决定

- 状态：`PASS_CONTRACT_FROZEN`
- 日期：2026-09-21（PRC）

## 决定

```text
STEP1_STATUS=PASS_CONTRACT_FROZEN
NOVELTY_POSITION=COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE
PCIA_FOUNDATIONAL_NOVELTY=REJECTED_CONDUCTANCE_EQUIVALENT_SPECIALIZATION
T_FOUNDATIONAL_NOVELTY=REJECTED_PATH_IG_OF_ENDPOINT_COMPONENT
R_OPERATOR_NOVELTY=REJECTED_WEIGHTED_HESSIAN_FAMILY
R_PORT_RESOLVED_IDENTITY_COVERAGE=NOT_FOUND_IN_FROZEN_SEARCH
OMEGA_OPERATOR_NOVELTY=REJECTED_STANDARD_ANTISYMMETRIC_DERIVATIVE
OMEGA_PORT_CUT_ATTRIBUTION_CONTRACT_COVERAGE=NOT_FOUND_IN_FROZEN_SEARCH
NO_GO_FOUNDATIONAL_NOVELTY=false
STEP2_ELIGIBLE=true
NEXT_AUTHORIZATION=STEP2_REQUIRES_EXPLICIT_USER_REQUEST
STEP2_STARTED=false
```

最低门满足的依据不是 $T$ 或 PCIA，而是冻结查新范围内尚未发现完整覆盖以下联合合同的工作：

1. $R$ 将 Endpoint–IG 的逐坐标差精确分配到“真实消费端口 × 形成来源坐标”；
2. $\Omega$ 在单端口可非零、而在原模型合法完整切面上严格零和；
3. 上述对象与 residual、Q/K/V、head、MLP 的 use-port 身份和负控制共同绑定。

这只开放 CPU 步骤 2。它不证明理论正确、实现闭合、ViT 端口忠实、成本可承受或实证有效，也不授权 GPU、JPEG、模型加载或步骤 3–6。

## Prompt-to-artifact 审计

| 显式要求 | 证据 | 状态 |
|---|---|---|
| 执行 `执行规划.md` 的步骤 1 | 本表逐项映射步骤 1 的工作、产物与门槛 | PASS |
| 核查 IG | `literature_matrix.md` 的 IG 行与 [L1] | PASS |
| 核查 conductance/neuron IG | 矩阵行、最近邻公式 4.1、[L2,L3] | PASS |
| 核查 Internal Influence | 矩阵行与 [L4] | PASS |
| 核查 Integrated Hessians | 矩阵行、最近邻公式 4.2、[L5] | PASS |
| 核查高阶/元归因 | Higher-Order 与 Metagame 两行、[L6,L7] | PASS |
| 核查 DeepLIFT | 矩阵行与 [L8] | PASS |
| 核查 LRP | LRP/AttnLRP 行与 [L9,L10] | PASS |
| 核查 Shapley Flow | 矩阵行与 [L11] | PASS |
| 核查 path-specific attribution | path-wise/path-specific 行与 [L12-L14] | PASS |
| 核查 ACDC | 矩阵行与 [L15] | PASS |
| 核查 EAP/EAP-IG | 两行与 [L16,L17] | PASS |
| 核查 DPA | Dual Path Attribution 行与 [L24] | PASS |
| 核查最新 transformer flow/circuit attribution | AtP*、二阶 patch 修正、Circuit Tracing、DePass、LIG/CA-LIG、IG-Lens、DPA/DAP、Unpack、GAF、DAG-SHAP、DAVE、HETA、Sparse Inter-Layer Dependencies、PA-LRP、GradSkip、HiLRP、ResLRP、ICAT、Contribution Weights、CAAP、LibraGrad、LeGrad、GMAR；[L42] 作跨架构覆盖复核，EP0 registry 作项目内差集复核 | PASS |
| 逐字段比较被解释对象等 10 项 | `literature_matrix.md` 第 2–3 节统一 10 字段矩阵 | PASS |
| 尝试将 $T$ 化约为 conductance/IH/普通 IG/EAP-IG/meta-attribution | `literature_matrix.md` 5.1；得到普通 path IG 完全化约、IH/EAP-IG 非数值等价、meta 类别覆盖 | PASS |
| 尝试将 $R$ 化约 | `literature_matrix.md` 5.2；判为 weighted-Hessian 家族，联合端口恒等式未检出 | PASS |
| 尝试将 $\Omega$ 化约 | `literature_matrix.md` 5.3；数学算子普通，端口合法切面零和合同未检出 | PASS |
| 冻结 $F,b,\gamma$ | `method_contract.md` 4.3–4.4 | PASS |
| 冻结变化根变量与输入坐标 | `method_contract.md` 4.1–4.2 | PASS |
| 冻结端口 ID 与合法切面 | `method_contract.md` 第 5 节 | PASS |
| 冻结积分规则、精度、闭合残差 | `method_contract.md` 第 6 节 | PASS |
| 冻结模型、数据、随机性、资源、失败类型 | `method_contract.md` 第 7–10 节 | PASS |
| 冻结 CPU fixture 输入/参数 | `method_contract.md` 6.2 | PASS |
| 核验 Endpoint 代码 provenance | `asset_inventory.md` 第 2–3、6 节；当前源码匹配 EP1 权威哈希 | PASS |
| 核验模型 provenance | `asset_inventory.md` 第 4 节；2 个 checkpoint 重新流式 SHA-256 | PASS |
| 核验数据 provenance | `asset_inventory.md` 第 4 节；tar、64 calibration、400 evaluation 均重新哈希 | PASS |
| 核验报告 provenance | `asset_inventory.md` 第 5 节；EP1/E3/E6 关键文件重新哈希并限定可支持主张 | PASS |
| 旧 source binding 漂移不得静默继承 | `asset_inventory.md` 第 2 节显式记录 EP0→EP1 授权演进 | PASS |
| 更新 `literature_matrix.md` | 已创建并冻结 | PASS |
| 更新 `method_contract.md` | 已创建并冻结 | PASS |
| 更新 `asset_inventory.md` | 已创建并冻结 | PASS |
| conductance 最近邻公式单列 | `literature_matrix.md` 4.1 | PASS |
| Integrated Hessians 最近邻公式单列 | `literature_matrix.md` 4.2 | PASS |
| 至少一个对象未被等价覆盖且可证伪主张写清 | $R/\Omega$ 联合合同未检出；`literature_matrix.md` 第 7 节列 C1–C5 | PASS_WITH_SCOPE_LIMIT |
| 若 $T$ 是 Endpoint 的普通 IG 则降级 | 已判定并降级为组合/系统化候选 | PASS |
| 若 $R/\Omega$ 也完整覆盖则 NO-GO | 未触发；发现新先例时强制触发 | PASS |
| 不启动步骤 2 | `method_contract.md` 第 11 节；本轮无新实现、模型加载或计算 | PASS |

## 验证命令

在仓库根执行：

```bash
python3 vit_pcia/validate_step1.py
```

验证器检查三份主文档、指定方法族、统一字段、$T/R/\Omega$ 审计、冻结合同字段、资产路径与 SHA-256、Endpoint provenance 漂移解释、门禁状态以及步骤 2 未启动。它不把文献覆盖检查自动化为“全球不存在”的证明；该边界本身也是必检文本。
