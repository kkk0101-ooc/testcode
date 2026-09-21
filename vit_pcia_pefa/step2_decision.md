# Endpoint–PCIA–PEFA 步骤 2 门禁决定

- 状态：`PASS_CPU_FLOAT64_REFERENCE`
- 日期：2026-09-21（PRC）
- 范围：CPU-only 数学与实现验证；未读取 JPEG、未加载 ViT、未使用 GPU。

## 决定

```text
STEP2_STATUS=PASS_CPU_FLOAT64_REFERENCE
MAIN_FIXTURES_PASS=6/6
NEGATIVE_CONTROLS_DETECTED=9/9
PRIMARY_QUADRATURE=GAUSS_LEGENDRE_64_FIXED
CONVERGENCE_DIAGNOSTIC=GAUSS_LEGENDRE_128_FIXED
CPU_DTYPE=FLOAT64
STEP3_REQUIRES_EXPLICIT_USER_REQUEST
STEP3_TO_STEP6_NOT_STARTED=true
```

步骤 2 的冻结出口门已经满足：线性、双线性、平滑饱和、fan-out、residual 和多输入微型 DAG 均在原始输入坐标和未聚合 use-port 上通过；解析闭式、Torch 自动微分和中心有限差分相互一致；64 点主结果通过五类恒等式并与 128 点预声明诊断一致；九个负控制均触发冻结失败类型。

该决定只确认步骤 2 的定义实现和数值闭合。它不证明基础新颖性、ViT 端口忠实性、成本或实证效用，也不自动授权步骤 3–6。项目定位仍是 `COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE`。

## 数值摘要

- 主 fixture：`6/6` PASS。每个 fixture 均检查 64 与 128 点结果。
- 三方比较：在每个 fixture 的 `t in {0, 0.37, 1}` 检查解析解、自动微分和有限差分。有限差分的最坏绝对误差为 `3.293050276909071e-10`，低于 CPU `atol=1e-9`，对应最坏相对 L2 为 `2.16828047537011e-10`，低于 `rtol=1e-7`。
- 解析闭式与 64 点主积分的误差量级不超过 `1e-15`；64 与 128 点固定求积均在冻结容差内。
- `BILINEAR2` 的 Endpoint–IG 逐坐标差复现为 `[-3,-3]`，非零形成项位于 off-diagonal；`LINEAR3` 的 `R` 与 `Omega` 为精确零。
- 所有端口先经显式路径枚举证明每条根—目标路径恰好相交一次，再运行数值恒等式；没有以数值闭合反推切面合法。

## 五类恒等式覆盖

每个主 fixture、每个求积计划均逐原始坐标验证：

1. 端口 Endpoint 闭合：`sum_e D_e = grad F`，`sum_e Q_e = Endpoint`；
2. PCIA 到 IG：`sum_e A_e = IG`，并检查 `sum_i,e A_ei = F(x)-F(b)`；
3. PEFA 到 Endpoint：逐端口 `sum_j T_eij = Q_ei`，完整切面 `sum_e,j T_eij = Endpoint_i`；
4. 分解恒等式：`T = delta*A + R`，另检查 `sum_e,j R_eij = Endpoint_i-IG_i`；
5. 完整切面曲率零和：`sum_e Omega_eij = 0`。

`mvp_validation.json` 保存每个 fixture、节点计划、恒等式和坐标数量的独立残差，不以顶层 PASS 或聚合后的相似度替代逐坐标检查。

## 负控制与非光滑边界

| 控制 | 预期且实际触发 | 判定依据 |
|---|---|---|
| 遗漏 residual identity | `FAIL_CUT_NOT_LEGAL` | identity 路径交点计数为 0 |
| 重复 residual identity | `FAIL_CUT_NOT_LEGAL` | identity 路径交点计数为 2 |
| 合并 fan-out consumer cotangent | `FAIL_PORT_ID_AMBIGUOUS` | 数值路径覆盖虽看似完整，但 consumer use 身份不再匹配冻结的两个端口 |
| 交换 multi-input slot | `FAIL_PORT_ID_AMBIGUOUS` | `mul.slot0/slot1` 的冻结端口身份被改变 |
| 重复串联切面 | `FAIL_CUT_NOT_LEGAL` | 每条路径被两个串联切面重复命中 |
| detach 二阶依赖 | `FAIL_SECOND_ORDER_GRAPH_TRUNCATED` | `D.requires_grad=false` 且无 `grad_fn`，在数值闭合前失败 |
| 只取 Hessian 对角 | `FAIL_QUADRATURE_CLOSURE` | `BILINEAR2` 的 off-diagonal `R` 被丢弃，`T=delta*A+R` 失败 |
| `RELU_SAFE` 跨 kink、节点不命中 | `FAIL_QUADRATURE_CLOSURE` | 几乎处处 Hessian 漏掉梯度跳跃的分布项；未用节点噪声伪造闭合 |
| `RELU_HIT` 命中冻结节点 | `FAIL_NONSMOOTH_NODE_HIT` | 在求导前用同一 float64 节点表达式检测精确命中 |

`RELU_SAFE` 与 `RELU_HIT` 均只作为负控制。前者说明“求积节点没有碰到 kink”并不足以保证 PEFA 恒等式；后者冻结节点命中即 fail-closed。主 PASS 只来自二阶光滑的六个 fixture。

## 产物

| 产物 | 作用 |
|---|---|
| `fixture_registry.json` | outcome 前冻结 6 个主 fixture、端口身份、路径、有限差分和容差 |
| `quadrature_registry.json` | 保存 64/128 点 float64 节点、权重、顺序及原始 bytes 哈希 |
| `src/analytic_fixtures.py` | 不导入 Torch 的解析导数与闭式路径解 |
| `src/reference.py` | CPU float64 Endpoint、端口 D/Q、PCIA A、PEFA T/R、Omega 实现 |
| `tests/test_step2_reference.py` | 三方比较、恒等式、收敛与负控制单元测试 |
| `negative_control_receipts/*.json` | 九项逐负控制收据 |
| `mvp_validation.json` | 逐 fixture、逐节点计划、逐恒等式的机器可读结果与源码/收据绑定 |
| `validate_step2.py` | 不信任顶层 PASS、逐字段复核覆盖和哈希的 fail-closed 验证器 |

## Prompt-to-artifact 审计

| 显式要求 | 具体证据 | 状态 |
|---|---|---|
| 执行 `执行规划.md` 步骤 2 | 本表逐项映射步骤 2 的工作、产物和门槛 | PASS |
| CPU float64 | `mvp_validation.json.environment` 记录 CPU、`torch.float64`、CUDA 不可见 | PASS |
| 线性 fixture | `LINEAR3` registry、测试和 validation 行 | PASS |
| 双线性 fixture | `BILINEAR2` registry、off-diagonal 与 `[-3,-3]` 解析断言 | PASS |
| 饱和非线性 fixture | `SATURATING3` 的 tanh 解析导数与闭式积分 | PASS |
| fan-out fixture | `FANOUT2` 两个不同 consumer use-port 与完整路径枚举 | PASS |
| residual fixture | `RESIDUAL2` identity/branch 两端口完整切面 | PASS |
| 多输入算子 fixture | `MULTIINPUT3` 的 `mul.slot0/slot1` 与两条线性 use | PASS |
| 独立 Endpoint 实现 | `reference.py` 的 `endpoint=delta*gradient`，与解析闭式比较 | PASS |
| 独立端口 D/Q 实现 | use-specific VJP 与 Q Jacobian；解析模块独立给出 D/Jacobian | PASS |
| 独立 PCIA A 实现 | 固定求积直接积 `dot_gamma_i*D_ei` | PASS |
| 独立 PEFA T/R 实现 | T 直接积 `Jac(Q)`，R 独立积 `Jac(D)`，未以分解式定义 T | PASS |
| 独立 Omega 实现 | 从未聚合逐端口 `Jac(D)` 的反对称部分计算 | PASS |
| 对比解析解 | 6 个 fixture 的点导数和不依赖求积的闭式路径量 | PASS |
| 对比自动微分 | Torch `autograd.grad(create_graph=True)`，保留 cotangent 全依赖 | PASS |
| 对比有限差分 | 6 个 fixture × 3 点；同时检查梯度、端口 D 的 Jacobian 与端口 Q 的 Jacobian | PASS |
| 冻结 64 点 Gauss–Legendre 主计划 | `quadrature_registry.json` 的 64 点 float64 bytes 哈希 | PASS |
| 预声明 128 点诊断 | 同 registry；每个 fixture 均与 64 点逐量比较 | PASS |
| 禁止按样本自适应 | 实现只接受 registry 的固定节点，无自适应分支 | PASS |
| 端口 Endpoint 闭合 | 6 fixture × 2 求积计划逐坐标验证 D 与 Q 闭合 | PASS |
| PCIA 到 IG | 逐坐标 A→IG 及标量 output delta 验证 | PASS |
| PEFA 到 Endpoint | 逐端口 T→Q 和完整切面 T→Endpoint 验证 | PASS |
| `T=delta*A+R` | 全张量逐元素验证 | PASS |
| 完整切面 Omega 零和 | 未聚合端口、全 `i,j` 验证 | PASS |
| 遗漏 residual 负控制 | 独立 receipt | PASS |
| 重复 residual 负控制 | 独立 receipt | PASS |
| 合并 fan-out 余切负控制 | 独立 receipt | PASS |
| 交换 input slot 负控制 | 独立 receipt | PASS |
| 重复串联切面负控制 | 独立 receipt | PASS |
| 错误 detach 二阶项负控制 | 独立 receipt | PASS |
| 只取 Hessian 对角负控制 | 独立 receipt | PASS |
| 分段光滑/几乎处处口径 | `RELU_SAFE` 明确暴露跨 kink 的 closure failure | PASS |
| 节点命中规则 | `RELU_HIT` 在求导前触发 `FAIL_NONSMOOTH_NODE_HIT` | PASS |
| 先原始坐标、后聚合 | 验证对象保留 `[port,i,j]`；没有展示聚合路径 | PASS |
| 参考实现 | `src/reference.py` | PASS |
| 解析 fixture | registry 加 `src/analytic_fixtures.py` | PASS |
| 单元测试 | `46 passed`，见下方复现命令 | PASS |
| `mvp_validation.json` | 存在且由 runner 生成 | PASS |
| 逐负控制收据 | `negative_control_receipts/` 下 9 份闭合 schema JSON | PASS |
| 正确实现全部恒等式在容差内 | 6/6 fixture；验证器逐 metric 检查 | PASS |
| 每个负控制触发预期失败 | 9/9 expected 与 observed failure code 相同 | PASS |
| 不进入后续步骤 | 未读取图像、模型或 GPU；`STEP3_REQUIRES_EXPLICIT_USER_REQUEST` | PASS |

## 复现与验证

在仓库根执行：

```bash
CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python vit_pcia_pefa/freeze_step2.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .conda/envs/vitnns/bin/python -m pytest -q vit_pcia_pefa/tests/test_step2_reference.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .conda/envs/vitnns/bin/python vit_pcia_pefa/run_step2.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .conda/envs/vitnns/bin/python vit_pcia_pefa/validate_step2.py
```

`validate_step2.py` 不以测试绿灯或 `mvp_validation.json.status` 单独作为完成证据。它重建求积节点 bytes，逐 fixture 核查三方比较、五类恒等式、64/128 结果、路径交点计数、九份收据及预期/实际失败码，并重算 registry、合同、源码和收据 SHA-256。
