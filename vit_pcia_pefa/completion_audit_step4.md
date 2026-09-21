# 步骤 4 完成审计

## 目标与成功标准

目标是执行 `执行规划.md` 步骤 4，在不改写 Endpoint 历史证据的前提下完成真实 ViT 端口适配，生成端口/切面/验证/成本产物，使 Q/K/V、head、residual、projection、MLP 的代表切面通过五类恒等式与适用负控制，并在冻结资源预算内给出可复核结论。步骤 5 不在本次授权范围。

## Prompt-to-artifact checklist

| 显式要求 | 实证与覆盖 | 结论 |
|---|---|---|
| 执行 `vit_pcia/执行规划.md` 步骤 4 | 原目录迁移为 `vit_pcia_pefa/`；两处规划内容一致；本审计使用新 root，未回填旧证据 | PASS |
| 步骤 1–3 先通过 | `validate_step2.py` 与 `validate_step3.py` 复跑 PASS；步骤 1 决策与合同已冻结，旧 validator 因后续产物存在按时间门拒绝 | PASS |
| 不改写历史 Endpoint 证据 | 复用文件 `endpoint_vit.py`、`test_endpoint_vit.py` 相对 HEAD 无差异；22 项旧回归通过 | PASS |
| 新 PEFA source binding | `step4_attempt2_source_binding.json` 自身 canonical digest 复算一致；16 个源/合同文件和两个 checkpoint 无漂移 | PASS |
| 目标/模型/数据真实绑定 | 两个冻结 checkpoint 严格加载；共同使用 discovery order 0，JPEG SHA-256 `6a9e...daf`，GT class 721 | PASS |
| ViT 端口注册表 | `vit_port_registry.json` 有 216 个 model×block×cut 条目，8 个必需 port-ID 字段齐全 | PASS |
| residual identity/branch | `RESIDUAL1_FULL` 与 `RESIDUAL2_FULL` | PASS |
| attention heads | `HEADS_WITH_BYPASS_FULL`，所有 head slices 互斥、完备 | PASS |
| 单 head Q/K/V | `QKV_WITH_BYPASS_FULL`，Q/K consumer slot 0/1 分离，V 为 AV slot 1 | PASS |
| projection | `PROJECTION_WITH_BYPASS_FULL` | PASS |
| 完整 MLP 合法切面 | fc1、GELU、fc2 三个替代切面各自包含 carry；未将串联 cut 相加 | PASS |
| 先单 block | 两模型 block 0 共 18/18 cut 通过 | PASS |
| 再完整 ViT | 两模型各 12/12 block 的全部 9 类一阶切面通过 | PASS |
| HVP/JVP/VJP，禁显式 Hessian | 实现使用 JVP/VJP；AST 审计无 `hessian`/`jacobian` 构造；结果声明未物化大张量 | PASS |
| patch-group 流式实现 | attempt0 实现并实测四个精确 patch groups；因成本/求积失败封存 | PASS（实现要求）；该分辨率 NO_GO |
| 稀疏化/聚合证明闭合 | attempt2 改为对全部来源 $j$ 精确收缩，使用端点式 $\bar T_e=Q_e(x)$；不是稀疏或近似；验证仍在逐 RGB 坐标 | PASS |
| Endpoint 端口闭合 | 两模型 9/9 主 cut；全模型 12/12 block 一阶验证 | PASS |
| PCIA→IG | 同次 512 节点执行的端口和恢复 `__full__` path-IG，18/18 cut | PASS |
| PEFA→Endpoint | source-summed PEFA 逐 RGB 恢复 Endpoint，18/18 cut | PASS |
| $T=\delta A+R$ | 收缩后 $\bar T=A+\bar R$，18/18 cut | PASS |
| 完整切面曲率零和 | 每 cut 两个冻结方向的 JVP/VJP 对称检查，36/36 probes | PASS |
| 全部适用负控制 | 18/18：residual 漏/重、head/QKV 漏项、slot 交换、串联 MLP 重复、cotangent 合并、二阶 detach | PASS |
| 实现测试 | Step 4 6/6；步骤 2 46/46；旧 Endpoint 22/22 | PASS |
| `vit_validation.json` | canonical digest 复算一致；独立 validator 核查 2 模型、18 cuts、24 blocks、18 controls | PASS |
| 成本计数 | `cost_report.json` 分 Endpoint/IG/PCIA/PEFA 记录 logical forward/reverse 与 512 节点 | PASS |
| 峰值内存与存储 | 最大 GPU reserved 3.31 GB，最大 RSS 1.75 GB，当前保留约 2.14 MB，均低于冻结 cap | PASS |
| 相对成本 | ViT-B PEFA 约 169× Endpoint、19× IG、1.002× PCIA；DeiT-S 约 97×、14.5×、1.002× | PASS |
| 生产成本门 | 64 图×2 模型×12 blocks 外推 13.586 H20-hours < 16；本步骤实际上界 0.616 H20-hours | PASS |
| 失败根保留 | attempt0 和 attempt1 receipt 及 runtime JSON 均保留，未改写成 PASS | PASS |
| 步骤 5 未启动 | `vit_validation.json` 为 `step5_started=false`；不存在 step5 runtime 产物 | PASS |

## 最终结论

所有步骤 4 明示交付与门槛均有直接证据覆盖。最终状态为 `PASS_VIT_PORT_FIDELITY_AND_COST_GATE`。该结论仅适用于 source-summed 精确收缩；四 patch-group 来源分辨率仍为封存的 `FAIL_QUADRATURE_CLOSURE + NO_GO_COST`，不能作为已通过能力对外陈述。
