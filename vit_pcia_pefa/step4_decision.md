# Endpoint–PCIA–PEFA 步骤 4 门禁决定

- 状态：`PASS_VIT_PORT_FIDELITY_AND_COST_GATE`
- 日期：2026-09-21（PRC）
- 范围：真实 ViT 端口适配与成本门；未启动步骤 5 pilot。

## 决定

```text
STEP4_STATUS=PASS_VIT_PORT_FIDELITY_AND_COST_GATE
SOURCE_RESOLUTION=ONE_EXACT_ALL_COORDINATE_CONTRACTION
MODELS_PASS=2/2
PRIMARY_CUTS_PASS=18/18
ALL_BLOCK_FIRST_ORDER_PASS=24/24
NEGATIVE_CONTROLS_PASS=18/18
PROJECTED_PRODUCTION_H20_HOURS=13.585906151284775
FROZEN_H20_HOURS_CAP=16
STEP5_STARTED=false
NEXT_AUTHORIZATION=STEP5_REQUIRES_EXPLICIT_USER_REQUEST
```

步骤 4 在前瞻性 attempt2 合同下通过。适配对象是 timm 0.9.16 的两个注册 unfused `eval()` 模型。端口 ID 均包含 model、graph revision、block、producer tensor、consumer operator、input slot、invocation 和 slice。机器注册表覆盖 residual identity/branch、全部 attention head、逐 head Q/K/V、projection，以及三种彼此替代的完整 MLP 内部切面。

来源坐标采用规划允许的分辨率收缩：对全部 $j$ 精确求和，但保留端口 $e$ 和 Endpoint 坐标 $i$。$\bar T_e=Q_e(x)-Q_e(0)$，$\bar R_e=\bar T_e-A_e$。这不是聚合近似；五类关系仍在 signed normalized RGB `[1,3,224,224]` 坐标上逐元素验证。

## 结果摘要

| 模型 | block 0 合法切面 | 全 12 block 一阶切面 | 负控制 | source-summed PEFA 秒/图/block | 峰值保留显存 |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 | 9/9 | 12/12 | 9/9 | 40.398 | 3.08 GiB |
| DeiT-S/16 | 9/9 | 12/12 | 9/9 | 23.286 | 1.49 GiB |

按 64 张 discovery 图、两个模型、每模型 12 blocks、单 ground-truth target 线性外推为 13.586 H20-hours，低于冻结 16 H20-hours 门。含两个失败 attempt、独立二阶诊断和保守补记 600 GPU 秒后，本步骤实际使用上界为 0.616 H20-hours。保留文件约 2.1 MB，远低于 32 GiB；GPU 与 CPU 峰值也低于每 worker 上限。

## 保留的失败证据

`step4_attempt0_receipt.json` 保留四 patch-group 来源分辨率的 `FAIL_QUADRATURE_CLOSURE` 与 `NO_GO_COST`；该路线生产投影 268.38 H20-hours。`step4_attempt1_receipt.json` 保留 ViT-B 在相同 512 节点、不同 alpha batch 执行之间的 $2.6331\times10^{-5}$ 最大逐坐标差。attempt2 以同一次 PCIA 执行的 `__full__` 通道作为 path-IG 恒等式参考；独立 batch=8 IG 仍记录在结果中，其中 DeiT-S 通过、ViT-B 未通过该跨执行诊断。

## Prompt-to-artifact 审计

| 要求 | 证据 | 状态 |
|---|---|---|
| 复用 Endpoint 目标解析与 unfused 捕获但不改旧证据 | 新目录实现引用只读 `endpoint_vit.py`；`step4_attempt2_source_binding.json` 绑定字节 | PASS |
| 新 source binding | `step4_attempt2_source_binding.json`，验证时零漂移 | PASS |
| ViT 端口注册表 | `vit_port_registry.json`，216 个 model-block-cut 条目 | PASS |
| 机器可读合法切面 | 同上，封闭 8 字段 port ID，9 类切面 | PASS |
| residual identity/branch | `RESIDUAL1_FULL`、`RESIDUAL2_FULL` | PASS |
| attention heads | `HEADS_WITH_BYPASS_FULL`，互斥完整 head slices | PASS |
| 单 head Q/K/V | `QKV_WITH_BYPASS_FULL`，每个 head 和 consumer slot 独立 | PASS |
| projection | `PROJECTION_WITH_BYPASS_FULL` | PASS |
| 完整 MLP 合法切面 | fc1/GELU/fc2 三个替代切面，各含 carry | PASS |
| 先单 block | 两模型 block 0 共 18/18 切面通过五类关系 | PASS |
| 再完整 ViT | 两模型全部 12 blocks 的全部 9 类一阶切面通过 | PASS |
| 避免显式 Hessian/三阶张量 | JVP/VJP 方向探针；无 Hessian/Jacobian 构造；仅保存摘要 | PASS |
| 收缩须保留闭合 | 全来源坐标求和由端点式精确给出；逐 RGB 坐标验证，不以展示聚合闭合 | PASS |
| 五类恒等式 | Endpoint、PCIA→IG、PEFA→Endpoint、乘积法则、Omega 零和全部逐 cut 通过 | PASS |
| 适用负控制 | 两模型各 9 项，遗漏/重复 residual、head/QKV、slot、串联 MLP、detach、cotangent merge 均被检出 | PASS |
| 实现测试 | `tests/test_step4_vit_adapter.py`：6 passed | PASS |
| `vit_validation.json` | 状态 `PASS_VIT_PORT_FIDELITY_AND_IDENTITIES` | PASS |
| 测量 forward/reverse/节点/显存 | `cost_report.json` 分模型、分 Endpoint/IG/PCIA/PEFA 记录 | PASS |
| 测量临时/持久存储 | `cost_report.json` 记录 runtime 与全部保留 bytes | PASS |
| 相对 Endpoint/IG/PCIA 成本 | `cost_report.json` 的 `relative_wall_cost` | PASS |
| 成本门 | 生产投影 13.586 < 16 H20-hours；其他资源 cap 均通过 | PASS |
| 失败证据不可改写 | attempt0/attempt1 receipt 与原始 runtime JSON 保留 | PASS |
| 不进入 pilot | 所有 source/validation 均为 `step5_started=false` | PASS |

## 复现

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:vitnns_rebuild/src .conda/envs/vitnns/bin/python -m pytest -q vit_pcia_pefa/tests/test_step4_vit_adapter.py
.conda/envs/vitnns/bin/python vit_pcia_pefa/validate_step4.py
```

本 PASS 只确认收缩后的真实端口实现、数值恒等式与成本门，不证明实证增量或干预预测力。步骤 5 必须由用户再次显式授权。
