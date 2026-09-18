# NNS-ViT Endpoint Scanner：ENDPOINT-E2 结果

日期：2026-09-18。权威出口：

```text
ENDPOINT_E2_PASS_NEXT_ENDPOINT_E3_REQUIRES_EXPLICIT_REQUEST
```

## 完整性

- 400 个冻结 evaluation ID，两个模型共享全部 ID；逐 tar member 重算 SHA-256；
- 800/800 image-model tasks 唯一完成，0 missing、0 duplicate、0 invalid、0 formal retry；
- ViT-B/16 457 targets，DeiT-S/16 385 targets，共 336,800/336,800 finite；
- 800 份 ground-truth/clean-top1 raw signed RGB map；
- 前 16 evaluation ID × 两模型共 32 份 full raw atlas；
- near-zero (`max_abs <= 1e-12`) 为 0，failure index 为 0；
- 固定前 8 ID gallery 已通过人工布局检查。

## 资源

| 指标 | 结果 | 冻结 cap |
|---|---:|---:|
| formal wall time | 1,752.71 s | 72 h |
| 4-card upper GPU-hours | 1.9475 | 64 |
| peak reserved GPU memory | 2.284 GB | 24 GiB / process |
| peak CPU RSS | 2.018 GB | 32 GiB / process |
| retained bytes | 9.290 GB | 128 GiB incremental |
| exclusive scan P50/P95/max | 8.754 / 9.658 / 10.615 s | descriptive |
| formal targets/s | 192.16 | descriptive |
| formal images/s | 0.4564 | descriptive |

完整 atlas 的 median / P95 / max：DeiT-S 为 7.853 / 8.015 / 8.240 秒，
ViT-B 为 9.485 / 9.751 / 10.615 秒。

## 边界与审计

本阶段只作 `C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES`。没有运行 Path IG、输入
fidelity/stability、有限干预或 utility 统计。用户在 formal 800/800 完成后取消全量
单卡重放；已完成的 86 个 replay task 和中止 claim 保留但不进入 PASS 或科学证据。

R1–R3 的 handoff manifest/provenance 失败均 fail closed 保留，不是权威 terminal。
权威 R4 handoff 位于
`reports/nns_vit_endpoint_e2_atlas_r4_20260918/terminal_receipt.json`；独立验证器从
frozen tar、registry、800 个 queue results、raw NPY、Git blob 和资源记录全量重算通过。
下一阶段 ENDPOINT-E3 尚未授权。
