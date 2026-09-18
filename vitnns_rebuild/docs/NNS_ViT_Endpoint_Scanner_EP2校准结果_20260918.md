# NNS-ViT Endpoint Scanner：EP2 校准结果

日期：2026-09-18。正式执行根为
[`reports/nns_vit_endpoint_ep2_r1_20260918`](../reports/nns_vit_endpoint_ep2_r1_20260918)。
Producer commit `e0b1c8b05dda0f3f5d5d2911bdc78dd6bd10eb94` 已在执行前推送。

```text
EP2_NUMERICAL_CALIBRATION_FAILED
SCIENTIFIC_RESULTS_NOT_RUN_CALIBRATION_ONLY
EVALUATION_ACCESS_FALSE
EP3_NOT_STARTED
NEXT_AUTHORIZATION_NONE_EP2_RECOVERY_REQUIRED
```

## 执行完整性

- 从 SHA-256 `c7e06a6c...0456ae0` 的官方 ImageNet validation tar 精确读取
  64 个冻结 calibration 成员；64/64 成员存在且逐文件绑定 SHA-256。两个模型
  各解码一次这些缓存字节，共 128 次 model-specific decode。
- Discovery member reads = 0；evaluation member reads = 0；replacement = 0。
- 两模型共 128 个 task，128/128 完成，0 遗漏、0 重复、0 非有限结果。
- 4 张空闲 H20 动态队列执行；GPU-hours 上界 `0.2857103`，峰值 reserved
  GPU memory `4,362,076,160` bytes，均低于 EP2 hard cap。
- 每个任务计算 ground-truth logit 与 frozen clean top-1 logit 的
  M=32/64/128/256/512 及对应 2M 检查图；不保留大 attribution maps。

## 预注册判定

每个模型、目标、候选 M 要求 64/64 finite，且至少 61/64 图同时满足
map relative L2 ≤ 0.02 和 completeness residual 门。

| 模型 | M | ground-truth pass | clean top-1 pass | 判定 |
|---|---:|---:|---:|---|
| DeiT-S/16 | 32 | 0/64 | 0/64 | FAIL |
| DeiT-S/16 | 64 | 6/64 | 7/64 | FAIL |
| DeiT-S/16 | 128 | 14/64 | 20/64 | FAIL |
| DeiT-S/16 | 256 | 45/64 | 47/64 | FAIL |
| DeiT-S/16 | 512 | **61/64** | **62/64** | PASS，选择 M=512 |
| ViT-B/16 | 32 | 0/64 | 0/64 | FAIL |
| ViT-B/16 | 64 | 0/64 | 0/64 | FAIL |
| ViT-B/16 | 128 | 1/64 | 1/64 | FAIL |
| ViT-B/16 | 256 | 9/64 | 7/64 | FAIL |
| ViT-B/16 | 512 | **41/64** | **40/64** | FAIL |

DeiT-S 选得 `M=512`；ViT-B 在最大冻结候选仍未达到 61/64，因此全阶段按
协议为 `EP2_NUMERICAL_CALIBRATION_FAILED`。这不是任务失败或 nonfinite：主要是
ViT-B 的 path quadrature 收敛/完整性没有在冻结候选范围内覆盖足够多图像。

EP0 禁止在看到本结果后追加 M、放宽 0.02/0.01 阈值、减少 cohort、按图选择
步数或只保留 DeiT-S。故不签发 EP3 权限。若继续研究，必须先制定新的前瞻性
EP2 recovery amendment 和新 root；原 R1 终态保持不可变。

## 独立验证

CPU-only validator 从 128 个 immutable task result 重算每个 M 的 64 图覆盖与
两模型决定，并从官方 tar 重新流式读取且哈希这 64 个成员，不依赖临时 cache；
同时检查 materialization、queue、producer source blob、文件哈希、资源与权限。
首次验证器把外层 task 生成顺序误用于队列 digest，产生 `TASK_DIGEST` false
rejection；[验证器修订](NNS_ViT_Endpoint_Scanner_EP2验证器修订_20260918.md)
只复现 producer 已冻结的 task-ID 排序，不改任何数值或判定。修订后独立验证正确
复现 `EP2_NUMERICAL_CALIBRATION_FAILED` 与 `next_authorization=NONE`。
