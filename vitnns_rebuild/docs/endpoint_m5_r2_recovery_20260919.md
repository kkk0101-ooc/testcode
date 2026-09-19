# Endpoint M5 R2 实现恢复说明

M5-R1 在完成只读前置门并写出 `predecessor_validation.json` 与 `freeze.json` 后，于第一次逐图汇总时因系统 Python 3.7 缺少 `statistics.fmean` 终止。R1 未写出 reference 结果、M2 复用结果、文稿或 terminal receipt；没有 JPEG、模型、checkpoint、Torch、timm 或 GPU 访问。

R1 根保持原样，绑定文件 SHA-256 为：

- `predecessor_validation.json`: `1da1c97d9184e823919d25746a72c44d3da2f2bed48f5f28346f99e5af29d784`；
- `freeze.json`: `0be599c0e430edd39bb9cec65954b54c982e72f2727c72de2b1e7d9997513b07`。

R2 仅把 `statistics.fmean(values)` 替换为 Python 3.7 可用且数值稳定的 `math.fsum(values) / len(values)`，写入新根 `reports/nns_vit_endpoint_m5_r2_20260919`。M5 的输入、指标、reference、聚合、bootstrap、方法矩阵、成本口径、Path IG 边界和停止规则均不改变。
