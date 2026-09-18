# NNS-ViT Endpoint Scanner：EP1 实现与回归

日期：2026-09-18。EP0 权威基线为 commit
`6237ef87a1af0205973b76bee172defe33a0a6da`。用户本轮明确授权“执行 EP1”。
本阶段只实现与验证工程合同，使用冻结 checkpoint 和确定性合成输入；未打开
discovery、calibration 或 evaluation 图像，未产生科学 outcome，也未启动 EP2。

```text
PASS_EP1_IMPLEMENTATION_AND_REGRESSION
SCIENTIFIC_RESULTS_NOT_RUN
PRODUCTION_OUTCOME_ACCESS_FALSE
EP2_NOT_STARTED
```

正式根：[R11 terminal receipt](../reports/nns_vit_endpoint_ep1_r11_20260918/terminal_receipt.json) ·
[验证报告](../reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json) ·
[覆盖审计](../reports/nns_vit_endpoint_ep1_r11_20260918/coverage_audit.json) ·
[task manifest](../reports/nns_vit_endpoint_ep1_r11_20260918/task_manifest.json) ·
[artifact manifest](../reports/nns_vit_endpoint_ep1_r11_20260918/artifact_manifest.json)。

## 1. 实现结果

- Endpoint target 新增 `patch_channel_energy`，严格计算同一捕获张量上的
  $\frac12\sum_{t=1}^{196}h_{t,c}^2$，排除 CLS；另加入显式
  `ground_truth_logit` 与 `frozen_clean_top1_logit` 目标合同。
- canonical unfused timm Attention 显式暴露 Q/K/V projection、row softmax、
  $AV$、projection 前后及 residual 边；scoped forward 在成功和异常路径均恢复。
  Fused backend 只做原生 forward parity，不登记结构分解支持。
- 结构分解显式区分 `A_F` branch reconstruction 与 `L_F` total
  reconstruction。Head、Q/K/V、routing+Value 的 branch/total cut 分别校验，
  不再把 branch-only cut 错当完整类别图。
- CNN paper-formula fixture 覆盖 convolution bias、ReLU 开/关、唯一
  MaxPool winner 和 residual；tie 明确拒绝。
- 新增 SQLite WAL 全局动态队列：固定 manifest、事务 claim、lease/heartbeat、
  递增 attempt/fencing token、过期提交拒绝、不可变结果和 exact coverage audit。
  Seed 只由稳定 task ID 导出，与 worker/GPU/完成顺序无关。

## 2. 回归与负例

最终 CPU suite 为 **44 passed, 0 failed**，包含以下 EP1 项：

1. paper-formula CNN 与独立 G×I parity；关闭 ReLU 门、唯一 pool winner、
   residual 和 tie rejection；
2. energy 同张量绑定、all-block target kinds、ground-truth/top-1 目标约束；
3. residual 重复计数、非零 head 遗漏、非零 routing 遗漏三个反例均返回
   `CUT_CLOSURE_FAILED`；
4. Q/K/V、routing/value、head branch `A_F` 与带 bypass total `L_F` 分离；
5. no-hook、no-op hook、正式 unfused edge capture 原生 forward parity；
6. float64/float32 absolute/relative near-zero 分支，有限差分、JVP 和
   activation-edge 局部方向合同；
7. 单/多 worker 等价、lease recovery、过期 fencing token、最终 attempt
   fail-closed、结果不可变、缺失/篡改检测。

## 3. GPU 工程验证

启动快照选择物理 GPU 0–3，共 4 张空闲 NVIDIA H20；GPU 4–7 上的既有服务
只记录、不干扰。每个 worker 在 Torch/CUDA import 前继承
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，运行 matmul+backward startup probe，关闭
TF32/autocast，并记录 Python realpath、CUDA/包版本、UUID 和已有 PID。

固定 8-task manifest 在一张 H20 和同一组 4 张 H20 上分别完整运行：

| 项目 | 结果 |
|---|---:|
| 模型 | ViT-B/16、DeiT-S/16 |
| 单卡覆盖 | 8/8，遗漏 0，重复 0 |
| 四卡覆盖 | 8/8，遗漏 0，重复 0 |
| 单卡/四卡逐 task numeric digest | 8/8 相等 |
| block 覆盖 | 两模型均覆盖 0–11 |
| 每 task target 数 | 16 |
| 成本轴 | K=1/16，M=4/8，C=14/20 |
| fused/unfused logits 最大绝对差 | $8.106231689453125\times10^{-6}$ |
| hook forward 最大绝对差 | 0 |
| cut closure 最大绝对差 | $1.2665987014770508\times10^{-7}$ |
| cut closure 最大 relative L2 | $1.6204431858568569\times10^{-6}$ |
| 单 task scan latency P50 | 0.11804 s（单卡） |
| 多卡 task throughput | 0.85246 tasks/s（端到端，含各 worker 模型加载） |
| 最大 peak reserved GPU memory | 约 4.05 GiB（单卡记录） |
| GPU-hours 上界记录 | 0.0133761 |
| retained evidence | 约 127 KiB |

当前单任务小样本的 P95 采用观测最大值作保守代理，未把四卡总吞吐解释成算法
加速。K/M/C 是 EP1 工程成本轴，只用于证明入口可测；EP2 仍须在冻结的 64 张
calibration cohort 上按 EP0 算法选择正式 IG 节点和 batch/粒度。

## 4. 失败根与恢复

R1–R5 是启动门禁失败的空根：依次暴露 EP0 源码漂移核验位置、隔离 worktree
缺本地 checkpoint/文件名快照等问题。R6 在 GPU 回归中发现 CPU 回读 token tensor 与
GPU 原始 energy scalar 的 float32 归约顺序差异；最大量级约 $3.8\times10^{-6}$。
恢复按 EP0 的 float32 `atol=1e-5, rtol=1e-4` 合同校验，没有放宽阈值。R7–R9
先后被独立 validator 的旧 K 常量或实现 hash 漂移拒绝。R10 完成最终开发态双验收；
为让 `executed_head` 直接绑定已提交实现，正式证据在实现 commit 后重发为 R11。
所有早期根保持本地只读诊断；正式结论只绑定最终 R11。

## 5. 阶段边界

EP1 只支持已注册的 timm 0.9.16 standard Block/Attention、ViT-B/16 和
DeiT-S/16。无输入相关 attention bias；若未来模型增加该边，必须扩展 registry
和闭合切面后再声明支持。Fused kernel 没有结构 edge 支持。

本阶段没有读取正式图像、没有计算评价/干预结果、没有产生 C1–C4 科学判定。
唯一下一步为：

```text
EP2_REQUIRES_EXPLICIT_USER_REQUEST_AND_PUBLISHED_EP1_PASS
```

复核命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  ../.conda/envs/vitnns/bin/python scripts/validate_endpoint_ep1.py
```
