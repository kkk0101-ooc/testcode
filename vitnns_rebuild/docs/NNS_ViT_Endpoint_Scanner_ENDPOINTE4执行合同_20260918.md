# NNS-ViT Endpoint Scanner：ENDPOINT-E4 执行合同

日期：2026-09-18。用户已明确要求执行统一路线中的 `ENDPOINT-E4`。本合同在读取
新的正式干预 outcome 前冻结，权威机器可读配置为
`config/nns_vit_endpoint/endpoint_e4_intervention_run_spec.json`。

## 1. 前置与边界

- 权威前置是 ENDPOINT-E3 R2 handoff 的
  `ENDPOINT_E3_PASS_CORRECTNESS_ONLY_NEXT_ENDPOINT_E4_REQUIRES_EXPLICIT_REQUEST`；
- 沿用 EP0 的 400 张 evaluation 图像、两个冻结模型、ground-truth raw logit、
  unfused attention、float32 和两种同图 reference；
- 允许局部方向合同、整成分 activation replacement 和 C4a 每图统计；
- 禁止 Path IG、输入 fidelity/stability、E5 空间区域干预、全局 bootstrap/Holm 和
  最终 C4a/C4b claim decision。后两项留给 ENDPOINT-E6；
- 现有未跟踪 `nns_vit_cut_intervention/p0_c` 文件不是本研究权威输入，不读、不改。

## 2. 局部 correctness

对全部 12 blocks 的主成分及 Q/K/joint routing-value 次级成分，在前 16 个冻结
evaluation order 上计算 $v_k=J_{z_k,x}x$，并以

$$
rac{F(x)-F(z_k-\eta v_k)}{\eta}
$$

比较 $a_k=\sum_i E_{k,i}$。固定 $\eta=0.01,0.003,0.001,0.0003$；完整曲线有限，
且最小绝对误差不超过 $10^{-4}+0.01|a_k|$。Clean-to-clean replacement 必须在
float32 前向容差内 no-op。任何一项失败均保留根并 fail closed；通过只支持实现
correctness，不计为 utility。

## 3. 有限 activation replacement

reference 固定为：

1. normalized-zero：模型实际 normalized 输入张量全零；
2. blur RGB：clean 224 RGB float32 crop 上一次 31×31、sigma 10、reflect Gaussian
   blur，再按各模型 mean/std 归一化。

对每张图、每模型、每 reference，主成分为所有 block 的全部 heads、routing、Value、
attention branch 和 MLP branch。每次只替换一个注册成分，其他同切面边保持 clean，
并重算原生下游：

$$
e_k=F(x)-F(x;z_k\leftarrow z_k^{ref}).
$$

每成分保存 $a_k,m_k,e_k,p_k^{ref}$、output norm、gradient norm 和
gradient×activation。每图分别计算主 $\operatorname{Spearman}(m_k,|e_k|)$、signed
次级关系和全部 mandatory descriptive comparators。前 64 个 evaluation order 追加
Q/K、joint routing-value 和已冻结的 head/attention/MLP zeroing；这些不进入 C4a 主族。
随机对照由 task seed、reference、family 和 repeat 确定，共 20 次。

## 4. 执行与出口

总任务固定为 400 图 × 2 模型 = 800 个唯一 image-model task，采用 SQLite WAL
动态全局队列、lease/heartbeat/fencing、最多两次 attempt。每张合格 H20 一个常驻
worker；启动前记录 GPU UUID、空闲显存、利用率和既有 PID，不终止他人进程。
Torch/CUDA import 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，关闭 TF32/autocast。

由于统一路线把原 EP5-A 与 EP5-B 合并为 E4，资源上限取两者冻结上限之和：136
GPU-hours、132 wall-hours、56 GiB 增量 retained；每进程 GPU 24 GiB、CPU RSS 32
GiB、单任务 1800 秒，磁盘至少保留 50 GiB。

有效完成出口是：

```text
ENDPOINT_E4_PASS_EXECUTION_LOCAL_CONTRACT_AND_FINITE_REPLACEMENT
ENDPOINT_E5_REQUIRES_EXPLICIT_USER_REQUEST
```

该 PASS 表示 E4 执行、覆盖、数值和局部合同有效，不预先等同于 C4a `SUPPORTED`。
E4 结果文档可以报告冻结每图统计的描述性汇总；正式 bootstrap、全 28 项 Holm 和
claim-level `SUPPORTED/NOT_SUPPORTED/INCONCLUSIVE/INVALID` 由 E6 执行。
