# NNS-ViT Endpoint Scanner：ENDPOINT-E5 执行合同

日期：2026-09-18。用户已明确要求执行统一路线中的 ENDPOINT-E5。本合同在读取新的
E5 正式 outcome 前冻结，机器可读合同为
config/nns_vit_endpoint/endpoint_e5_spatial_run_spec.json。

## 1. 前置、样本与边界

- 唯一有效前置为 E4-R3 的
  ENDPOINT_E4_R3_PASS_EXECUTION_CONDITION_SCALED_JVP_AND_FINITE_REPLACEMENT；
- 沿用 E2 物化且逐文件哈希的 400 张 evaluation 图、两个冻结模型和 ground-truth
  raw logit，不换图、不删错分图；
- 允许 C4b 空间干预、逐图主分数及配对差和描述性汇总；
- 禁止 Path IG、原 EP3、输入 fidelity/stability、bootstrap、Holm、最终 C4b claim
  decision 以及 E6。后四项只可在 ENDPOINT-E6 获得单独授权后运行；
- 现有未跟踪 nns_vit_cut_intervention/p0_c 文件不读、不改、不纳入证据。

## 2. 空间干预

固定 block 5；ViT-B heads 为 0/6/11，DeiT-S heads 为 0/3/5。224×224 crop 被分为
row-major 的 4×4 个不重叠 56×56 区域，预算为 0/1/2/4 个区域，对应实际面积
0、1/16、2/16、4/16。两种 reference 仍为 normalized-zero 和在 clean RGB crop 上
以 31×31、sigma 10、reflect 生成后再归一化的 blur。

对每个 head，主方法按原始 signed component map 在每个区域的 RGB 坐标 absolute mass
降序排序。四个 mandatory baselines 为 total Endpoint/G×I、raw input gradient、
regional activation-space Taylor 和 20 个冻结随机区域排列。确定性排序 ties 按区域 ID
升序；随机种子为 namespace、queue task seed、model、reference、head 和 repeat 的 SHA-256
前 8 bytes big-endian，经 NumPy 1.26.4 PCG64 生成。禁止查看有限效果后选区。

Regional Taylor 对 16 个单区域输入替换各做一次 forward，同一结果供三个 head 共用；
其分数为 clean head cotangent 与 clean-minus-perturbed head activation 位移内积的绝对值。
对排序冻结后的 1/2/4 区域 union 运行真实有限前向，并计算主 isolated effect：

$$
F(x)-F(z_h(x_R),z_{-h}(x),\mathrm{identity}(x)).
$$

同一输入替换的 full-forward effect 单独保留为描述性结果，不能冒充 head 独立效应。

## 3. 每图统计输入

每条曲线保留四个预算点、选择区域、isolated/full-forward raw logit 和 effect。每个
确定性方法先在每个 head 上算 0–25% 实际面积的 absolute isolated-effect 梯形 AUC 并
除以 0.25，再对三个 head 取均值。随机 baseline 必须先在同一 head/reference 内对 20 条
排列的 absolute effect 曲线逐预算求均值，再算 AUC，最后对三个 head 取均值。保存
Endpoint 减四个 baseline 的逐图配对差；随机方法严格按冻结表述先平均 signed effect
曲线，再取绝对值计算 AUC。不在 E5 汇总成全 cohort 推断。

每个 image-model task 固定 2 reference、144 条曲线（24 条确定性、120 条随机）、
32 次单区域 Taylor 输入前向、432 个有限 union full-forward 曲线点和 432 次 isolated-head
evaluation；相同 union 的 full-forward 只计算一次并记录实际唯一输入数。前 16 个 evaluation order 每模型另存 total G×I、raw gradient 和三个 head
component 的原始 float32 [1,3,224,224] map；显示变换不进入本阶段。

## 4. 执行、资源与出口

正式任务固定为 400 图 × 2 模型 = 800 个唯一任务。使用 SQLite WAL 动态队列、
lease/heartbeat/fencing 和最多两次 attempt；每张合格 H20 一个常驻 worker。Torch/CUDA
import 前设 CUBLAS_WORKSPACE_CONFIG=:4096:8，关闭 TF32/autocast，记录解释器、版本、
GPU UUID、既有 PID 和资源。

由于统一路线的 E4/E5 共同承接原 EP5-A/EP5-B，本轮不扩张已冻结总预算：E4-R3 与
E5 合计不超过 136 GPU-hours、132 wall-hours、56 GiB retained；单进程 GPU 24 GiB、
CPU RSS 32 GiB、单任务 1800 秒，磁盘至少剩余 50 GiB。触限保存完成集与缺失清单并
标记 PAUSED_RESOURCE_CAP，不得用部分有利样本通过。

仅当 800/800 唯一任务、所有 raw 数值、曲线、随机重复、图像级分数、hash、资源与
执行身份通过独立 source-recomputing validator 后，出口为：

    ENDPOINT_E5_PASS_SPATIAL_INTERVENTION_COMPLETE_NO_CLAIM_DECISION
    ENDPOINT_E6_REQUIRES_EXPLICIT_USER_REQUEST

该 PASS 只表示 E5 执行和数据合同完整，不表示 C4b SUPPORTED。
