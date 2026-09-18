# NNS-ViT Endpoint Scanner：Endpoint / Path 研究轨道拆分修订

日期：2026-09-18。该修订在任何新的正式科学 outcome 访问前签发，基线 commit
为 `ef0c500c2472c4e3db23a06877806254c2eea0bd`。它不覆盖 EP2-R1，不把失败
改成通过，而是明确校准依赖范围。

```text
ENDPOINT_PRIMARY_TRACK_AUTHORIZED_BY_AMENDMENT
PATH_DEPENDENT_TRACK_BLOCKED
EP2_R1_REMAINS_IMMUTABLE
ORIGINAL_EP3_NOT_AUTHORIZED
```

机器可读合同：[scope amendment](../config/nns_vit_endpoint/endpoint_primary_track_amendment_r1.json) ·
[closed schema](../config/nns_vit_endpoint/endpoint_primary_track_amendment_r1.schema.json) ·
[validation report](../reports/nns_vit_endpoint_scope_amendment_r1_20260918/validation_report.json) ·
[terminal receipt](../reports/nns_vit_endpoint_scope_amendment_r1_20260918/terminal_receipt.json)。

## 1. 修订理由与不变事实

Endpoint 的定义为

$$
L_s^{\mathrm{Endpoint}}(x)=x\odot\nabla_xs(x),
$$

不需要 baseline forward、路径积分或积分阶数。EP2-R1 校准的是 normalized-zero
直线路径上的 full-target Path IG；其 ViT-B 数值失败只阻塞依赖高精度 Path map
的成本—质量对照。它不改变 EP1 已验证的 Endpoint 数学身份、ViT 算子覆盖、
结构切面闭合和工程入口。

基础 Endpoint 数值上就是同一内部标量目标上的 Gradient × Input。论文贡献边界
固定为：原始 NNS endpoint 原则到 ViT 的适配、内部目标注册与扫描、结构化
endpoint 分解及其干预审计；不把基础公式写成新归因算法，不声明普遍替代 IG。

EP2-R1 保持：

```text
EP2_R1_PATH_NUMERICAL_CALIBRATION_FAILED
SCIENTIFIC_RESULTS_NOT_RUN_CALIBRATION_ONLY
EVALUATION_ACCESS_FALSE
EP3_NOT_AUTHORIZED
```

## 2. Endpoint 主轨

为避免与仓库历史 Stage E 混淆，新阶段使用 `ENDPOINT-E*`：

| 阶段 | 任务 | 当前状态 |
|---|---|---|
| ENDPOINT-E0 | 本范围修订与协议冻结 | 本轮完成 |
| ENDPOINT-E1 | 数学、算子、闭合和工程合同 | 由 EP1 R11 证据重绑定满足 |
| ENDPOINT-E2 | 注册目标 atlas 与规模/资源审计 | 需用户另行明确授权 |
| ENDPOINT-E3 | 正式 cohort 的全 block 结构闭合 | 需前置合同通过和明确授权 |
| ENDPOINT-E4 | 局部与有限 activation replacement | 需明确授权 |
| ENDPOINT-E5 | 空间定向干预与 mandatory baselines | 需明确授权 |
| ENDPOINT-E6 | 统计汇总与论文 release | 所有启用阶段完成后执行 |

ENDPOINT-E2 只允许原 EP0 冻结的 400 evaluation rows、两个模型、457/385 个
registered targets 和原 atlas 合同。只执行 canonical Endpoint。Path IG、输入
fidelity/stability、有限干预和 utility claim 在该阶段均禁止。C2 只报告完整覆盖、
targets/sec、images/sec、peak memory、stored bytes/target 和摊销成本，不声称
Endpoint 优于 Gradient × Input 或 Path IG。原 EP0 的 EP3 入口仍是未授权状态。

ENDPOINT-E3 至 E5 沿用 EP0 已冻结的 target、cut、reference、C4a/C4b、阈值、
mandatory baselines 和 28-test Holm family。轨道拆分不重写这些 endpoint-specific
统计选择。局部方向合同只属于 correctness；finite activation replacement 和空间
定向干预才属于 utility。

## 3. Path 轨

- `P-IG1`：既有 20 图结果只作历史探索性边界证据。
- `P-IG2`：EP2-R1 两模型数值校准失败，immutable。
- `P-IG3`：可选未来恢复，必须有新的前瞻性 amendment；当前授权为 `NONE`。

任何依赖 high-precision Path IG 的 Endpoint–Path fidelity、稳定性、成本或跨模型
主张均保持 blocked。不得只选已通过的 DeiT-S，不得增加 M、放宽阈值或删除
ViT-B/难样本来改变 P-IG2 结论。

## 4. 本轮访问和下一授权

本修订只读取已提交协议、EP1/R11 与 EP2-R1 receipt/report 并重算文件哈希；
新图像读取、模型 forward、GPU job 均为 0。下一可请求阶段为：

```text
ENDPOINT_E2_ATLAS_REQUIRES_EXPLICIT_USER_REQUEST
```

这只开放未来 Endpoint atlas 阶段的申请资格，不是本轮直接执行授权。
