# NNS-ViT Endpoint Scanner：ENDPOINT-E6 R3 结果

日期：2026-09-18。权威状态：

```text
ENDPOINT_E6_R3_PASS_STATISTICS_AND_PAPER_RELEASE_COMPLETE
C4A_NOT_SUPPORTED
C4B_NOT_SUPPORTED
NEXT_AUTHORIZATION=NONE_ENDPOINT_PRIMARY_TRACK_COMPLETE
```

这里的 PASS 是执行、覆盖、统计和发布包完整性结论；`NOT_SUPPORTED` 是冻结规则下的
科学结论。两者没有混用。权威 terminal 为
`reports/nns_vit_endpoint_e6_release_r3_20260918/terminal_receipt.json`。

## 1. 执行与验证

- 全程 CPU-only，`CUDA_VISIBLE_DEVICES=''`；GPU-hours 为 0；未加载模型、未解码图像。
- 从 E4-R3/E4-R2 与 E5 的封存逐任务 JSON 独立读取并校验 manifest SHA-256；共形成
  1,600 个 image-model-reference 行。
- 统计单位为图像。对 16 个冻结 family 各运行 10,000 次 PCG64 paired image-cluster
  bootstrap；对 28 个冻结 test ID 运行双侧精确符号检验和一个全局 Holm family。
- 每个 family 均有 400 个有效图像，所有 28 个检验均有 400 个非零图，未触发
  `INVALID` 或 `INCONCLUSIVE`。
- 独立验证器不导入 Torch/timm/PIL，从原始 component rows 与 curve points 重算逐图值、
  bootstrap、sign test、Holm 和 family decision，最终通过。

## 2. C4a：有限 activation replacement 关联

主统计量是每图 `Spearman(m_k, |e_k|)`，family 点估计为 400 图中位数。实际效应阈值
为 0.10，CI 最大宽度为 0.30。

| 模型 | Reference | 成分族 | 中位数 | 95% CI | Holm p | 判定 |
|---|---|---|---:|---:|---:|---|
| ViT-B/16 | normalized-zero | heads | 0.2909 | [0.2813, 0.2982] | 2.17e-119 | SUPPORTED |
| ViT-B/16 | normalized-zero | routing/value | 0.0996 | [0.0748, 0.1226] | 3.00e-14 | NOT_SUPPORTED |
| ViT-B/16 | normalized-zero | residual branches | 0.1774 | [0.1548, 0.1952] | 2.56e-46 | SUPPORTED |
| ViT-B/16 | blur | heads | 0.3063 | [0.2957, 0.3181] | 2.17e-119 | SUPPORTED |
| ViT-B/16 | blur | routing/value | 0.1565 | [0.1322, 0.1774] | 9.96e-29 | SUPPORTED |
| ViT-B/16 | blur | residual branches | 0.2091 | [0.1817, 0.2374] | 1.46e-41 | SUPPORTED |
| DeiT-S/16 | normalized-zero | heads | 0.1910 | [0.1769, 0.2052] | 1.88e-76 | SUPPORTED |
| DeiT-S/16 | normalized-zero | routing/value | 0.1065 | [0.0817, 0.1291] | 2.11e-10 | SUPPORTED |
| DeiT-S/16 | normalized-zero | residual branches | 0.1352 | [0.1148, 0.1643] | 4.35e-19 | SUPPORTED |
| DeiT-S/16 | blur | heads | 0.1817 | [0.1688, 0.1951] | 3.82e-72 | SUPPORTED |
| DeiT-S/16 | blur | routing/value | 0.1600 | [0.1278, 0.1822] | 9.90e-31 | SUPPORTED |
| DeiT-S/16 | blur | residual branches | 0.2287 | [0.2035, 0.2504] | 3.70e-54 | SUPPORTED |

11/12 family 达到全部支持条件。`vit_b16 × normalized_zero × routing_value` 的
中位数为 0.099565，只比冻结阈值 0.10 低 0.000435；关联方向为正且通过 Holm 校正。
因此，**RQ3 整体得到支持，但存在一个预注册边界例外**。机器可审计的 12/12 omnibus
C4a 仍为 `NOT_SUPPORTED`，因为不能在看到结果后移动阈值。这个严格字段不应被解释成
“RQ3 整体失败”。C4a 也不检验 Endpoint 是否优于 Taylor。

## 3. C4b：空间信息增益

每个 family 必须同时相对 `total_gxi`、`raw_gradient`、`regional_taylor` 和
`random_regions` 达到中位差 0.05、正方向和 Holm 显著。四个 family 均为
`NOT_SUPPORTED`：

| 模型 | Reference | Endpoint−total G×I | Endpoint−raw gradient | Endpoint−regional Taylor | Endpoint−random | 判定 |
|---|---|---:|---:|---:|---:|---|
| ViT-B/16 | normalized-zero | -0.0009 | -0.0000 | -0.0234 | 0.0011 | NOT_SUPPORTED |
| ViT-B/16 | blur | -0.0010 | -0.0007 | -0.0237 | 0.0042 | NOT_SUPPORTED |
| DeiT-S/16 | normalized-zero | -0.0018 | -0.0006 | -0.0299 | 0.0107 | NOT_SUPPORTED |
| DeiT-S/16 | blur | 0.0002 | -0.0002 | -0.0356 | 0.0085 | NOT_SUPPORTED |

Endpoint component ranking 没有在任一注册 family 中同时超过四个 mandatory baseline；
其相对 regional Taylor 的中位差在四个 family 中均为负。因而不支持“完整输入分辨
component map 在该空间干预任务中带来联合信息增益”的 C4b 主张。

## 4. 三层结论与历史边界

- **Identity**：对同一输入端标量目标，基础 Endpoint 图是
  $x\odot\nabla_x s(x)$，即 Gradient × Input。
- **Correctness**：E3-R2 的 76,800 项正向闭合检查和 28,800 项负例通过；E4-R3 的
  6,144 项条件尺度解析 JVP 检查通过。
- **Utility**：RQ3 整体支持（11/12 family），严格 all-family C4a 为 `NOT_SUPPORTED`；
  RQ4/C4b 为 `NOT_SUPPORTED`。完整子 family 结果保留。
- **Path IG**：EP2-R1 的 ViT-B 数值校准失败保持 immutable；不发布跨模型 Endpoint–Path
  fidelity、stability 或成本—质量结论。
- **恢复历史**：E6-R1 在推断前因未定义变量 fail closed；E6-R2 的数值结果与 R3
  逐字节相同，但论文 Markdown 格式不合格；R3 使用新根完整重算并成为唯一权威 release。

## 5. 发布产物

- `reports/nns_vit_endpoint_e6_release_r3_20260918/`：逐图统计输入、bootstrap、sign test、
  Holm、family/aggregate decisions、两张 SVG 结果图、source read audit、reproduction
  manifest、validation report、artifact manifest 和 terminal receipt；
- `papers/nns_vit_endpoint/en/endpoint_scanner_results_release_r3.md` 与
  `papers/nns_vit_endpoint/zh/endpoint_scanner_results_release_r3.md`：同步双语论文结果；
- `papers/nns_vit_endpoint/notes/endpoint_scanner_e6_evidence_matrix_r3.md`：claim-to-evidence 映射；
- `papers/nns_vit_endpoint/qc/endpoint_scanner_e6_defensive_writing_audit_r3.md`：防御性写作审计。
- `reports/nns_vit_endpoint_unified_paper_figures_r2_20260918/`：28 张 Endpoint 论文图、
  84 个 PNG/PDF/SVG 导出、双语 caption、逐图分析与封存回执。

目录迁移说明：E6 封存回执、run spec 和 reproduction manifest 保留生产时的
`manuscript/...` 路径作为 provenance；当前活动论文根已迁移至
`papers/nns_vit_endpoint/`，活动验证器按
[论文目录迁移修订 R1](NNS_ViT_Endpoint_Scanner论文目录迁移修订R1_20260918.md)
解析旧路径并继续校验 bytes 与 SHA-256。
