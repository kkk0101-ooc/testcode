# NNS-ViT Endpoint Scanner：ENDPOINT-E6 执行合同

日期：2026-09-18。授权来自用户对统一研究计划中 `ENDPOINT-E6` 的明确执行请求。
本阶段只做 CPU 统计、结果图、论文正文/附录 release 和复现封存，不运行新模型前向、
不读取或解码 JPEG、不加载 checkpoint，也不启动被阻塞的 Path IG 轨。

## 1. 前置门与输入

正式执行前必须逐项满足：

1. 本地 `HEAD`、tracking ref 和 remote branch 指向同一已发布 commit；
2. E4-R3 terminal 为
   `ENDPOINT_E4_R3_PASS_EXECUTION_CONDITION_SCALED_JVP_AND_FINITE_REPLACEMENT`；
3. E5 terminal 为
   `ENDPOINT_E5_PASS_SPATIAL_INTERVENTION_COMPLETE_NO_CLAIM_DECISION`，且下一授权恰为 E6；
4. E4-R3 和 E5 的独立验证器在本机重新通过；
5. EP0 statistical plan、400 图 cohort、E4/E5 terminal、validation、manifest、index/coverage
   的 SHA-256 与 run spec 完全一致；
6. E6 源码、测试、合同和 run spec 先提交并推送，之后才运行 10,000 次 bootstrap、
   28 项符号检验、Holm 和 claim decision。

正式读集仅含已封存 JSON 元数据与 E4/E5 逐任务 JSON。程序不得读取图像、NPZ 展示图、
checkpoint、队列数据库或日志；每个实际读取文件都进入 `source_read_audit.json`。

## 2. 冻结统计

- 统计单位为图像；head、block 和 20 次随机排列都先聚合到图像级。
- C4a 为 `2 model × 2 reference × 3 component family = 12` 个 family；每图从原始
  component rows 重新计算 `Spearman(m_k, |e_k|)`，精确 ties 使用平均秩。
- C4b 为 `2 model × 2 reference = 4` 个 family；从 curve points 重算 AUC、三 head
  平均以及 Endpoint 相对 `total_gxi`、`raw_gradient`、`regional_taylor`、
  `random_regions` 的四个图像级配对差。随机基线先对每个 head 的 20 次 signed curve
  求均值，再取绝对值和 AUC，最后平均三个 head。
- 共 16 个 family、28 个 test ID；使用 EP0 冻结的 10,000 次 PCG64 image-cluster
  percentile bootstrap、双侧 exact sign test 和一个跨 C4a/C4b 的 28-test Holm family。
- 每 family 至少 360 个共同有效图；每 test 至少 300 个非零图。C4a 的实际阈值/CI
  最大宽度为 0.10/0.30；C4b 的四个 mandatory difference 各为 0.05，family 最大
  CI 宽度为 0.50。
- 每 family 按 `INVALID → INCONCLUSIVE → SUPPORTED → NOT_SUPPORTED` 判定。C4b 只有
  四个 mandatory baseline 的 point estimate、方向和 Holm 条件全通过才可 SUPPORTED。
  C4a 支持不表示优于 reference Taylor。

## 3. 输出与主张边界

正式根为 `reports/nns_vit_endpoint_e6_release_r1_20260918`，至少包含逐图统计输入、
bootstrap、sign-test、Holm、16-family decision、C4a/C4b aggregate decision、
Identity/Correctness/Utility 分层总结、两张结果图、复现清单、源读取审计、验证报告、
artifact manifest 和原子 terminal receipt。

论文 release 同步写入：

- `manuscript/en/endpoint_scanner_results_release.md`；
- `manuscript/zh/endpoint_scanner_results_release.md`；
- `manuscript/notes/endpoint_scanner_e6_evidence_matrix.md`；
- `manuscript/qc/endpoint_scanner_e6_defensive_writing_audit.md`。

Identity 只陈述 Endpoint 与同一目标 Gradient × Input 的数值身份；Correctness 只使用
E3 closure 与 E4-R3 局部 JVP；Utility 只使用本阶段冻结 C4a/C4b 决策。负结果、失败史、
NA、区间和 Path IG 校准失败全部保留。E6 的执行 PASS 表示发布包完整有效，不把科学
`NOT_SUPPORTED` 或 `INCONCLUSIVE` 改写成流程失败。

成功出口：

```text
ENDPOINT_E6_PASS_STATISTICS_AND_PAPER_RELEASE_COMPLETE
NEXT_AUTHORIZATION=NONE_ENDPOINT_PRIMARY_TRACK_COMPLETE
```
