# NNS-ViT Endpoint Scanner：ENDPOINT-E6 封存验证器修订 R1

日期：2026-09-18。E6-R2 正式根已原子完成，生产 terminal 为
`ENDPOINT_E6_PASS_STATISTICS_AND_PAPER_RELEASE_COMPLETE`。首次独立封存验证在
`READ_WHITELIST` 处停止：验证器错误地要求每个允许输入都以 `.json` 结尾，而 E6 run spec
明确绑定并读取统一计划 Markdown。生产读取审计记录的 1,623 个路径中，唯一非 JSON 路径
就是该计划；没有 JPEG、checkpoint、NPZ、SQLite 或日志读取。

本修订不改生产根、统计、seed、bootstrap、Holm、family decision、论文文本或 terminal。
验证器改为从 run spec、predecessor 路径表、E4 combined index 和 E5 固定任务路径机械构造
精确允许集，然后要求读取审计的路径集合与之完全相等。原 producer commit 中的生产验证器
字节仍由 `reproduction_manifest.json` 校验；修订后的验证器和本修订文件另绑定到当前提交。
