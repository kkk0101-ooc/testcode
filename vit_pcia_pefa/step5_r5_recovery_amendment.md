# 步骤 5 r5 前瞻性恢复修订

r4 已将显存门前瞻提高为 40 GiB，并继承 8 个 DeiT-S 结果。首批 ViT-B 尚未产生结果时，静态预演发现独立验证器把预访问测试摘要硬编码为 `7 passed`，而 r4 冻结时实际通过 8 项测试；该条件会制造确定性的假失败。r4 因此在新 ViT-B outcome 落盘前停止，以 `TERMINAL_INFRASTRUCTURE_FAIL` 封存。

r5 只修复验证器，使其验证冻结记录确实为成功的 pytest 摘要，不绑定易漂移的测试数量。r4 的 40 GiB per-worker 显存门与所有科学对象、样本、算法、节点、精度、容差、干预、统计量、阈值和 claim 规则完全不变。r5 按 canonical digest 继承 8 个 DeiT-S 结果，只调度 8 个缺失的 ViT-B；最终资源账本累计 r1-r5。
