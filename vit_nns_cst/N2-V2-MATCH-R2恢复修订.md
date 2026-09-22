# N2-V2 low-level matching R2 恢复修订

日期：2026-09-22

## 1. 触发原因

R1 生产前的控制面探索读取了 legacy N2 producer/validator 代码。该读取不包含 checkpoint、模型输出或 behavior outcome，也未改变 V2 contract 中的指标和阈值，但它超出本轮明确列出的上游读取白名单。因此 R1 不作为权威 matching PASS，另由 post-hoc invalidation receipt 保留。

## 2. R2 不变项

- I7 accepted registry、43 张绑定 scene、12 组 recipient 原图与 mask、V2 contract 和 legacy lineage ledger 不变。
- 五项指标、`2.5` per-metric caliper、`1.25` aggregate caliper、`8/8`、unmatched rate `0`、background reuse max `1` 不变。
- 选择目标、SHA-256 tie-break、reserve 排序和 outcome-blind 边界不变。
- R2 只做等价缓存和安全下界剪枝。缓存等价误差必须小于 `1e-12`；剪枝只删除“候选自身 objective + 其余 7 个 recipient 的乐观最小值”已严格劣于可行 incumbent 的候选。

## 3. R2 额外审计

R2 生产命令必须设置空 `CUDA_VISIBLE_DEVICES` 并由 `strace openat` 记录实际文件读取。完成后只允许出现：matching 实现、Python/动态库系统依赖、I7 accepted registry、其绑定的 scene 原图、12 组 recipient 原图与 mask、V2 contract、legacy lineage ledger，以及 R2 自身输出。任何 checkpoint、behavior outcome 或其他 evidence 输入出现，R2 立即 `INVALID_PROTOCOL`。

R2 通过后仍停止在输入层；行为门和 N3 均需新的显式用户授权。
