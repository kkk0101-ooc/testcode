# ENDPOINT-E5 封存验证器修订 R1

日期：2026-09-18。正式 E5 根已经 800/800 计算完成且保持不可变。首次独立验证在
`RANKER_SCHEMA` fail closed；检查表明这不是生产结果或科学合同失败，而是验证器把
共享 ranker 的已冻结序列表示误读成了带 `shared` 键的对象。

生产 schema 在实现发布前已经固定为：

- `endpoint_component` 与 `regional_taylor` 是按 head 标识的对象；
- `total_gxi` 与 `raw_gradient` 是直接共享给三个 head 的 16 项数组。

本修订只让验证器按上述生产 schema 读取两个共享数组，并同步修正 raw-map 重算路径。
它不修改正式根、task/result bytes、区域分数、排序、随机种子、AUC、阈值、统计口径或
claim 边界。修订后仍必须从 raw maps、曲线和 queue records 独立重算所有 E5 验收量；
若出现任何其他不一致继续 fail closed。
