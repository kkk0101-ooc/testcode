# Endpoint M6 独立验证修订 R2

M6 首次独立验证在完成 12 类案例的全量重新选择、1,402 个源文件哈希复核和 9 项 raw-map 闭合检查后，停在 M3_MAP_COUNT。原因是 R1 验收器硬编码预期 10 张 M3 map，而 selection manifest 中 m3_early_local_patch 与 m3_cls_patch_contrast 的 patch 单元按冻结 tie-break 合法落在同一坐标；raw package 按唯一 (task, block, token, channel) 去重，因此实际为 8 张 M3 map。

该错误属于验收器覆盖计数假设，不改变：

- 12 类案例的冻结选择；
- selection manifest 或 source access manifest；
- 已选 VJP maps；
- 五张图；
- 任何 M3/M4/M2 科学量或结论。

R2 只将硬编码计数替换为：从 selection manifest 的 selected_units 推导唯一 M3 array key 集，并要求 raw package 中的 M3 keys 与该集合精确相等。首次失败保存在 validation_failure_r1.json；修订后的 verifier 状态显式标为 PASS_R2，并引用本 amendment。

文档验收首次运行还把 M6 插入前全文中 M4 已有的“不是因果证据”边界误判为 M6 overclaim。修订只把该措辞检查限定到独立 M6 结果页、双语 M6 supplement 和 full draft 的 M6 marker 区间；既有正文不改，失败保存在 document_validation_failure_r1.json。
