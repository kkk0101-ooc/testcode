# Stage F F3C1-R5 manifest sealing recovery addendum（2026-09-07）

F3C1-R4 在 CPU-only sealing validation 中 fail closed：`attention_binding.json` 将已签名 attention manifest 的 `canonical_digest` 展开到顶层，覆盖了顶层自签名；whole-root inventory 也存在同类字段命名风险。R4 pending、manifests 和 failure event 必须永久保留，不得改写为 PASS。

本恢复只允许：

1. 使用新 roots `evidence/stage_f/f3c1_implementation_r5_cpu_seal_20260907/formal/` 与 `reports/stage_f_f3c1_implementation_r5_cpu_seal_20260907/`；
2. 从 R3 VALID numerical terminals 逐文件哈希后字节级复制，不重新加载模型、不解码 JPEG、不运行 forward/backward/IG/fidelity；
3. 所有嵌套 digest 使用明确字段名：`attention_manifest.canonical_digest`、`source_tree_digest`、`file_inventory_digest`；每个顶层 JSON 只保留自己的 `canonical_digest`；
4. 重建 control manifests、validation、false-pass audit、inventory 和 pending→terminal completion。

不得删除或覆盖 v1/R1/R2/R3/R4，不得访问 pilot/formal cohort，不得查看或渲染 map，不得产生科学比较、winner、GO/NO-GO，也不得启动 F3C2、F3C3、F3C4 或 F4。R5 PASS 的唯一下一授权仍为 `F3C2_PILOT_ONLY`。
