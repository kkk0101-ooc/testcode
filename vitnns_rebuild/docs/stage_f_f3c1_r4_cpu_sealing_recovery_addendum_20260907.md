# Stage F F3C1-R4 CPU sealing recovery addendum（2026-09-07）

F3C1-R3 已完成唯一一次成功的 order-44 numerical-only 五方法工程检查并发布 VALID group terminal 与五个 VALID sample-method terminals，但 validation 在 control-plane `source_binding.json` 的嵌套 digest 字段发生命名碰撞后 fail closed。R3 pending、全部 typed terminals、manifests 和 failure event 必须永久保留，不得改写为 PASS。

本恢复只允许 CPU-only revalidation 与 sealing：

1. 使用新 roots `evidence/stage_f/f3c1_implementation_r4_cpu_seal_20260907/formal/` 与 `reports/stage_f_f3c1_implementation_r4_cpu_seal_20260907/`；
2. 逐文件哈希验证后字节级复制 R3 已发布的 immutable numerical terminals，不重新加载模型、不解码 JPEG、不运行 forward/backward/IG/fidelity；
3. 将 source binding 的内层 source digest 明确命名为 `digest`，再对顶层文档生成唯一 `canonical_digest`；
4. 重建 F3C1 control manifests、validation、false-pass audit、inventory 和 pending→terminal completion。

不得删除或覆盖 v1/R1/R2/R3，不得访问 pilot/formal cohort，不得查看或渲染 map，不得产生科学比较、winner、GO/NO-GO，也不得启动 F3C2、F3C3、F3C4 或 F4。R4 PASS 的唯一下一授权仍为 `F3C2_PILOT_ONLY`。
