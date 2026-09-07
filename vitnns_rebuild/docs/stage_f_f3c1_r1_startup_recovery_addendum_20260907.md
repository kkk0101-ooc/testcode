# Stage F F3C1-R1 startup recovery addendum（2026-09-07）

F3C1 v1 在 order-44 engineering check 的模型加载、JPEG 解码和 attribution 开始前 fail closed。原因是当前 Torch 在 CUDA context 初始化前拒绝 `reset_peak_memory_stats("cuda:0")`。v1 pending receipt 和 failure event 必须永久保留，不得改写为 PASS。

本恢复只允许：

1. 使用新 roots `evidence/stage_f/f3c1_implementation_r1_20260907/formal/` 与 `reports/stage_f_f3c1_implementation_r1_20260907/`；
2. 在重置 peak-memory 统计前显式选择并初始化同一 CUDA device；
3. 原样复用 F3C0 冻结方法、order-44、模型、预处理、IG schedule、阈值、五方法 roster、统计实现和全部数据访问边界；
4. 重新从 clean local/tracking/remote implementation commit 运行唯一一次 numerical-only order-44 engineering/profile check。

不得删除或覆盖 v1，不得读取 pilot/formal cohort，不得渲染或人工查看 map，不得产生科学比较、winner、GO/NO-GO，也不得启动 F3C2、F3C3、F3C4 或 F4。R1 PASS 的唯一下一授权仍为 `F3C2_PILOT_ONLY`。
