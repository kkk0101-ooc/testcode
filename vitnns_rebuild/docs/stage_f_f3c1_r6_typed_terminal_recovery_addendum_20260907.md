# Stage F F3C1-R6 typed-terminal recovery addendum（2026-09-07）

F3C1-R5 已完成 terminal sealing 并创建 archive ref，但独立提交后审阅发现 generic typed-terminal writer 没有强制拒绝 FAILED/NA 状态携带数值数组。R5 的五个实际 terminal 均为 VALID，数值结果不受影响；但 false-pass 合同要求该攻击必须由实现本身拒绝，因此 R5 completion 被前瞻性标记为 `BLOCKED_SUPERSEDED`，其 commit、archive ref、completion 和 payload 均保留。

本恢复只允许：

1. 使用新 roots `evidence/stage_f/f3c1_implementation_r6_cpu_seal_20260907/formal/` 与 `reports/stage_f_f3c1_implementation_r6_cpu_seal_20260907/`；
2. 在 typed-terminal publish 和 validate 两侧同时强制 VALID 必须有 payload、FAILED/NA 必须没有数值 payload；
3. 从 R5 terminals 逐文件哈希后字节级复制，并对当前代码重新构建 source binding；
4. CPU-only 重跑 tests、false-pass、validation、inventory 和 pending→terminal sealing。

不得重新加载模型、解码 JPEG、运行 forward/backward/IG/fidelity，不得访问 pilot/formal cohort，不得查看或渲染 map，不得产生科学比较、winner、GO/NO-GO，也不得启动 F3C2、F3C3、F3C4 或 F4。R6 PASS 后将 archive ref 前移到 R6 terminal commit；唯一下一授权仍为 `F3C2_PILOT_ONLY`。
