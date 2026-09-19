# M3 数值验证恢复修正案

M3 的 200 个 atlas 任务全部完成后，冻结验收命令在 float32 中心有限差分门槛处 fail-closed。不可变记录 `initial_numeric_validation_failure.json` 显示：57,600 个 target-direction 条件中 36 个失败，分布于 10 个 image-model task；最坏条件在四个冻结步长中仍出现方向符号不一致。生产采集本身没有失败，19,200 个 batched VJP 对 scalar VJP 的同图精确控制通过，预冻结四图的 1,152 个 float64 finite-difference 条件全部通过。

本修正案不修改图像、token、channel、统计方法、科学假设或既有 float32 门槛，也不删除初始 FAIL。恢复范围由初始失败文件机械决定：只对其中 10 个 task 的完整冻结 96-target × 3-direction 子集做 float64 多步长中心有限差分；不能根据 atlas 指标或预期科学方向增删任务。使用原 checkpoint、原 normalized input bytes、原 target registry 和原 Rademacher directions，epsilon 固定为 `[1e-3,3e-4,1e-4,3e-5,1e-5,3e-6]`，逐 target-direction 以最佳冻结步长满足 `|FD-JVP| <= 1e-7 + 1e-4|JVP|`。

若 10 个恢复任务全部通过，则数值状态签发为 `PASS_FLOAT64_RECOVERY_WITH_RETAINED_FLOAT32_FAILURE`，原始 36 项 float32 FAIL 永久保留并在结果文档披露；这只恢复导数验证的数值精度结论，不将失败解释为科学效应。若任一 float64 条件失败，M3 停止并保持数值 FAIL。atlas 的 1,228,800 个正式目标不重算。
