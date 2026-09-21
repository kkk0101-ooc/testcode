# 步骤 4 attempt2：积分执行一致性修订

- 状态：`FROZEN_PROSPECTIVE_ATTEMPT2`
- 原失败：`step4_attempt1_receipt.json`，不可改写

attempt1 中，ViT-B 的端口 PCIA 完整切面与独立批处理 IG 之间最大逐坐标差为 $2.6331\times10^{-5}$，超过冻结逐坐标容差；DeiT-S 通过。两者采用同一 512 节点，但前者逐节点 batch=1，后者按 8 个 alpha 合批。该差异来自 float32 执行顺序，端口一阶闭合、source-summed PEFA、乘积法则、曲率探针、12/12 block 和负控制均通过。

attempt2 不改数学对象或门槛。PCIA→IG 与 Endpoint−IG→R 两项使用同一次端口积分中的 `__full__` 通道作为 path-IG reference；它与各端口使用相同节点、相同 forward 和相同 float32 执行顺序，因而直接检验端口求和是否恢复完整梯度路径积分。独立 batch=8 IG 继续运行并作为成本基线与数值敏感性诊断，不作为端口恒等式的异执行参考。

其余绑定完全继承 attempt1：同一 discovery 图、ground-truth raw logit、两个模型、512 点、9 类切面、12 个 block、原始 normalized RGB 坐标、`1e-5/1e-4` 容差和 16 H20-hours 生产投影门。
