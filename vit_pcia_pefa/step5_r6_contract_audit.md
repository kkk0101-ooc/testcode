# 步骤 5 r6 合同覆盖审计

r6 完成 16/16 任务并签发 `NOT_SUPPORTED_EMPIRICAL_VALUE`，其原始结果、终态和 artifact manifest 保持不可变。完成后的逐条人工审计发现，r6 不能作为步骤 5 最终合同覆盖证据：队列 seed 沿用了旧 Endpoint 队列命名空间，没有按方法合同的 `ENDPOINT_PCIA_PEFA|step5|task_id` 构造；任务只保存单端口 Omega，没有逐任务验收完整合法切面的 Omega 零和；运行身份没有保存可恢复的 Python、NumPy、Torch CPU/CUDA RNG state，也未记录 driver 与 freeze-time git dirty snapshot。

这些缺口来自 `method_contract.md` 的既有条款，不来自 r6 的效应方向。r6 的科学负结果不被改写，但其 `PASS_STEP5_PROMPT_TO_ARTIFACT_AUDIT` 只表示当时验证器覆盖范围内通过，不可解释为全合同覆盖。恢复必须建立新 root、明确记录 r6 outcome 已知、全量重跑 16 个任务，并将 r1-r6 资源计入累计账本。步骤 6 未启动。
