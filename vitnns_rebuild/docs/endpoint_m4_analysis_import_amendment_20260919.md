# Endpoint M4 离线分析导入边界修订 A2

冻结的 `scripts/run_endpoint_m4_analysis.py` 在加载 matplotlib 后检查 `PIL` 不得出现在 `sys.modules`，但 matplotlib 的 PNG writer 会导入 Pillow；这会在读取任何 M4 结果之前固定失败。该条件不是《阶段M_endpointnnsvit.md》的科学或统计门，也不涉及 JPEG 解码。

本修订在正式 GPU 队列运行期间、完整 cohort 汇总之前发布。分析仍设置 `CUDA_VISIBLE_DEVICES=''`，禁止导入 Torch 和 timm，不读取任何 JPEG 或 checkpoint，仅允许 matplotlib 为写出已聚合统计图而导入 Pillow。GPU 生产协议、task manifest、M4 指标、统计 family 和阈值均不改变；现有 GPU 任务不重跑。原冻结脚本的哈希由 `freeze.json` 保留，本修订提交及修改后脚本哈希由 `analysis_import_amendment.json` 在分析启动前记录。
