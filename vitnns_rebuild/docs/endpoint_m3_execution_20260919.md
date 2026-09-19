# M3 冻结执行协议

本轮执行《阶段M_endpointnnsvit.md》的完整 M3，并止于 M3。前置 M2 必须先通过封存哈希、完成清单和 17 项 prompt-to-artifact 验收复核。M3 分为三条不可互相替代的证据链：M3-A 原始 NNS 等价及 ViT 算子测试；M3-B 双模型分层 token-channel Endpoint 审计；M3-C image-level atlas 分析。

## M3-A

原始 NNS 路径显式传播每个激活坐标对 normalized input 的 learning-image，不调用 autograd；比较路径对普通 Torch 前向使用 autograd 并计算 `x * grad`。网络固定为 FCN、CNN、CNN+MaxPool、residual MLP、residual CNN；每个网络、每种 float64/float32 dtype 各 100 个随机输入，每个网络至少 5 个隐藏或输出标量目标。报告 relative L2、最大绝对误差及逐坐标容差比。ReLU 零点沿用 Torch 的零次梯度；MaxPool 使用实际 winner。该等价是局部输入依赖图等价，不是函数值 completeness；bias、CLS、position embedding 不作为图像输入坐标。

独立 ViT 算子审计固定 LayerNorm、GELU、softmax、scaled QK、AV、head concatenation、output projection、residual add、MLP、final LayerNorm 和 classifier。每算子 8 个 float64 Rademacher-like 随机方向，比较解析 JVP、autograd JVP、VJP 对偶恒等式和 9 个 epsilon 的中心有限差分曲线。

## M3-B

从 M1 绑定的共享 400 图中，按 `SHA256(ENDPOINT_M3_COHORT_20260919 | sample_id | image_sha256)` 最小的 100 图冻结选取；两模型严格共享同一图像身份。对每模型全部 12 blocks，固定 16 tokens 和每层 32 channels，共每模型每图 6,144 个 token-channel 标量。

16 tokens 为 CLS 加 15 个 patch token。14x14 grid 被固定切为 3x5 个近似等面积矩形 strata，每个 stratum 取距几何中心最近的 patch，按 row-major tie break。每模型每层 channel 从全宽度中取 `SHA256(ENDPOINT_M3_CHANNEL_20260919 | model | block | channel)` 最小的 32 个，不读取 activation、类别、M1/M2 结果或图质量。

每个目标流式保存 signed sum、absolute mass、L2、positive/negative mass、zero-gradient、cancellation、normalized entropy、196 维 patch absolute mass、196 维 patch signed sum及数值状态。zero absolute mass 时 cancellation、entropy、locality、distance 均为 NA，不加 epsilon。预冻结 10 图/模型、blocks 0/5/11、CLS 与首个 patch、首个 channel 的原始 full-resolution float32 map。

每层 CLS 与首个 patch × 前四 frozen channels 为 96 目标/模型/图的数值子集。生产运行逐目标比较 batched VJP 与 scalar VJP，并在 3 个固定单位范数 Rademacher input directions 和 4 个 epsilon 上报告中心有限差分误差曲线。float32 有限差分按 `atol=2e-3, rtol=5e-2` 的逐目标最佳步长合同验收。每模型前 2 张 frozen image 另做 float64 小批量复核，合同为 `atol=1e-7, rtol=1e-4`。

## M3-C

图像为唯一推断单位。每图先在冻结 units 内取中位数，再跨 100 图报告中位数、10,000 次 image bootstrap 95% percentile CI、配对差值和双侧 exact sign test。每模型 Holm family 固定为五项：late−early entropy（全部有效 units）、late−early locality（patch units）、late−early distance（patch units）、late−early cancellation（全部有效 units）、CLS−patch entropy（全部 blocks）。未定义比较保留并设 p=1。early/middle/late 分别是 blocks 0–3/4–7/8–11。

空间位置描述中，center patch 固定为到 14x14 grid center 的 Chebyshev 距离不超过 3；edge patch 固定为到 grid 边界距离不超过 2。相似性每层每组按冻结 hash 各抽 256 对：same token/different channel 与 different token；分别计算 patch absolute distribution cosine 与 patch signed cosine，不能跨模型把 channel ID 解释为对应神经元。

只有 entropy 上升、own-token locality 下降且 input-dependency distance 上升均通过方向与 Holm 门槛，才允许写抽样单元从局部趋向全局。不允许写全部 ViT neuron scan；正式术语为“对冻结的、层级与空间分层的 6,144 个 token-channel 单元进行了神经元级 Endpoint 审计”。

## 运行与封存

实现、配置和 task manifest 必须先提交并确认远端包含，才读取正式 JPEG。生产采用 float32、汇总 float64，TF32/autocast 关闭、确定性开启，Torch 导入前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。最多 8 张满足 free≥32768 MiB、util≤10% 的 H20，以协作锁和动态 SQLite 原子队列执行，已有 PID 仅记录。失败保留根且不在终态根原地篡改。

最终必须提供：冻结配置和抽样清单、M3-A 全记录、200 个 image-model 任务及 1,228,800 个 target summaries、数值复核、M3-C image-level 统计和 Holm 决策、相似性表、full-map manifest、图表、结果文档、独立验收、artifact manifest 与 terminal receipt。execution PASS 与科学方向性结论严格分离，`next_authorization=STOP_AFTER_M3`。
