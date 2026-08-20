# 阶段 B：受约束地重写 NNS-ViT 归因

## 当前结论

阶段 B 已建立可审计的实现，并完成 12 层开发 audit 与独立的真实重复性/sanity 验证。下列 12 层运行来自 Git `1854838` 的 dirty tree，provenance 为 `formal=false`，且该次运行关闭了 repeatability 和 sanity；它不是正式论文产物。

## 方法的精确定义

当前实现的方法名是 `conditional_fixed_attention_value_path_ig`。它解释一个真实 timm block 内、输出投影之前的单一标量：

```text
O_h[q, c] = sum_j A_h[q, j] * V_h[j, c]
```

目标由 block、attention head、query token 和 head channel 唯一确定。程序使用真实 forward 捕获的中间量，不自行重演 timm Transformer Block。归因路径仅改变输入到该 block Value path 的变量；attention `A` 固定为端点真实 forward 的值。因此它是“固定 attention 条件下的 Value-path integrated gradients”，不是完整 attention 归因。

`class_target` 只记录 predicted、ground-truth 或显式类别索引的上下文，不改变上述守恒目标。当前 learning image 是原生输入空间的 signed RGB float32 attribution，形状为 `3 x 224 x 224`；metric-ready 图由 RGB 通道求和得到 signed `224 x 224` float32。resize、百分位增强、max-pool、Gaussian 平滑和旧式 `0.7 class_feature + 0.3 attention` 均不是方法定义；若用于展示，必须是显式、可选、默认关闭的 display transform。

## 方法边界

- 已提供 `full_model_ig` 作为覆盖真实 LayerNorm、GELU、residual、QK 乘法和 softmax 的通用审计基线，但不把它冒充 NNS 或 Value-path。
- 已提供 `conditional_fixed_attention_model_ig` 解释固定某层 A 后的类别 logit；它仍包含 residual 等旁路，不能称为纯 Value-path。
- 对上述算子的覆盖采用真实 forward + autograd 路径积分；没有发明逐元素比例式的手工传播规则。
- 不把固定 attention Value path 等同于完整 NNS-ViT。
- 不把插值后的 patch 图称为原生像素级证据。
- 不把旧扫描器、旧 learning image 或渲染 PNG 当作可信定量结果。
- 不因开发运行通过 conservation 就声称 formal 验收完成。

## 配置、命令与预算

配置采用严格 JSON schema，未知字段失败关闭。默认 Gauss-Legendre IG schedule 为 `[512, 1024, 2048, 4096, 8192]`，存储预算为 64 MiB，`compute_budget.max_forward_calls` 和 `max_backward_calls` 均为 30,000。预检会同时拒绝存储或计算预算超限的任务。

```bash
vitnns-stage-b estimate --config config/stage_b.json

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 vitnns-stage-b attribute \
  --config config/stage_b.json \
  --device cuda \
  --relative-image n03976467/n03976467_9546.JPEG \
  --output-dir outputs/stage_b
```

已知样本标识仅为 `n03976467/n03976467_9546.JPEG`；本文不根据 WNID 猜测图片俗名。

## 留存边界

- `debug` 用于开发诊断，不能自动升级为论文证据。
- `paper` 只保存预注册目标的紧凑二维 metric-ready 数组；禁止 PNG、逐像素文件和逐维文件。
- `audit` 保存 float32 RGB raw attribution 和可复算 paper 指标的 float32 signed 2D metric-ready 图。

每次运行必须在写入前估算预算，写入时按真实序列化字节数再次 fail closed。输出目录不得静默覆盖或续写不完整结果。定量指标读取 raw/metric-ready tensor，不读取 display PNG。

## 当前开发 audit 证据（非正式）

- Git HEAD：`1854838`；工作树 dirty；`formal=false`。
- 设备：NVIDIA H20，GPU 1；设置 `CUBLAS_WORKSPACE_CONFIG`。
- 样本：`n03976467/n03976467_9546.JPEG`。
- 本次固定 attention endpoint parity：12 层最大误差均为 `0.0`；阶段 A 另行记录的 fused/unfused logits 最大误差为 `8.58e-06`。
- IG schedule：`[512, 1024, 2048, 4096, 8192]`。

| block | relative conservation error |
|---:|---:|
| 0 | 0.000716 |
| 1 | 0.000437 |
| 2 | 0.000002 |
| 3 | 0.0000535 |
| 4 | 0.000554 |
| 5 | 0.0000566 |
| 6 | 0.000642 |
| 7 | 0.001498 |
| 8 | 0.000836 |
| 9 | 0.002175 |
| 10 | 0.004749 |
| 11 | 0.004924 |

所有层均小于 `1%`。实际 artifact 字节数 `9,164,616`，小于预估 `9,748,480`，也小于 cap `67,108,864`。留存内容为 12 个 raw RGB float32、12 个 metric signed 2D float32、0 个 PNG。开发 artifact 位于 `/tmp/vitnns_stage_b_audit`；该路径是临时开发证据位置，不是稳定论文归档。

这次 12 层运行关闭了 repeatability 和 sanity，不能据此通过这两项门禁。另有独立的真实 block 0 开发运行：重复运行 attribution 最大绝对差为 `0`；输入扰动后的 relative L1 change 为 `0.08975`，达到预设下限 `0.05`。该独立结果不能替代 clean-tree formal 全配置复跑。

## 下一步

阶段 B 的实现门槛已经由单元测试、12 层 conservation audit、独立重复性/sanity 和 retention replay 覆盖。提交后可在 clean commit 上生成新的 `formal=true` 持久产物；这用于绑定最终 Git SHA，不改变已验证的算法门槛。随后进入阶段 C 的 3–5 图校准与同 20 图 pilot，不能直接跳到批量主实验。
