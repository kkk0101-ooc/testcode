# 阶段 A：先把普通 ViT 分析做正确

## 一句话说明

阶段 A **还没有进入 NNS 阶段**。

这一阶段做的是普通 ViT 分析：让同一张图片经过真实的 ViT 模型，正确取得模型预测、每一层的 Q/K/V、attention 和层输入输出，并证明这些数据没有被提取程序改坏。

真正的 NNS learning image、像素贡献传播和 NNS-ViT 方法，会从下一阶段 B 才开始实现。

## 为什么不能直接开始 NNS

旧项目虽然已经生成了很多 NNS 图片和张量，但审计发现其中存在几个基础问题：

- Q、K、V 的拆分顺序有误；
- attention 使用了错误的缩放因子；
- 扫描器手工模拟的前向过程与真实 timm ViT 不一致；
- ImageNet 标签映射不正确；
- 一些脚本失败后仍会显示“完成”。

如果不先解决这些问题，即使换成更大的模型或生成更多图片，也无法判断 NNS 结果是否来自真实模型。

因此，阶段 A 的任务相当于先建立一把可靠的“尺子”。阶段 B 实现 NNS 后，所有 NNS 中间量和最终结果都必须与这条普通 ViT 基线对齐。

## 阶段 A 做了什么

### 1. 固定一个可信模型

当前统一使用：

```text
vit_base_patch16_224.augreg2_in21k_ft_in1k
```

这是标准 ViT-B/16，包含 12 层 Transformer Block、12 个注意力头和约 8657 万参数。

正式运行只加载经过哈希校验的纯 `state_dict`。完整模型 pickle 只用于兼容性核对，不作为后续实验的主要加载方式。

### 2. 使用模型真正需要的图像预处理

预处理直接读取该模型的 timm 配置：

- 图片转换为 RGB；
- 短边缩放到 248；
- 中心裁剪为 224×224；
- 使用 bicubic 插值；
- mean 和 std 都是 `(0.5, 0.5, 0.5)`。

旧脚本中部分预处理与该模型不匹配，阶段 A 已统一修正。

### 3. 修正 ImageNet 标签

阶段 A 使用官方 ImageNet WNID 与类别索引映射：

- `imagenet-scanner` 中的 100 张图片已按 synset 目录获得正确标签；
- 平铺的 500 张 `ILSVRC2012_val` 图片已通过官方 devkit 恢复真实标签。

现在不再使用“目录枚举位置”“排序后每 50 张一类”或模型预测来代替真实标签。

### 4. 从真实 ViT 前向中提取中间量

程序通过 hook 观察 timm 的真实前向过程，可以取得：

- 12 层的输入和输出；
- 每层的 Q、K、V；
- 每层每个头的 attention；
- 最终 1000 类 logits。

这些属于 **direct ViT 数据**，不是 NNS 结果。

### 5. 记录每次结果的来源

每次正式运行都会记录：

- 使用的模型和模型哈希；
- 输入图片和真实标签；
- 图像预处理方式；
- Python、PyTorch、timm 等版本；
- Git 提交；
- 设备、随机种子和数值误差；
- top-1、top-5 和原始 logits。

这样后续看到一个实验结果时，可以明确知道它由什么代码、模型和输入生成。

## 阶段 A 没有做什么

为了避免概念混淆，阶段 A 明确没有完成以下工作：

- 没有实现新的 NNS-ViT；
- 没有生成可信的 NNS learning image；
- 没有实现像素贡献的逐层传播；
- 没有证明 NNS 优于 attention、Grad-CAM 或其他方法；
- 没有开始论文级 NNS 批量实验；
- 没有把旧 NNS 图修复成有效论文结果。

阶段 A 的成果是“可信的普通 ViT 参照组”，不是论文最终方法。

## 验证结果

阶段 A 已在 NVIDIA H20 上完成验证：

| 检查内容 | 结果 | 说明 |
|---|---:|---|
| 标准 ViT 前向与可观测前向的 logits 最大误差 | `8.58e-06` | 小于要求的 `1e-5` |
| 添加 hook 前后 logits 误差 | `0.0` | 提取过程没有改变预测 |
| 添加 hook 前后各层输入输出误差 | `0.0` | 提取过程没有改变层状态 |
| 重算 attention 与模型真实 attention 误差 | `0.0` | Q/K 顺序和缩放正确 |
| 本地完整模型与 state_dict 模型误差 | `0.0` | 两种模型资产一致 |

用于端到端验证的样本真实类别索引为 `732`，模型 top-1 也预测为 `732`。

自动测试结果：

```text
13 passed, 1 deselected
No broken requirements found.
```

## 当前代码在哪里

`vitnns_rebuild/` 是当前正式实现目录：

- `src/vitnns/modeling.py`：模型加载和预处理；
- `src/vitnns/dataset.py`：数据集和 manifest；
- `src/vitnns/labels.py`：ImageNet 官方标签映射；
- `src/vitnns/extractor.py`：普通 ViT 中间量提取；
- `src/vitnns/provenance.py`：结果来源记录；
- `src/vitnns/runner.py`：阶段 A 验证入口；
- `tests/`：阶段 A 自动测试。

旧目录 `49_vit `、`scan_vit/` 和 `ppt_results_nns/` 继续保留用于追溯，但不再作为新实现目录。

## 下一步：阶段 B

阶段 B 才正式进入 NNS-ViT。它将以阶段 A 提取出的真实模型中间量为输入，逐步实现：

1. 明确 NNS 要解释的目标，例如类别 logit、某个 attention head 或 Value 路径；
2. 为 LayerNorm、GELU、残差连接、QK 乘法和 softmax 定义贡献传播规则；
3. 生成真正可验证的 learning image；
4. 检查像素贡献之和是否与目标激活变化守恒；
5. 与 direct attention、attention rollout、Grad-CAM 等方法做同图公平比较。

只有阶段 B 的传播规则和守恒测试通过后，才进入阶段 C 的多图片小型试验。
