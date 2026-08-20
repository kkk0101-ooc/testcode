# 阶段 A：可审计的 ViT 直接前向基线

## 1. 阶段目标

阶段 A 的目标是在重写 NNS-ViT 之前，先建立一条数值可信、标签正确、结果可追溯的 ViT 直接前向基线。该阶段不复用旧扫描器的手工 Transformer 前向，而是以真实 timm 前向传播作为唯一数值来源。

本阶段代码位于 `vitnns_rebuild`。旧目录 `49_vit ` 只作为模型与数据归档读取，不在其中继续叠加修复脚本。

## 2. 已完成内容

### 2.1 唯一模型基线

- 模型：`vit_base_patch16_224.augreg2_in21k_ft_in1k`
- 正式加载来源：本地纯 `state_dict`
- state_dict SHA-256：`2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0`
- Transformer Block：12 层
- 注意力头：12 个
- head dimension：64
- 224×224 输入对应 token 数：197
- 参数量：86,567,656

完整模型 `.pth` 是 Python pickle，不作为正式运行依赖。它只在文件哈希匹配且显式允许不安全反序列化时，用于兼容性对照。

### 2.2 正确的模型预处理

预处理配置从该 checkpoint 的 timm 元数据读取：

- 图像统一转换为 RGB；
- bicubic 插值；
- 短边缩放到 248；
- 中心裁剪为 224×224；
- mean 为 `(0.5, 0.5, 0.5)`；
- std 为 `(0.5, 0.5, 0.5)`；
- 开启 antialias。

这与旧脚本硬编码的 torchvision ImageNet mean/std 不同。后续实验必须复用这一统一预处理，避免模型与输入协议错配。

### 2.3 官方标签映射与数据清单

已实现确定性的 manifest 生成流程：

- `imagenet-scanner`：10 个 synset、100 张有标签图像；
- `ILSVRC2012_val`：500 张平铺验证图，通过官方 ILSVRC2012 devkit 恢复真实标签。

标签规则如下：

- synset 目录通过 ImageNet-1k 官方 WNID 顺序映射到模型输出索引；
- 平铺验证图必须经过“图像编号 → ILSVRC ID → WNID → 模型索引”的官方映射；
- 禁止使用目录枚举位置、排序后每 50 张一类、旧扫描日志或模型预测反推真值。

官方 devkit SHA-256：`b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953`。

### 2.4 原生前向提取器

提取器通过 hook 观察 timm 的真实前向，不自行重演 Transformer Block。当前可提取：

- 12 层 Block 的输入与输出；
- 12 层的 Q、K、V；
- `attn_drop` 前后的 attention；
- 最终分类 logits。

Q/K/V 严格按 timm 0.9.16 的顺序拆分：

```text
[B, N, 3C] -> [B, N, 3, heads, head_dim] -> [3, B, heads, N, head_dim]
```

attention 按 timm 的实际运算顺序重算：

```text
softmax((Q * head_dim**-0.5) @ K.T)
```

对 ViT-B/16，缩放因子为 `64**-0.5 = 0.125`。

### 2.5 结果可追溯性

每次正式验证都会记录：

- Git commit 与工作区是否干净；
- 模型、图像和依赖锁文件的 SHA-256；
- 完整结构化预处理配置；
- Python、PyTorch、torchvision、timm 和 NumPy 版本；
- 设备、数据类型、随机种子和确定性设置；
- ground truth、top-1、top-5 与原始 logits 产物；
- logits、Block 和 attention 的全部验收误差。

## 3. 验证结果

正式端到端验证在 NVIDIA H20 上以 float32 运行，并开启 PyTorch 确定性算法。

| 验收项 | 实测结果 | 验收门槛 | 结论 |
|---|---:|---:|---|
| 标准 fused timm 前向 vs 提取器 logits | `8.58306884765625e-06` | `< 1e-5` | 通过 |
| unfused 普通前向 vs 加 hook 后 logits | `0.0` | 不得改变 | 通过 |
| 加 hook 前后 Block 输入最大误差 | `0.0` | 不得改变 | 通过 |
| 加 hook 前后 Block 输出最大误差 | `0.0` | 不得改变 | 通过 |
| 重算 attention vs timm 真实 attention | `0.0` | `< 1e-6` | 通过 |
| 完整 pickle vs state_dict logits | `0.0` | `< 1e-5` | 通过 |
| 完整 pickle vs state_dict Block 输入 | `0.0` | `< 1e-5` | 通过 |
| 完整 pickle vs state_dict Block 输出 | `0.0` | `< 1e-5` | 通过 |

端到端样本：

- 样本：`imagenet-scanner/n03976467/n03976467_9546.JPEG`；
- 真值：WNID `n03976467`，模型索引 `732`；
- top-1：`732`；
- top-5：`[732, 593, 710, 605, 662]`；
- top-1 与 top-5 均预测正确。

## 4. 自动测试结果

常规安全测试：

```text
13 passed, 1 deselected
```

显式开启的完整 pickle 兼容测试：

```text
1 passed, 13 deselected
```

依赖一致性检查：

```text
No broken requirements found.
```

## 5. 主要文件与产物

核心实现：

- `src/vitnns/modeling.py`：唯一模型加载和预处理；
- `src/vitnns/dataset.py`：数据清单与 Dataset；
- `src/vitnns/labels.py`：官方 ImageNet 标签映射；
- `src/vitnns/extractor.py`：真实前向 hook 提取器；
- `src/vitnns/provenance.py`：结果来源记录与校验；
- `src/vitnns/runner.py`：阶段 A 端到端验收入口。

验证产物：

- `artifacts/imagenet-scanner/manifest.csv`；
- `artifacts/imagenet-val-subset/manifest.csv`；
- `outputs/stage_a/extraction.pt`；
- `outputs/stage_a/provenance.json`；
- `outputs/stage_a/validation.json`。

大型数据、模型和运行产物均由 `.gitignore` 排除，不提交到 Git。正式 provenance 记录了当次验证所使用的代码提交与干净工作区状态；本文件随阶段 A 最终提交一并保存。

## 6. 阶段结论与下一步

阶段 A 已完成并达到全部验收门槛。当前已经具备可信的真实 ViT 前向、正确标签、Q/K/V/A 中间量、可复现实验配置与完整来源记录。

下一步进入阶段 B：以阶段 A 保存的真实中间张量为基础重写 NNS-ViT 归因传播，为 LayerNorm、GELU、残差连接、QK 乘法和 softmax 建立明确的传播规则与 conservation 测试；不再继续修补旧扫描器的手工前向。
