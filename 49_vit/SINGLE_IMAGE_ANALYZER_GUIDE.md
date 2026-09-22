# 单张图片分析工具 - 使用指南

## 概述

提供两个脚本用于分析单张或多张图片：
1. **generate_single_image_analysis.py** - 命令行工具，支持参数配置
2. **quick_analyze.py** - 快速配置脚本，直接修改代码运行

## 快速开始

### 方法1: 使用quick_analyze.py（推荐）

**步骤1**: 编辑`quick_analyze.py`，修改配置：

```python
# 配置区域
IMAGE_PATH = "datas/ILSVRC2012_val/ILSVRC2012_val_00028758.JPEG"
OUTPUT_DIR = "do_imagenet_vit_28758"
IMAGE_NAME = "ILSVRC2012_val_00028758"
```

**步骤2**: 运行脚本：

```bash
python quick_analyze.py
```

就这么简单！

### 方法2: 使用命令行

```bash
python generate_single_image_analysis.py \
    --image "datas/ILSVRC2012_val/ILSVRC2012_val_00028758.JPEG" \
    --output "do_imagenet_vit_28758" \
    --name "ILSVRC2012_val_00028758"
```

### 方法3: 批量分析多张图片

**步骤1**: 编辑`quick_analyze.py`：

```python
# 启用批量模式
BATCH_MODE = True

# 配置要分析的图片
BATCH_IMAGES = [
    ("datas/ILSVRC2012_val/ILSVRC2012_val_00000441.JPEG", "do_imagenet_vit_441"),
    ("datas/ILSVRC2012_val/ILSVRC2012_val_00006100.JPEG", "do_imagenet_vit_6100"),
    ("datas/ILSVRC2012_val/ILSVRC2012_val_00028758.JPEG", "do_imagenet_vit_28758"),
]
```

**步骤2**: 运行：

```bash
python quick_analyze.py
```

## 输出内容

### 目录结构

为每张图片生成完整的分析目录：

```
do_imagenet_vit_XXXXX/
├── visualize_attention/           # Attention分析
│   ├── analysis_report.html       # HTML报告 ★
│   ├── layer_progression/         # 3个PNG
│   ├── head_specialization/       # 12个PNG
│   ├── hotspot_identification/    # 18个PNG
│   ├── attention_evolution/       # 1个PNG
│   ├── spatial_distribution/      # 3个PNG
│   └── quantitative/              # 1个PNG + JSON
│
└── visualize_v/                   # V向量分析
    ├── analysis_report.html       # HTML报告 ★
    ├── layer_progression/         # 3个PNG
    └── value_evolution/           # 1个PNG
```

### 可视化内容

#### visualize_attention (38个PNG)

1. **Layer Progression (3个PNG)**
   - head0_progression.png
   - head5_progression.png
   - head11_progression.png
   - 每张图：2×6网格，展示该head在12层中的演化

2. **Head Specialization (12个PNG)**
   - 6个代表层 × 2类型（all heads + similarity）
   - layerXX_heads.png: 3×4网格展示12个head
   - layerXX_similarity.png: 相似度矩阵

3. **Hotspot Identification (18个PNG)**
   - 6个代表层 × 3个head
   - 绿色轮廓标记热点区域

4. **Attention Evolution (1个PNG)**
   - 3×4网格展示12层的演化

5. **Spatial Distribution (3个PNG)**
   - Layer 0, 6, 11的空间分布热力图

6. **Quantitative (1个PNG + JSON)**
   - 统计指标曲线图
   - statistics.json数据文件

#### visualize_v (4个PNG)

1. **Layer Progression (3个PNG)**
   - V向量强度在12层中的演化

2. **Value Evolution (1个PNG)**
   - V向量整体演化

## 命令行参数说明

### generate_single_image_analysis.py

```bash
python generate_single_image_analysis.py [参数]
```

**必需参数**:
- `--image` : 图片路径
- `--output` : 输出目录名

**可选参数**:
- `--name` : 图片显示名称（用于HTML标题）

**示例**:

```bash
# 基础用法
python generate_single_image_analysis.py \
    --image "datas/ILSVRC2012_val/ILSVRC2012_val_00028758.JPEG" \
    --output "my_analysis"

# 指定显示名称
python generate_single_image_analysis.py \
    --image "datas/ILSVRC2012_val/ILSVRC2012_val_00028758.JPEG" \
    --output "do_imagenet_vit_28758" \
    --name "Golden Retriever Image"

# 使用相对路径
python generate_single_image_analysis.py \
    --image "../test_images/cat.jpg" \
    --output "cat_analysis"
```

## quick_analyze.py 配置详解

### 单张图片模式

```python
# 配置区域
IMAGE_PATH = "datas/ILSVRC2012_val/ILSVRC2012_val_00028758.JPEG"  # 图片路径
OUTPUT_DIR = "do_imagenet_vit_28758"                               # 输出目录
IMAGE_NAME = "ILSVRC2012_val_00028758"                            # 显示名称（可选）

# 模式设置
BATCH_MODE = False  # 单张模式
```

### 批量模式

```python
# 模式设置
BATCH_MODE = True  # 启用批量模式

# 批量图片列表
BATCH_IMAGES = [
    # (图片路径, 输出目录)
    ("datas/ILSVRC2012_val/ILSVRC2012_val_00000441.JPEG", "do_imagenet_vit_441"),
    ("datas/ILSVRC2012_val/ILSVRC2012_val_00006100.JPEG", "do_imagenet_vit_6100"),
    ("datas/ILSVRC2012_val/ILSVRC2012_val_00028758.JPEG", "do_imagenet_vit_28758"),
    # 添加更多...
]
```

## 已生成的示例

### 当前已生成

✓ **do_imagenet_vit_28758** - ILSVRC2012_val_00028758.JPEG的完整分析

查看报告：
```bash
start do_imagenet_vit_28758\visualize_attention\analysis_report.html
start do_imagenet_vit_28758\visualize_v\analysis_report.html
```

## 性能参考

| 项目 | 时间 |
|------|------|
| 单张图片完整分析 | ~15秒 |
| - 数据提取 | ~1秒 |
| - visualize_attention | ~10秒 |
| - visualize_v | ~2秒 |
| - HTML生成 | <1秒 |

## 常见问题

### Q: 可以分析任何图片吗？

A: 可以！支持常见图片格式（JPEG, PNG等），会自动resize到224×224。

### Q: 输出目录已存在怎么办？

A: 会覆盖已有内容。建议每次使用不同的输出目录名。

### Q: 如何修改分析的层数？

A: 编辑`generate_single_image_analysis.py`中的`representative_layers`变量：

```python
representative_layers = [0, 2, 4, 6, 8, 10]  # 修改为你想要的层
```

### Q: 如何只生成attention分析，不生成V向量？

A: 注释掉脚本中的"PART 2: visualize_v"部分。

### Q: 可以保存原始数据吗？

A: 可以！在脚本中添加：

```python
# 在提取数据后添加
import numpy as np
for i, (attn, v) in enumerate(zip(attention_maps, v_vectors)):
    np.save(f"raw_attention_layer{i}.npy", attn.numpy())
    np.save(f"raw_v_layer{i}.npy", v.numpy())
```

### Q: 批量分析失败了部分图片怎么办？

A: 脚本会继续处理其他图片，最后显示成功/失败的数量。检查失败的图片路径是否正确。

## 高级用法

### 自定义分析

你可以修改`generate_single_image_analysis.py`来添加自定义分析：

```python
# 在PART 1后添加自定义分析
# 例如：分析特定head的注意力分布
special_head_idx = 3
for layer_idx in range(num_layers):
    attn = attention_maps[layer_idx]
    special_attn = attn[0, special_head_idx, 0, 1:].numpy()
    # 自定义处理...
```

### 与其他工具集成

```python
# 作为Python模块使用
from generate_single_image_analysis import generate_analysis

# 在你的脚本中调用
for img_path in my_image_list:
    output_dir = f"analysis_{Path(img_path).stem}"
    generate_analysis(img_path, output_dir)
```

### 自动化Pipeline

```bash
# 创建一个批处理脚本
for img in datas/ILSVRC2012_val/ILSVRC2012_val_000*.JPEG; do
    python generate_single_image_analysis.py \
        --image "$img" \
        --output "analysis_$(basename $img .JPEG)"
done
```

## 对比：单图分析 vs 多图分析

| 特性 | generate_single_image_analysis | generate_do_imagenet_vit |
|------|-------------------------------|--------------------------|
| 图片数量 | 1张 | 20张 |
| 可视化方法 | 单图的详细分析 | 多图统计和对比 |
| 输出大小 | ~40个PNG | ~44个PNG |
| 运行时间 | ~15秒 | ~40秒 |
| 适用场景 | 深入研究单张图片 | 跨图片的统计分析 |

## 扩展建议

### 短期
- [ ] 添加更多统计指标
- [ ] 支持自定义配色方案
- [ ] 添加原始数据导出选项

### 中期
- [ ] 支持视频输入（逐帧分析）
- [ ] 交互式HTML报告
- [ ] 对比分析功能（多张图片对比）

### 长期
- [ ] Web界面
- [ ] 实时分析
- [ ] GPU批处理优化

## 相关文件

- `generate_single_image_analysis.py` - 核心分析脚本
- `quick_analyze.py` - 快速配置脚本
- `load_local_model.py` - 模型加载器
- `generate_do_imagenet_vit.py` - 多图分析脚本（参考）

## 技术细节

### Attention提取

```python
# 手动前向传播，逐层提取attention
for block in model.blocks:
    qkv = block.attn.qkv(x_norm)
    q, k, v = split_qkv(qkv)
    attn = softmax(q @ k.T / sqrt(d_k))
```

### 热点识别

```python
# 阈值 = 均值 + 标准差
threshold = attention.mean() + attention.std()
hotspots = attention > threshold

# OpenCV提取轮廓
contours = cv2.findContours(hotspots, cv2.RETR_EXTERNAL, ...)
```

### V向量强度

```python
# 使用L2 norm作为强度指标
v_intensity = v[batch, head, patches, :].norm(dim=-1)
```

## 示例输出

查看已生成的示例：

```bash
# 浏览器打开
start do_imagenet_vit_28758\visualize_attention\analysis_report.html

# 或在文件管理器中打开
explorer do_imagenet_vit_28758
```

## 总结

✓ **两个脚本，灵活使用**  
✓ **支持单张和批量分析**  
✓ **完整的HTML报告**  
✓ **38个PNG可视化（attention）+ 4个PNG（V向量）**  
✓ **15秒快速生成**

---

**版本**: v1.0  
**更新**: 2024-11-22  
**状态**: 生产就绪
