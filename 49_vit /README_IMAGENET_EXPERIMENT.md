# ImageNet + ViT-base-16 完整实验框架

## 📋 项目概述

将Tiny-ImageNet上的ViT NNS可视化工作迁移到完整ImageNet数据集，并进行深入的量化分析和对比研究。


## 📊 量化指标体系

### 核心指标

#### 1. **Entropy (熵)**
- **作用**: 衡量注意力分散程度
- **公式**: `H = -Σ p(i) * log2(p(i))`
- **解释**: 
  - 低熵 → 注意力集中
  - 高熵 → 注意力分散
- **用途**: 判断哪个层/头更专注，不凭肉眼感受

#### 2. **IoU (Intersection over Union)**
- **作用**: 评估注意力准确性
- **公式**: `IoU = |A ∩ B| / |A ∪ B|`
- **解释**: 注意力区域与目标区域的重叠度
- **用途**: 判断哪个方法更"集中"且更准确

#### 3. **Peak Activation (最大激活值)**
- **作用**: 衡量关键区域突出程度
- **公式**: `Peak = max(attention_map)`
- **解释**:
  - 高峰值 → 强烈关注某区域
  - 低峰值 → 注意力平均分布
- **用途**: 看哪个方法更突出关键区域

#### 4. **Concentration Index (集中度指数)**
- **作用**: 量化注意力聚焦程度
- **公式**: `CI = Σ(top_10%_values) / Σ(all_values)`
- **解释**: Top-10%激活值占总激活的比例
- **用途**: 客观评估注意力集中度

### 辅助指标

5. **Layer Similarity**: 层间相似度（余弦相似度）
6. **Head Diversity**: 多头注意力差异度
7. **Effective Resolution**: 有效分辨率

## 🚀 快速开始

### 方式1: 运行完整实验流程

```bash
# 使用现有结果（跳过处理步骤）
python run_complete_imagenet_experiment.py --skip_processing

# 或者重新处理200张图片
python run_complete_imagenet_experiment.py --image_count 200
```

### 方式2: 分步运行

#### 步骤1: 扩展ImageNet处理（可选）
```bash
python train_scanner_cls.py --device 0 --count 200 --dataset ImageNet \
    --classes 1000 \
    --imagenet_path "D:\project\vit_114\NNS_model_and_dataset\dataset\imagenet_full\val" \
    --process_wrong_classification True \
    --nns_resize 64
```

#### 步骤2: 生成网格图
```bash
python combine_layer_attention.py \
    --base_dir "scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained" \
    --grid_size 3 4
```

#### 步骤3: 量化分析
```bash
python analyze_imagenet_quantitative.py \
    --base_dir "scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained" \
    --output_dir "analysis_results/imagenet_vitbase16"
```

#### 步骤4: 对比分析
```bash
python compare_tiny_vs_imagenet.py \
    --tiny_metrics "do_images/visualize_attention/results_n01443537_24/quantitative/all_metrics.json" \
    --imagenet_metrics "analysis_results/imagenet_vitbase16/metrics/all_metrics.json" \
    --output_dir "analysis_results/comparison"
```

## 📁 文件结构

```
vitnns/49_vit/
├── IMAGENET_VITBASE16_EXPERIMENT_PLAN.md  # 详细实验计划
├── README_IMAGENET_EXPERIMENT.md          # 本文件
├── run_complete_imagenet_experiment.py    # 主控脚本
├── analyze_imagenet_quantitative.py       # 量化分析脚本
├── compare_tiny_vs_imagenet.py            # 对比分析脚本
│
├── scan_results/                          # NNS扫描结果
│   └── scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained/
│       ├── 0_ILSVRC2012_val_00000001_wrong_pred65/
│       │   ├── _layer_attention_grid.png  # 12层演化网格图
│       │   ├── scan_attn_map/             # 注意力图 (4896张)
│       │   ├── scan_v/                    # V向量可视化 (3888张)
│       │   └── scan_RGB/                  # RGB特征 (1000张)
│       ├── 1_ILSVRC2012_val_00000002_wrong_pred795/
│       └── ... (共63张图片)
│
├── analysis_results/                      # 分析结果
│   ├── imagenet_vitbase16/
│   │   ├── metrics/
│   │   │   └── all_metrics.json          # 所有量化指标
│   │   ├── statistics/
│   │   │   └── summary_statistics.json   # 统计摘要
│   │   └── visualizations/
│   │       ├── metrics_distribution.png  # 指标分布图
│   │       └── layer_evolution.png       # 层级演化图
│   └── comparison/
│       ├── overall_comparison.json       # 整体对比
│       ├── comparison_report.md          # 对比报告
│       ├── metrics_comparison_boxplot.png
│       └── layer_evolution_comparison.png
│
└── do_images/                             # Tiny-ImageNet分析
    ├── visualize_attention/
    └── visualize_v/
```

## 📈 当前进展

### ✅ 已完成
1. ImageNet数据集下载和组织（50,000张）
2. ViT-base-16模型加载（timm预训练）
3. 基础NNS扫描（63张图片，525,249张可视化）
4. 层注意力网格图生成（55张）
5. 量化分析框架实现
6. 对比分析框架实现




## 🔍 关键发现（预期）

### Tiny-ImageNet vs ImageNet 对比

| 维度 | Tiny-ImageNet | ImageNet | 预期差异 |
|------|---------------|----------|---------|
| **数据规模** | 200类, 10K图 | 1000类, 50K图 | 5倍类别，5倍数据 |
| **分辨率** | 64×64 | 224×224 | 12倍像素 |
| **模型** | 自训练ViT | ViT-base-16预训练 | 更强特征提取 |




## 💡 使用建议

### 对于快速验证
```bash
# 使用现有63张图片的结果
python run_complete_imagenet_experiment.py --skip_processing
```

### 对于完整实验
```bash
# 处理200张图片并运行完整分析
python run_complete_imagenet_experiment.py --image_count 200
```

### 对于单独步骤
```bash
# 只运行量化分析
python run_complete_imagenet_experiment.py --step 3

# 只运行对比分析
python run_complete_imagenet_experiment.py --step 4
```

## 🔧 技术细节

### 模型信息
- **名称**: vit_base_patch16_224.augreg2_in21k_ft_in1k
- **预训练**: ImageNet-21k (1400万图片，21000类)
- **微调**: ImageNet-1k (128万图片，1000类)
- **架构**: 12层Transformer，12个注意力头，768维嵌入
- **来源**: timm库（HuggingFace）

