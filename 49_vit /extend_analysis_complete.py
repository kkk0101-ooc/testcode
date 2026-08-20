"""
扩展20张图片的分析 - 补齐do_images的所有子目录和可视化
包括：layer_progression, hotspot_identification, cross_category等
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2
from scipy.ndimage import label, find_objects

print("="*70)
print(" 扩展分析 - 补齐所有do_images风格的可视化")
print("="*70)

# 配置
INPUT_DIR = Path("analysis_results_20images")
RAW_DATA_DIR = INPUT_DIR / "raw_data" / "attention"
OUTPUT_ATTENTION = INPUT_DIR / "visualize_attention"

# 创建缺失的子目录
missing_dirs = [
    "layer_progression",
    "hotspot_identification", 
    "cross_category"
]

for subdir in missing_dirs:
    (OUTPUT_ATTENTION / subdir).mkdir(parents=True, exist_ok=True)

# 获取所有图片ID
image_ids = sorted([d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
num_images = len(image_ids)
print(f"\n找到{num_images}张图片的数据")

# 加载所有数据
print("\n加载所有attention数据...")
all_data = {}
for img_id in tqdm(image_ids):
    img_data = []
    for layer_idx in range(12):
        attn_file = RAW_DATA_DIR / img_id / f"layer{layer_idx:02d}.npy"
        attn = np.load(attn_file)  # (1, 12, 197, 197)
        img_data.append(attn)
    all_data[img_id] = img_data

print(f"[OK] 已加载{len(all_data)}张图片的数据\n")

# ============================================================
# 1. Layer Progression - 展示每个head跨层的演化
# ============================================================
print("1. 生成Layer Progression（层级进展）...")

# 为每个head（选择3个代表性head）生成跨层演化
representative_heads = [0, 5, 11]

for head_idx in tqdm(representative_heads, desc="  生成head progression"):
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle(f'Head {head_idx} - Layer Progression (Average of {num_images} images)', 
                 fontsize=16, fontweight='bold')
    
    for layer_idx in range(12):
        ax = axes[layer_idx // 6, layer_idx % 6]
        
        # 收集该层该head的所有图片的attention
        all_attn = []
        for img_id in image_ids:
            attn = all_data[img_id][layer_idx]  # (1, 12, 197, 197)
            cls_attn = attn[0, head_idx, 0, 1:].reshape(14, 14)  # CLS对patches的attention
            all_attn.append(cls_attn)
        
        # 平均所有图片
        mean_attn = np.mean(all_attn, axis=0)
        
        # 可视化
        im = ax.imshow(mean_attn, cmap='hot', interpolation='bilinear')
        ax.set_title(f'Layer {layer_idx}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_ATTENTION / "layer_progression" / f"head{head_idx}_progression.png", 
                dpi=150, bbox_inches='tight')
    plt.close()

print("  [OK] Layer Progression完成\n")

# ============================================================
# 2. Hotspot Identification - 识别attention热点区域
# ============================================================
print("2. 生成Hotspot Identification（热点识别）...")

# 选择几个代表性层和head
representative_layers = [2, 6, 10]
representative_heads_hotspot = [0, 1, 2]

for layer_idx in tqdm(representative_layers, desc="  生成hotspot"):
    for head_idx in representative_heads_hotspot:
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        fig.suptitle(f'Layer {layer_idx} Head {head_idx} - Hotspot Identification', 
                     fontsize=16, fontweight='bold')
        
        # 显示20张图片的前20个
        for i, img_id in enumerate(image_ids[:20]):
            if i >= 20:
                break
            
            ax = axes[i // 5, i % 5]
            
            # 获取attention map
            attn = all_data[img_id][layer_idx]  # (1, 12, 197, 197)
            cls_attn = attn[0, head_idx, 0, 1:].reshape(14, 14)
            
            # 识别热点（高于平均值+1倍标准差）
            threshold = cls_attn.mean() + cls_attn.std()
            hotspots = cls_attn > threshold
            
            # 可视化
            im = ax.imshow(cls_attn, cmap='hot', interpolation='bilinear')
            
            # 叠加热点轮廓
            if hotspots.any():
                # 找到连通区域
                labeled_array, num_features = label(hotspots)
                for region_idx in range(1, num_features + 1):
                    region_mask = labeled_array == region_idx
                    # 绘制轮廓
                    contours = np.where(region_mask)
                    if len(contours[0]) > 0:
                        ax.plot(contours[1], contours[0], 'g-', linewidth=2, alpha=0.5)
            
            ax.set_title(f'{img_id[:10]}...', fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_ATTENTION / "hotspot_identification" / 
                   f"layer{layer_idx:02d}_head{head_idx}_hotspots.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()

print("  [OK] Hotspot Identification完成\n")

# ============================================================
# 3. Cross Category - 跨类别比较（简化版：跨图片比较）
# ============================================================
print("3. 生成Cross Category（跨图片比较）...")

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle(f'Cross-Image Comparison - Layer 11 Average Attention', 
             fontsize=16, fontweight='bold')

for i, img_id in enumerate(image_ids[:20]):
    ax = axes[i // 5, i % 5]
    
    # 最后一层的平均attention
    attn = all_data[img_id][11]  # (1, 12, 197, 197)
    cls_attn = attn[0, :, 0, 1:].mean(0).reshape(14, 14)  # 平均所有head
    
    im = ax.imshow(cls_attn, cmap='viridis', interpolation='bilinear')
    ax.set_title(f'{img_id[:15]}...', fontsize=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig(OUTPUT_ATTENTION / "cross_category" / "category_comparison.png", 
            dpi=150, bbox_inches='tight')
plt.close()

print("  [OK] Cross Category完成\n")

# ============================================================
# 4. 为每张图片生成单独的详细分析目录
# ============================================================
print("4. 为每张图片生成单独分析...")

# 只为第一张图片生成详细分析作为示例
sample_img_id = image_ids[0]
sample_output = OUTPUT_ATTENTION / f"results_{sample_img_id}"

for subdir in ["attention_evolution", "head_specialization", "layer_progression", 
               "hotspot_identification", "quantitative"]:
    (sample_output / subdir).mkdir(parents=True, exist_ok=True)

print(f"  为图片 {sample_img_id} 生成详细分析...")

# 4.1 该图片的attention演化
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle(f'{sample_img_id} - Attention Evolution Across Layers', fontsize=14)

for layer_idx in range(12):
    ax = axes[layer_idx // 4, layer_idx % 4]
    attn = all_data[sample_img_id][layer_idx]
    cls_attn = attn[0, :, 0, 1:].mean(0).reshape(14, 14)
    
    im = ax.imshow(cls_attn, cmap='hot', interpolation='bilinear')
    ax.set_title(f'Layer {layer_idx}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig(sample_output / "attention_evolution" / "layer_evolution.png", dpi=150)
plt.close()

# 4.2 该图片的head专业化
for layer_idx in [0, 4, 8, 11]:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'{sample_img_id} - Layer {layer_idx} All Heads', fontsize=14)
    
    attn = all_data[sample_img_id][layer_idx]
    
    for head_idx in range(12):
        ax = axes[head_idx // 4, head_idx % 4]
        cls_attn = attn[0, head_idx, 0, 1:].reshape(14, 14)
        
        im = ax.imshow(cls_attn, cmap='viridis', interpolation='bilinear')
        ax.set_title(f'Head {head_idx}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(sample_output / "head_specialization" / f"layer{layer_idx:02d}_heads.png", dpi=150)
    plt.close()

print(f"  [OK] 已为 {sample_img_id} 生成详细分析\n")

# ============================================================
# 5. 生成汇总报告
# ============================================================
print("5. 生成汇总统计...")

# 计算更多统计指标
stats_extended = {
    'hotspot_ratio': [],  # 热点区域占比
    'max_attention': [],   # 最大attention值
    'attention_std': []    # attention标准差
}

for layer_idx in range(12):
    layer_hotspot = []
    layer_max = []
    layer_std = []
    
    for img_id in image_ids:
        attn = all_data[img_id][layer_idx]
        cls_attn = attn[0, :, 0, 1:].mean(0)  # 平均所有head
        
        # 热点占比
        threshold = cls_attn.mean() + cls_attn.std()
        hotspot_ratio = (cls_attn > threshold).sum() / len(cls_attn)
        layer_hotspot.append(hotspot_ratio)
        
        # 最大值
        layer_max.append(cls_attn.max())
        
        # 标准差
        layer_std.append(cls_attn.std())
    
    stats_extended['hotspot_ratio'].append(np.mean(layer_hotspot))
    stats_extended['max_attention'].append(np.mean(layer_max))
    stats_extended['attention_std'].append(np.mean(layer_std))

# 可视化扩展统计
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(stats_extended['hotspot_ratio'], 'o-', color='purple', linewidth=2)
axes[0].set_title('Hotspot Ratio Evolution', fontsize=14)
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Hotspot Ratio')
axes[0].grid(True, alpha=0.3)

axes[1].plot(stats_extended['max_attention'], 'o-', color='red', linewidth=2)
axes[1].set_title('Max Attention Evolution', fontsize=14)
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Max Attention Value')
axes[1].grid(True, alpha=0.3)

axes[2].plot(stats_extended['attention_std'], 'o-', color='blue', linewidth=2)
axes[2].set_title('Attention Std Evolution', fontsize=14)
axes[2].set_xlabel('Layer')
axes[2].set_ylabel('Standard Deviation')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_ATTENTION / "quantitative" / "extended_statistics.png", dpi=150)
plt.close()

print("  [OK] 扩展统计完成\n")

print("="*70)
print(" 扩展分析完成！")
print("="*70)
print(f"\n新增内容：")
print(f"  - layer_progression/: {len(representative_heads)}个PNG")
print(f"  - hotspot_identification/: {len(representative_layers)*len(representative_heads_hotspot)}个PNG")
print(f"  - cross_category/: 1个PNG")
print(f"  - results_{sample_img_id}/: 详细单图分析")
print(f"  - quantitative/extended_statistics.png")
print(f"\n请查看: {INPUT_DIR}")
