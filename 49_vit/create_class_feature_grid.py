"""
直接使用类别特征图（0-9.png）创建网格可视化
这些图能看到鱼的形状，效果比注意力图更好
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# 配置
class_feature_dir = r"D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final\0_n01443537_24scan_RGB\50"
original_image_path = r"D:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG"
output_dir = r"D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final"
output_path = os.path.join(output_dir, "n01443537_24_class_features_grid.png")

print(f"类别特征图目录: {class_feature_dir}")
print(f"原图路径: {original_image_path}")
print(f"输出路径: {output_path}")

# 查找所有类别特征图（0.png到9.png）
class_images = []
if os.path.exists(class_feature_dir):
    for i in range(10):  # 0-9
        img_path = os.path.join(class_feature_dir, f"{i}.png")
        if os.path.exists(img_path):
            class_images.append((i, img_path))
    
    print(f"\n找到 {len(class_images)} 个类别特征图")
    for class_idx, _ in class_images:
        print(f"  类别 {class_idx}")
else:
    print(f"错误：类别特征图目录不存在: {class_feature_dir}")
    exit(1)

if len(class_images) == 0:
    print("错误：没有找到类别特征图")
    exit(1)

# 选择要显示的类别（最多11个，因为原图占1格，共12格=2x6）
# 优先显示预测类别（类别0，鱼）和其他重要类别
selected_classes = class_images[:11] if len(class_images) >= 11 else class_images

print(f"\n选择 {len(selected_classes)} 个类别用于显示")

# 创建 2x6 网格
fig = plt.figure(figsize=(24, 8))
gs = GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.25)

# 第一个位置：原图
print("\n添加原图...")
ax = fig.add_subplot(gs[0, 0])
if os.path.exists(original_image_path):
    original_img = Image.open(original_image_path).convert('RGB')
    ax.imshow(original_img)
    ax.set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
    print("  ✓ 原图添加成功")
else:
    ax.text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center', fontsize=12)
    ax.set_title('Original Image', fontsize=14, pad=10)
    print("  ⚠️  原图未找到")
ax.axis('off')

# 其余位置：类别特征图
positions = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
             (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

print("\n添加类别特征图...")
for idx, (class_idx, img_path) in enumerate(selected_classes):
    if idx >= len(positions):
        break
    
    row, col = positions[idx]
    ax = fig.add_subplot(gs[row, col])
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax.imshow(img)
        # 如果是类别0（鱼），加粗标题
        if class_idx == 0:
            ax.set_title(f'Class {class_idx} (Fish)', fontsize=13, fontweight='bold', pad=8, color='red')
        else:
            ax.set_title(f'Class {class_idx}', fontsize=13, fontweight='bold', pad=8)
        print(f"  ✓ Class {class_idx} 添加成功")
    else:
        ax.text(0.5, 0.5, f'Class {class_idx}\nNot Found', ha='center', va='center', fontsize=10)
        ax.set_title(f'Class {class_idx}', fontsize=13, pad=8)
        print(f"  ⚠️  Class {class_idx} 图像未找到")
    ax.axis('off')

print(f"\n保存网格图像: {output_path}")
plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
plt.close()

print(f"\n✅ 完成！网格图像已保存: {output_path}")
print(f"   布局: 2行×6列")
print(f"   包含: 原图 + {len(selected_classes)} 个类别特征图")
print(f"   类别0（鱼）已高亮显示")

