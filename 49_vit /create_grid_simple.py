"""
简单脚本：从已有的NNS结果创建包含原图的网格
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# 配置
scan_result_dir = r"D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final\0_n01443537_24scan_attn_map"
original_image_path = r"D:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG"
output_dir = r"D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final"
output_path = os.path.join(output_dir, "n01443537_24_attention_layers_grid.png")

print(f"扫描结果目录: {scan_result_dir}")
print(f"原图路径: {original_image_path}")
print(f"输出路径: {output_path}")

# 查找所有层目录
layer_dirs = []
if os.path.exists(scan_result_dir):
    for item in os.listdir(scan_result_dir):
        item_path = os.path.join(scan_result_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            layer_dirs.append((int(item), item_path))
    layer_dirs.sort()

print(f"\n找到 {len(layer_dirs)} 个层")
for layer_num, _ in layer_dirs:
    print(f"  Layer {layer_num:02d}")

if len(layer_dirs) == 0:
    print("错误：没有找到层目录")
    exit(1)

# 选择要显示的层（最多11层，因为原图占1格，共12格=2x6）
# 均匀选择层，显示从浅到深的过程
if len(layer_dirs) >= 11:
    # 均匀选择11层
    step = len(layer_dirs) / 11
    selected_indices = [int(i * step) for i in range(11)]
    selected_layers = [layer_dirs[i] for i in selected_indices]
else:
    selected_layers = layer_dirs

print(f"\n选择 {len(selected_layers)} 层用于显示")

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

# 其余位置：层的注意力图
positions = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
             (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

print("\n添加层注意力图...")
for idx, (layer_num, layer_path) in enumerate(selected_layers):
    if idx >= len(positions):
        break
    
    row, col = positions[idx]
    ax = fig.add_subplot(gs[row, col])
    
    # 优先查找类别引导的注意力图，如果没有则使用普通的
    possible_paths = [
        os.path.join(layer_path, 'input_attn', 'head0', 'avg_cls_attn_head0_class_guided.png'),
        os.path.join(layer_path, 'input_attn', 'head0', 'avg_cls_attn_head0.png'),
        os.path.join(layer_path, 'input_attn', 'avg_cls_attn_head0.png'),
        os.path.join(layer_path, 'avg_cls_attn_head0.png'),
        os.path.join(layer_path, 'input_attn', 'avg_cls_attn.png'),
    ]
    
    img_path = None
    for path in possible_paths:
        if os.path.exists(path):
            img_path = path
            break
    
    if img_path and os.path.exists(img_path):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f'Layer {layer_num:02d}', fontsize=13, fontweight='bold', pad=8)
        print(f"  ✓ Layer {layer_num:02d} 添加成功")
    else:
        ax.text(0.5, 0.5, f'Layer {layer_num}\nNot Found', ha='center', va='center', fontsize=10)
        ax.set_title(f'Layer {layer_num:02d}', fontsize=13, pad=8)
        print(f"  ⚠️  Layer {layer_num:02d} 图像未找到")
    ax.axis('off')

print(f"\n保存网格图像: {output_path}")
plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
plt.close()

print(f"\n✅ 完成！网格图像已保存: {output_path}")
print(f"   布局: 2行×6列")
print(f"   包含: 原图 + {len(selected_layers)} 层注意力图")

