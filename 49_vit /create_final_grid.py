"""生成最终的12层注意力网格图"""
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 配置
result_dir = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final\0_n01443537_24"
fish_image = r"d:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG"
output_path = os.path.join(result_dir, "fish_12layers_final_grid.png")

# 查找scan_attn_map目录
scan_attn_dir = None
for item in os.listdir(result_dir):
    if "scan_attn_map" in item:
        scan_attn_dir = os.path.join(result_dir, item)
        break

if not scan_attn_dir:
    print("ERROR: scan_attn_map directory not found")
    exit(1)

print(f"Scan directory: {scan_attn_dir}")
print(f"Fish image: {fish_image}")
print(f"Output: {output_path}")

# 查找所有层目录
layer_dirs = []
for item in os.listdir(scan_attn_dir):
    item_path = os.path.join(scan_attn_dir, item)
    if os.path.isdir(item_path) and item.isdigit():
        layer_dirs.append((int(item), item_path))
layer_dirs.sort()

print(f"\nFound {len(layer_dirs)} layers:")
for layer_num, _ in layer_dirs:
    print(f"  Layer {layer_num:02d}")

if not layer_dirs:
    print("ERROR: No layer directories found")
    exit(1)

# 选择11层用于显示（原图占1格，共12格）
if len(layer_dirs) >= 11:
    # 均匀选择11层
    step = len(layer_dirs) / 11
    selected_layers = [layer_dirs[int(i * step)] for i in range(11)]
else:
    selected_layers = layer_dirs

print(f"\nSelected {len(selected_layers)} layers for visualization:")
for layer_num, _ in selected_layers:
    print(f"  Layer {layer_num:02d}")

# 创建 3x4 网格
fig = plt.figure(figsize=(20, 15))
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)

# 第一个位置：原图
print("\nAdding original image...")
ax = fig.add_subplot(gs[0, 0])
if os.path.exists(fish_image):
    original_img = Image.open(fish_image).convert('RGB')
    ax.imshow(original_img)
    ax.set_title('Original Fish Image', fontsize=16, fontweight='bold', pad=10)
    print("  SUCCESS: Original image added")
else:
    ax.text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center', fontsize=12)
    ax.set_title('Original Image', fontsize=16, pad=10)
    print("  WARNING: Original image not found")
ax.axis('off')

# 其余位置：层的注意力图
positions = [(0, 1), (0, 2), (0, 3),
             (1, 0), (1, 1), (1, 2), (1, 3),
             (2, 0), (2, 1), (2, 2), (2, 3)]

print("\nAdding layer attention maps...")
for idx, (layer_num, layer_path) in enumerate(selected_layers):
    if idx >= len(positions):
        break
    
    row, col = positions[idx]
    ax = fig.add_subplot(gs[row, col])
    
    # 查找注意力图（优先级顺序）
    possible_paths = [
        os.path.join(layer_path, 'input_attn', 'head0', 'avg_cls_attn_head0.png'),
        os.path.join(layer_path, 'input_attn', 'head0', 'avg_cls_attn_head0_class_guided.png'),
        os.path.join(layer_path, 'input_attn', 'avg_cls_attn_head0.png'),
        os.path.join(layer_path, 'avg_cls_attn_head0.png'),
    ]
    
    img_path = None
    for path in possible_paths:
        if os.path.exists(path):
            img_path = path
            break
    
    if img_path:
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f'Layer {layer_num:02d} Attention', fontsize=14, fontweight='bold', pad=8)
        print(f"  SUCCESS: Layer {layer_num:02d} added")
    else:
        ax.text(0.5, 0.5, f'Layer {layer_num}\nNot Found', ha='center', va='center', fontsize=10)
        ax.set_title(f'Layer {layer_num:02d}', fontsize=14, pad=8)
        print(f"  WARNING: Layer {layer_num:02d} image not found")
        print(f"    Searched paths:")
        for path in possible_paths:
            print(f"      {path}")
    ax.axis('off')

# 保存网格图
print(f"\nSaving grid image: {output_path}")
plt.suptitle('ViT Depth-12 NNS Visualization: Fish Image Attention Evolution', 
             fontsize=18, fontweight='bold', y=0.98)
plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
plt.close()

print(f"\n{'='*80}")
print("SUCCESS: 12-Layer Attention Grid Generated!")
print(f"{'='*80}")
print(f"\nFile: {output_path}")
print(f"Layout: 3 rows x 4 columns")
print(f"Content: Original image + {len(selected_layers)} layer attention maps")
print(f"\nPlease open the image to verify:")
print("- Red/Yellow regions = Fish contours (high attention)")
print("- Blue regions = Background (low attention)")
print("- Evolution from shallow to deep layers")
print(f"\n{'='*80}")

# 尝试打开图像
import platform
if platform.system() == 'Windows':
    try:
        os.startfile(output_path)
        print("Image opened automatically!")
    except:
        print("Please manually open the image file.")
