"""
生成类似参考图片的cls_attn网格图
从注意力图中提取并重命名为02_cls_attn.png, 06_cls_attn.png等格式
使用统一的归一化使颜色一致
"""
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import shutil

# 配置
scan_attn_dir = r"D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final\0_n01443537_24scan_attn_map"
original_image_path = r"D:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG"
output_dir = r"D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final"
cls_attn_dir = os.path.join(output_dir, "cls_attn_images")  # 存储单个cls_attn图像
output_path = os.path.join(output_dir, "n01443537_24_cls_attn_grid.png")

print(f"注意力图目录: {scan_attn_dir}")
print(f"原图路径: {original_image_path}")
print(f"输出目录: {output_dir}")
print(f"cls_attn图像目录: {cls_attn_dir}")

# 创建cls_attn图像目录
os.makedirs(cls_attn_dir, exist_ok=True)

# 查找所有层目录
layer_dirs = []
if os.path.exists(scan_attn_dir):
    for item in os.listdir(scan_attn_dir):
        item_path = os.path.join(scan_attn_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            layer_dirs.append((int(item), item_path))
    layer_dirs.sort()

print(f"\n找到 {len(layer_dirs)} 个层")
for layer_num, _ in layer_dirs:
    print(f"  Layer {layer_num:02d}")

if len(layer_dirs) == 0:
    print("错误：没有找到层目录")
    exit(1)

# 选择要显示的层（均匀选择，显示从浅到深的过程）
# 参考图片显示：02, 06, 10, 14, 18, 22
selected_layers = []
if len(layer_dirs) >= 6:
    # 均匀选择6层（类似参考图片）
    step = len(layer_dirs) / 6
    selected_indices = [int(i * step) for i in range(6)]
    selected_layers = [layer_dirs[i] for i in selected_indices]
else:
    selected_layers = layer_dirs

print(f"\n选择 {len(selected_layers)} 层用于显示")
for layer_num, _ in selected_layers:
    print(f"  Layer {layer_num:02d}")

# 第一步：加载所有注意力图并收集所有像素值（用于统一归一化）
print("\n步骤1: 加载所有注意力图并计算全局范围...")
all_attention_images = {}
all_attention_arrays = {}
all_values = []

for layer_num, layer_path in selected_layers:
    # 查找avg_cls_attn_head0.png
    possible_paths = [
        os.path.join(layer_path, 'input_attn', 'head0', 'avg_cls_attn_head0.png'),
        os.path.join(layer_path, 'input_attn', 'avg_cls_attn_head0.png'),
        os.path.join(layer_path, 'avg_cls_attn_head0.png'),
    ]
    
    img_path = None
    for path in possible_paths:
        if os.path.exists(path):
            img_path = path
            break
    
    if img_path and os.path.exists(img_path):
        # 读取图像（RGB热力图）
        img = cv2.imread(img_path)
        if img is not None:
            # 转换为灰度（从热力图中提取原始值）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            all_attention_images[layer_num] = img
            all_attention_arrays[layer_num] = gray
            all_values.extend(gray.flatten())
            print(f"  ✓ Layer {layer_num:02d}: {img.shape}")
        else:
            print(f"  ✗ Layer {layer_num:02d}: 无法读取")
    else:
        print(f"  ✗ Layer {layer_num:02d}: 文件未找到")

if len(all_attention_images) == 0:
    print("错误：没有找到任何注意力图")
    exit(1)

# 计算全局最小值和最大值
global_min = np.min(all_values)
global_max = np.max(all_values)
global_range = global_max - global_min

print(f"\n步骤2: 计算全局范围...")
print(f"  全局最小值: {global_min}")
print(f"  全局最大值: {global_max}")
print(f"  全局范围: {global_range}")

# 第二步：使用全局范围重新归一化所有图像并保存为cls_attn格式
print(f"\n步骤3: 使用全局范围重新归一化并保存为cls_attn格式...")

for layer_num, layer_path in selected_layers:
    if layer_num in all_attention_arrays:
        # 获取当前图像的灰度值
        gray = all_attention_arrays[layer_num]
        
        # 归一化到[0, 1]使用全局范围
        if global_range > 1e-8:
            normalized = (gray.astype(np.float32) - global_min) / global_range
        else:
            normalized = gray.astype(np.float32) / 255.0
        
        # 确保在[0, 1]范围内
        normalized = np.clip(normalized, 0, 1)
        
        # 转换为uint8
        normalized_uint8 = (normalized * 255).astype(np.uint8)
        
        # 应用JET colormap（与原始可视化一致）
        heat_img = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
        
        # 保存为cls_attn格式（如02_cls_attn.png）
        cls_attn_path = os.path.join(cls_attn_dir, f"{layer_num:02d}_cls_attn.png")
        cv2.imwrite(cls_attn_path, heat_img)
        print(f"  ✓ Layer {layer_num:02d}: 已保存为 {layer_num:02d}_cls_attn.png")

# 第三步：创建网格图（原图 + 注意力图）
print(f"\n步骤4: 创建网格图...")

# 创建 2x6 网格（原图 + 6个注意力图，或更多）
num_attn = len(selected_layers)
num_cols = 6
num_rows = 2 if num_attn <= 5 else 3  # 如果超过5个，使用3行

fig = plt.figure(figsize=(24, 8 if num_rows == 2 else 12))
gs = GridSpec(num_rows, num_cols, figure=fig, hspace=0.35, wspace=0.25)

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

# 其余位置：注意力图
positions = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
             (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

if num_rows == 3:
    positions.extend([(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)])

print("\n添加注意力图...")
for idx, (layer_num, _) in enumerate(selected_layers):
    if idx >= len(positions):
        break
    
    row, col = positions[idx]
    ax = fig.add_subplot(gs[row, col])
    
    cls_attn_path = os.path.join(cls_attn_dir, f"{layer_num:02d}_cls_attn.png")
    
    if os.path.exists(cls_attn_path):
        img = Image.open(cls_attn_path)
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

print(f"\n✅ 完成！")
print(f"   cls_attn图像目录: {cls_attn_dir}")
print(f"   网格图像: {output_path}")
print(f"   布局: {num_rows}行×{num_cols}列")
print(f"   包含: 原图 + {len(selected_layers)} 个注意力图")
print(f"   所有注意力图使用统一的归一化范围，颜色一致！")

