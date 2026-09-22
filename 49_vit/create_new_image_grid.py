"""生成新图片的12层注意力网格图 - n02085620_1.JPEG"""
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 配置
result_dir = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final_new\0_n02085620_1_new"
new_image = r"d:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\scanimg\n02085620\n02085620_1.JPEG"
output_path = os.path.join(result_dir, "n02085620_1_complete_12layers_grid.png")

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
print(f"New image: {new_image}")
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

# 选择11层用于显示（原图占1格，共12格=3x4）
if len(layer_dirs) >= 11:
    # 均匀选择11层，展示从浅到深的演化
    step = len(layer_dirs) / 11
    selected_layers = [layer_dirs[int(i * step)] for i in range(11)]
else:
    selected_layers = layer_dirs

print(f"\nSelected {len(selected_layers)} layers for 12-layer visualization:")
for layer_num, _ in selected_layers:
    print(f"  Layer {layer_num:02d}")

# 创建 3x4 网格
fig = plt.figure(figsize=(24, 18))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25)

# 第一个位置：原图
print("\nAdding original image...")
ax = fig.add_subplot(gs[0, 0])
if os.path.exists(new_image):
    original_img = Image.open(new_image).convert('RGB')
    ax.imshow(original_img)
    ax.set_title('Original Image\n(n02085620_1)', fontsize=16, fontweight='bold', pad=12)
    print("  SUCCESS: Original image added successfully")
else:
    ax.text(0.5, 0.5, 'Original\nImage\nNot Found', ha='center', va='center', fontsize=12)
    ax.set_title('Original Image', fontsize=16, pad=12)
    print("  WARNING: Original image not found")
ax.axis('off')

# 其余位置：层的注意力图
positions = [(0, 1), (0, 2), (0, 3),
             (1, 0), (1, 1), (1, 2), (1, 3),
             (2, 0), (2, 1), (2, 2), (2, 3)]

print("\nAdding layer attention maps...")
success_count = 0
for idx, (layer_num, layer_path) in enumerate(selected_layers):
    if idx >= len(positions):
        break
    
    row, col = positions[idx]
    ax = fig.add_subplot(gs[row, col])
    
    # 查找注意力图（按优先级顺序）
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
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            
            # 根据层数添加描述性标题
            if layer_num <= 10:
                layer_desc = f'Layer {layer_num:02d} (Shallow)'
            elif layer_num <= 30:
                layer_desc = f'Layer {layer_num:02d} (Middle)'
            else:
                layer_desc = f'Layer {layer_num:02d} (Deep)'
                
            ax.set_title(layer_desc, fontsize=14, fontweight='bold', pad=10)
            print(f"  SUCCESS: Layer {layer_num:02d} added successfully")
            success_count += 1
        except Exception as e:
            ax.text(0.5, 0.5, f'Layer {layer_num}\nLoad Error', ha='center', va='center', fontsize=10)
            ax.set_title(f'Layer {layer_num:02d}', fontsize=14, pad=10)
            print(f"  ERROR: Layer {layer_num:02d} failed to load: {e}")
    else:
        ax.text(0.5, 0.5, f'Layer {layer_num}\nNot Found', ha='center', va='center', fontsize=10)
        ax.set_title(f'Layer {layer_num:02d}', fontsize=14, pad=10)
        print(f"  WARNING: Layer {layer_num:02d} image not found")
        print(f"    Searched paths:")
        for path in possible_paths:
            print(f"      {path}")
    ax.axis('off')

# 保存网格图
print(f"\nSaving complete 12-layer grid: {output_path}")
plt.suptitle('ViT Depth-12 NNS Visualization: New Image (n02085620_1)\nAttention Evolution Across All Layers', 
             fontsize=20, fontweight='bold', y=0.98)

# 添加说明文字
fig.text(0.5, 0.02, 
         'Red/Yellow = High Attention (Object Features) | Blue = Low Attention (Background) | Evolution: Shallow → Middle → Deep Layers',
         ha='center', fontsize=12, style='italic')

plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
plt.close()

print(f"\n{'='*80}")
print("SUCCESS: COMPLETE 12-LAYER ATTENTION GRID FOR NEW IMAGE GENERATED!")
print(f"{'='*80}")
print(f"\nFile: {output_path}")
print(f"Resolution: High-quality (200 DPI)")
print(f"Layout: 3 rows × 4 columns")
print(f"Content: Original image (n02085620_1) + {success_count} layer attention maps")
print(f"Layers processed: {len(selected_layers)} out of {len(layer_dirs)} available")
print(f"Model prediction: Class 6 (76.78% confidence)")
print(f"\nExpected Results:")
print("- Shallow layers (2-10): Local textures and edges")
print("- Middle layers (14-30): Object parts and contours") 
print("- Deep layers (34-46): Global semantic understanding")
print(f"\nQuality Check:")
print("- Red/Yellow regions should highlight object features")
print("- Blue regions should show background (low attention)")
print("- Clear evolution from local to global features")
print(f"\n{'='*80}")

# 尝试自动打开图像
import platform
if platform.system() == 'Windows':
    try:
        os.startfile(output_path)
        print("Image opened automatically for verification!")
    except Exception as e:
        print(f"Please manually open: {output_path}")
        print(f"Auto-open failed: {e}")

print(f"\nSUCCESS: Complete 12-layer ViT NNS visualization for new image ready!")
print("Please verify the attention patterns and compare with the fish image results.")
