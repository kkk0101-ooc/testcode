"""使用现有结果创建基础网格图"""
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 配置
result_dir = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final\0_n01443537_24"
fish_image = r"d:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG"
output_path = os.path.join(result_dir, "fish_basic_result.png")

print(f"Result directory: {result_dir}")
print(f"Fish image: {fish_image}")
print(f"Output: {output_path}")

# 检查tem0目录
tem0_dir = os.path.join(result_dir, "tem0")
if not os.path.exists(tem0_dir):
    print("ERROR: tem0 directory not found")
    exit(1)

# 查找可用的图像文件
available_images = []
for item in os.listdir(tem0_dir):
    if item.endswith('.png'):
        available_images.append(os.path.join(tem0_dir, item))

print(f"\nFound {len(available_images)} result images:")
for img in available_images:
    print(f"  {os.path.basename(img)}")

# 创建 2x2 网格（简单版本）
fig = plt.figure(figsize=(12, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# 位置1：原图
ax = fig.add_subplot(gs[0, 0])
if os.path.exists(fish_image):
    original_img = Image.open(fish_image).convert('RGB')
    ax.imshow(original_img)
    ax.set_title('Original Fish Image', fontsize=14, fontweight='bold')
    print("\nSUCCESS: Original image added")
else:
    ax.text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center', fontsize=12)
    ax.set_title('Original Image', fontsize=14)
    print("\nWARNING: Original image not found")
ax.axis('off')

# 位置2-4：结果图像
positions = [(0, 1), (1, 0), (1, 1)]
for idx, img_path in enumerate(available_images[:3]):  # 最多3个结果图
    if idx >= len(positions):
        break
    
    row, col = positions[idx]
    ax = fig.add_subplot(gs[row, col])
    
    try:
        img = Image.open(img_path)
        ax.imshow(img)
        img_name = os.path.basename(img_path)
        ax.set_title(f'NNS Result: {img_name}', fontsize=12, fontweight='bold')
        print(f"SUCCESS: Added {img_name}")
    except Exception as e:
        ax.text(0.5, 0.5, f'Error\nLoading\n{os.path.basename(img_path)}', 
                ha='center', va='center', fontsize=10)
        ax.set_title(f'Error: {os.path.basename(img_path)}', fontsize=12)
        print(f"ERROR: Failed to load {img_path}: {e}")
    ax.axis('off')

# 如果没有足够的图像，填充空白
for idx in range(len(available_images), 3):
    if idx >= len(positions):
        break
    row, col = positions[idx]
    ax = fig.add_subplot(gs[row, col])
    ax.text(0.5, 0.5, 'No Additional\nResults', ha='center', va='center', fontsize=12)
    ax.set_title('Empty', fontsize=12)
    ax.axis('off')

# 保存图像
plt.suptitle('ViT Depth-12 NNS Basic Results: Fish Image Analysis', 
             fontsize=16, fontweight='bold', y=0.95)
plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
plt.close()

print(f"\n{'='*60}")
print("SUCCESS: Basic Grid Generated!")
print(f"{'='*60}")
print(f"\nFile: {output_path}")
print(f"Layout: 2x2 grid")
print(f"Content: Original + {len(available_images)} NNS results")
print("\nNote: This is a basic version. For full 12-layer visualization,")
print("the NNS scan needs to complete all attention layer processing.")
print(f"\n{'='*60}")

# 尝试打开图像
import platform
if platform.system() == 'Windows':
    try:
        os.startfile(output_path)
        print("Image opened automatically!")
    except:
        print("Please manually open the image file.")
