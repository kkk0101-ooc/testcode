"""
为ImageNet + ViT-base-16创建10层网格图
"""
import os
from PIL import Image, ImageDraw, ImageFont

# 配置
base_dir = r"scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained\0_n03976467_9546_wrong_pred732scan_attn_map"
output_path = r"scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained\imagenet_vit_base_10layers_grid.png"

# 找到所有层
layers = []
if os.path.exists(base_dir):
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            layers.append(item)

print(f"找到 {len(layers)} 个层: {layers}")

# 选择10个层（均匀分布）
if len(layers) >= 10:
    step = len(layers) / 10
    selected_layers = [layers[int(i * step)] for i in range(10)]
else:
    selected_layers = layers[:10]

print(f"选择的10个层: {selected_layers}")

# 加载图像
images = []
labels = []

for idx, layer in enumerate(selected_layers):
    img_path = os.path.join(base_dir, layer, 'input_attn', 'head0', 'avg_cls_attn_head0.png')
    
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            images.append(img)
            labels.append(f"Layer {layer}")
            print(f"  {idx}. Layer {layer}: ✓")
        except Exception as e:
            print(f"  {idx}. Layer {layer}: ✗ ({e})")
    else:
        print(f"  {idx}. Layer {layer}: ✗ (文件不存在)")

# 创建2x5网格
rows, cols = 2, 5
img_width, img_height = 512, 512
grid_width = cols * img_width
grid_height = rows * img_height

grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
draw = ImageDraw.Draw(grid_image)

try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

for idx, (img, label) in enumerate(zip(images, labels)):
    row = idx // cols
    col = idx % cols
    x = col * img_width
    y = row * img_height
    
    grid_image.paste(img, (x, y))
    
    # 添加标签
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 半透明背景
    draw.rectangle([x + 10, y + 10, x + text_width + 30, y + text_height + 30],
                  fill=(255, 255, 255, 220), outline=(0, 0, 0))
    draw.text((x + 20, y + 20), label, fill='black', font=font)

# 保存
grid_image.save(output_path)
print(f"\n✓ 已保存10层网格图到: {output_path}")
print(f"  网格大小: 2×5")
print(f"  总尺寸: {grid_width}×{grid_height} 像素")
print(f"  包含层: {', '.join([f'Layer {l}' for l in selected_layers])}")

