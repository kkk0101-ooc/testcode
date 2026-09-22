"""
为两张图像生成10层网格图
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

def generate_10grid_for_image(scan_attn_map_dir, image_name, output_filename):
    """为单个图像生成10层网格"""
    
    # 找到所有层
    layers = []
    if os.path.exists(scan_attn_map_dir):
        for item in sorted(os.listdir(scan_attn_map_dir)):
            item_path = os.path.join(scan_attn_map_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                layers.append(item)
    
    print(f"\n{image_name}:")
    print(f"  找到 {len(layers)} 个层: {layers}")
    
    if len(layers) < 10:
        print(f"  警告: 只有{len(layers)}个层，少于10个")
        selected_layers = layers
    else:
        # 选择10个层（均匀分布）
        step = len(layers) / 10
        selected_layers = [layers[int(i * step)] for i in range(10)]
    
    print(f"  选择的层: {selected_layers}")
    
    # 加载图像
    images = []
    labels = []
    
    for idx, layer in enumerate(selected_layers):
        img_path = os.path.join(scan_attn_map_dir, layer, 'input_attn', 'head0', 'avg_cls_attn_head0.png')
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                images.append(img)
                labels.append(f"Layer {layer}")
                print(f"    {idx}. Layer {layer}: ✓")
            except Exception as e:
                print(f"    {idx}. Layer {layer}: ✗ ({e})")
        else:
            print(f"    {idx}. Layer {layer}: ✗ (文件不存在)")
    
    if not images:
        print(f"  错误: 没有找到任何图像")
        return None
    
    # 填充到10个（如果不足）
    while len(images) < 10:
        blank = Image.new('RGB', (512, 512), color='white')
        images.append(blank)
        labels.append("(空)")
    
    # 创建2x5网格
    rows, cols = 2, 5
    img_width, img_height = 512, 512
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
    
    for idx, (img, label) in enumerate(zip(images[:10], labels[:10])):
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
                      fill=(255, 255, 255, 230), outline=(0, 0, 0))
        draw.text((x + 20, y + 20), label, fill='black', font=font)
    
    # 保存
    grid_image.save(output_filename)
    print(f"\n  ✓ 已保存: {output_filename}")
    return output_filename

# 处理两张图像
base_dir = r"scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained"

# 图像1: n03976467_9546
scan_attn_1 = os.path.join(base_dir, "0_n03976467_9546_wrong_pred732scan_attn_map")
output_1 = os.path.join(base_dir, "n03976467_9546_10layers_grid.png")
if os.path.exists(scan_attn_1):
    generate_10grid_for_image(scan_attn_1, "n03976467_9546", output_1)

# 图像2: n01443537_24
scan_attn_2 = None
# 查找n01443537_24的scan_attn_map目录
if os.path.exists(base_dir):
    for item in os.listdir(base_dir):
        if 'n01443537_24' in item and 'scan_attn_map' in item:
            scan_attn_2 = os.path.join(base_dir, item)
            break

if scan_attn_2 and os.path.exists(scan_attn_2):
    output_2 = os.path.join(base_dir, "n01443537_24_10layers_grid.png")
    generate_10grid_for_image(scan_attn_2, "n01443537_24", output_2)
else:
    print(f"\n警告: 未找到n01443537_24的scan_attn_map目录")

print("\n" + "="*70)
print("所有网格图已生成！")
print("="*70)

