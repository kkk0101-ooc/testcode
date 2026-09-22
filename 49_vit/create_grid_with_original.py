"""
创建包含原图的12层注意力可视化网格（2×6布局：原图+11层）
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import argparse

def create_grid_with_original(scan_attn_dir, original_img_path, output_path=None, grid_size=(2, 6)):
    """
    创建包含原图的注意力可视化网格
    
    Args:
        scan_attn_dir: scan_attn_map目录路径
        original_img_path: 原始图像路径
        output_path: 输出图像路径
        grid_size: 网格大小 (rows, cols)，默认(2, 6)表示原图+11层注意力
    """
    
    if not os.path.exists(scan_attn_dir):
        print(f"错误: scan_attn_map目录不存在 {scan_attn_dir}")
        return False
    
    if not os.path.exists(original_img_path):
        print(f"错误: 原图不存在 {original_img_path}")
        return False
    
    # 查找所有层目录（02, 06, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46）
    layer_dirs = sorted([d for d in os.listdir(scan_attn_dir) 
                        if os.path.isdir(os.path.join(scan_attn_dir, d)) and d.isdigit()],
                       key=lambda x: int(x))
    
    if not layer_dirs:
        print(f"未找到层目录")
        return False
    
    print(f"找到 {len(layer_dirs)} 个层: {', '.join(layer_dirs)}")
    
    rows, cols = grid_size
    # 需要cols-1个注意力层（因为第一个位置是原图）
    num_attention_layers = cols * rows - 1
    
    # 均匀选择层（从12层中选择11层）
    if len(layer_dirs) >= num_attention_layers:
        # 均匀分布选择
        step = len(layer_dirs) / num_attention_layers
        selected_layers = [layer_dirs[int(i * step)] for i in range(num_attention_layers)]
    else:
        selected_layers = layer_dirs[:num_attention_layers]
    
    print(f"选择的层: {', '.join(selected_layers)}")
    
    # 加载原图
    try:
        original_img = Image.open(original_img_path)
        # 调整到512×512
        if original_img.size != (512, 512):
            original_img = original_img.resize((512, 512), Image.Resampling.LANCZOS)
        print(f"✓ 加载原图: {original_img_path}")
    except Exception as e:
        print(f"加载原图失败: {e}")
        return False
    
    # 为每一层找到注意力图
    attention_images = []
    attention_type = 'avg_cls_attn_head0'  # 使用平均注意力图
    
    for layer_num in selected_layers:
        layer_path = os.path.join(scan_attn_dir, layer_num, 'input_attn', 'head0', f'{attention_type}.png')
        
        if os.path.exists(layer_path):
            try:
                img = Image.open(layer_path)
                if img.size != (512, 512):
                    img = img.resize((512, 512), Image.Resampling.LANCZOS)
                attention_images.append((layer_num, img))
                print(f"  层 {layer_num}: ✓")
            except Exception as e:
                print(f"  层 {layer_num}: ✗ (加载失败: {e})")
        else:
            print(f"  层 {layer_num}: ✗ (未找到 {attention_type}.png)")
    
    if len(attention_images) < num_attention_layers:
        print(f"警告: 只找到 {len(attention_images)} 个层，需要 {num_attention_layers} 个")
        # 如果不够，重复最后一个层
        while len(attention_images) < num_attention_layers:
            if attention_images:
                attention_images.append(attention_images[-1])
    
    # 创建网格
    img_width, img_height = 512, 512
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    # 创建空白画布
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # 放置图像
    draw = ImageDraw.Draw(grid_image)
    
    # 尝试加载字体
    try:
        # Windows字体路径
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, 36)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 第一张：原图
    grid_image.paste(original_img, (0, 0))
    text = "Original"
    # 在图像左上角添加文本（白色背景框）
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    # 绘制白色背景
    draw.rectangle([10, 10, 10 + text_width + 10, 10 + text_height + 10], fill='white', outline='black', width=2)
    draw.text((15, 15), text, fill='black', font=font)
    
    # 后续：注意力层
    for idx, (layer_num, img) in enumerate(attention_images[:num_attention_layers]):
        pos = idx + 1  # 从位置1开始（位置0是原图）
        row = pos // cols
        col = pos % cols
        x = col * img_width
        y = row * img_height
        grid_image.paste(img, (x, y))
        
        # 添加层标签
        text = f"Layer {layer_num}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        # 绘制白色背景
        draw.rectangle([x + 10, y + 10, x + 10 + text_width + 10, y + 10 + text_height + 10], 
                       fill='white', outline='black', width=2)
        draw.text((x + 15, y + 15), text, fill='black', font=font)
    
    # 保存
    if output_path is None:
        parent_dir = os.path.dirname(scan_attn_dir)
        output_path = os.path.join(parent_dir, 'n01443537_24_depth12_2x6_grid.png')
    
    grid_image.save(output_path, quality=95, dpi=(200, 200))
    print(f"\n✓ 已保存网格图到: {output_path}")
    print(f"  尺寸: {grid_width}×{grid_height} 像素")
    print(f"  布局: {rows}×{cols} (原图 + {num_attention_layers}层注意力)")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='创建包含原图的注意力可视化网格')
    parser.add_argument('--scan_attn_dir', type=str, 
                       default='scan_results/scan_vit/114VisionTransformer_tiny200ILSVRC/depth12_final/0_n01443537_24scan_attn_map',
                       help='scan_attn_map目录路径')
    parser.add_argument('--original_img', type=str,
                       default='D:/project/vit_114/NNS_model_and_dataset/dataset/tiny-imagenet-5/test111/n01443537/n01443537_24.JPEG',
                       help='原始图像路径')
    parser.add_argument('--output', type=str, default=None, help='输出图像路径')
    parser.add_argument('--grid_rows', type=int, default=2, help='网格行数')
    parser.add_argument('--grid_cols', type=int, default=6, help='网格列数')
    
    args = parser.parse_args()
    
    create_grid_with_original(
        args.scan_attn_dir,
        args.original_img,
        args.output,
        grid_size=(args.grid_rows, args.grid_cols)
    )

