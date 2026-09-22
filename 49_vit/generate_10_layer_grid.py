"""
生成10个子图的网格（类似您图片中的效果）
可以从不同类型的可视化中选择10个
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import argparse
import glob

def generate_10_grid(base_dir, output_path=None, visualization_types=None):
    """
    生成10个子图的网格
    
    Args:
        base_dir: 结果基础目录（如 0_n01443537_24scan_attn_map/）
        output_path: 输出路径
        visualization_types: 要包含的可视化类型列表
    """
    
    if visualization_types is None:
        # 自动发现所有层
        layers = []
        if os.path.exists(base_dir):
            for item in sorted(os.listdir(base_dir)):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    layers.append(item)
        
        print(f"  发现 {len(layers)} 个层: {layers}")
        
        if len(layers) >= 10:
            # 如果有10个或更多层，选择均匀分布的10个层的avg图
            step = len(layers) / 10
            selected_layers = [layers[int(i * step)] for i in range(10)]
            visualization_types = [(layer, 'avg_cls_attn_head0.png') for layer in selected_layers]
        else:
            # 如果层数少于10，混合使用avg和top patches
            visualization_types = []
            for layer in layers:
                visualization_types.append((layer, 'avg_cls_attn_head0.png'))
            
            # 添加top patches直到达到10个
            for layer in layers:
                if len(visualization_types) >= 10:
                    break
                visualization_types.append((layer, 'top0_patch_*_attn_head0.png'))
            
            for layer in layers:
                if len(visualization_types) >= 10:
                    break
                visualization_types.append((layer, 'top1_patch_*_attn_head0.png'))
            
            for layer in layers:
                if len(visualization_types) >= 10:
                    break
                visualization_types.append((layer, 'top2_patch_*_attn_head0.png'))
    
    images = []
    labels = []
    
    print(f"从 {base_dir} 收集图像...")
    
    for idx, (layer, pattern) in enumerate(visualization_types):
        layer_path = os.path.join(base_dir, layer, 'input_attn', 'head0')
        
        if not os.path.exists(layer_path):
            print(f"  警告: 层 {layer} 不存在")
            continue
        
        # 查找匹配的文件
        search_pattern = os.path.join(layer_path, pattern)
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            img_path = matching_files[0]
            try:
                img = Image.open(img_path)
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                images.append(img)
                
                # 创建标签
                filename = os.path.basename(img_path)
                label = f"{idx}.png\nLayer {layer}"
                labels.append(label)
                print(f"  {idx}.png: ✓ {filename}")
                
                if len(images) >= 10:
                    break
            except Exception as e:
                print(f"  {idx}.png: ✗ ({e})")
    
    if len(images) < 10:
        print(f"\n警告: 只找到 {len(images)} 个图像，少于10个")
        # 用空白图填充
        while len(images) < 10:
            blank = Image.new('RGB', (256, 256), color='white')
            images.append(blank)
            labels.append(f"{len(images)-1}.png\n(空)")
    
    # 创建2x5网格
    rows, cols = 2, 5
    img_width, img_height = 256, 256
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for idx, (img, label) in enumerate(zip(images[:10], labels[:10])):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        
        grid_image.paste(img, (x, y))
        
        # 添加标签
        lines = label.split('\n')
        y_offset = 10
        for line in lines:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 半透明背景
            draw.rectangle([x + 5, y + y_offset, x + text_width + 15, y + y_offset + text_height + 10],
                          fill=(255, 255, 255, 200))
            draw.text((x + 10, y + y_offset + 5), line, fill='black', font=font)
            y_offset += text_height + 15
    
    # 保存
    if output_path is None:
        parent_dir = os.path.dirname(base_dir)
        base_name = os.path.basename(base_dir).replace('scan_attn_map', '')
        output_path = os.path.join(parent_dir, f"{base_name}_10grid.png")
    
    grid_image.save(output_path)
    print(f"\n✓ 已保存10子图网格到: {output_path}")
    print(f"  网格大小: 2×5")
    print(f"  总尺寸: {grid_width}×{grid_height} 像素")
    
    return output_path


def find_and_generate_all(base_dir):
    """自动查找所有scan_attn_map目录并生成10子图网格"""
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'scan_attn_map' in dir_name:
                scan_attn_dir = os.path.join(root, dir_name)
                print(f"\n{'='*70}")
                print(f"处理: {dir_name}")
                print(f"{'='*70}")
                
                result = generate_10_grid(scan_attn_dir)
                if result:
                    count += 1
    
    print(f"\n{'='*70}")
    print(f"总计处理了 {count} 个图像")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成10个子图的网格')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='scan_attn_map目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图像路径')
    parser.add_argument('--auto_find', type=str,
                        default=r'D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit',
                        help='自动查找并处理所有图像')
    
    args = parser.parse_args()
    
    if args.base_dir:
        generate_10_grid(args.base_dir, args.output)
    else:
        print("自动查找并生成所有10子图网格...")
        find_and_generate_all(args.auto_find)

