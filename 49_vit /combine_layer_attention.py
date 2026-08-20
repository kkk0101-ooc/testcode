"""
合并不同层的注意力可视化到一张网格图
"""
import os
import sys
from PIL import Image
import argparse

def combine_layer_attention(result_dir, output_path=None, grid_size=(2, 3), attention_type='avg_cls_attn_head0'):
    """
    合并不同层的注意力图到一张网格图
    
    Args:
        result_dir: 结果目录（如 0_n01443537_198_wrong_pred6scan_attn_map/）
        output_path: 输出图像路径（默认保存在result_dir的父目录）
        grid_size: 网格大小 (rows, cols)
        attention_type: 注意力图类型（avg_cls_attn_head0, top0_patch等）
    """
    
    if not os.path.exists(result_dir):
        print(f"错误: 目录不存在 {result_dir}")
        return False
    
    # 查找所有层目录（02, 06, 10, 14, 18, 22）
    layer_dirs = sorted([d for d in os.listdir(result_dir) 
                        if os.path.isdir(os.path.join(result_dir, d)) and d.isdigit()])
    
    if not layer_dirs:
        print(f"未找到层目录")
        return False
    
    print(f"找到 {len(layer_dirs)} 个层: {', '.join(layer_dirs)}")
    
    rows, cols = grid_size
    max_layers = rows * cols
    layer_dirs = layer_dirs[:max_layers]
    
    # 为每一层找到注意力图
    attention_images = []
    for layer_num in layer_dirs:
        layer_path = os.path.join(result_dir, layer_num, 'input_attn', 'head0', f'{attention_type}.png')
        
        if os.path.exists(layer_path):
            attention_images.append((layer_num, layer_path))
            print(f"  层 {layer_num}: OK")
        else:
            print(f"  层 {layer_num}: X (未找到 {attention_type}.png)")
    
    if not attention_images:
        print("未找到任何注意力图")
        return False
    
    # 加载所有图像
    images = []
    layer_labels = []
    for layer_num, img_path in attention_images:
        try:
            img = Image.open(img_path)
            # 保持原始尺寸或调整到统一尺寸
            if img.size != (512, 512):
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
            images.append(img)
            layer_labels.append(layer_num)
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            continue
    
    if not images:
        print("没有成功加载的图像")
        return False
    
    # 创建网格
    img_width, img_height = images[0].size
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    # 创建空白画布
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # 放置图像
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid_image)
    
    for idx, (img, label) in enumerate(zip(images[:max_layers], layer_labels[:max_layers])):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid_image.paste(img, (x, y))
        
        # 添加层标签
        try:
            # 尝试使用默认字体
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        text = f"Layer {label}"
        # 在图像左上角添加文本
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 绘制半透明背景
        draw.rectangle([x + 5, y + 5, x + text_width + 15, y + text_height + 15], 
                      fill=(255, 255, 255, 200))
        draw.text((x + 10, y + 10), text, fill='black', font=font)
    
    # 保存
    if output_path is None:
        parent_dir = os.path.dirname(result_dir)
        base_name = os.path.basename(result_dir).replace('scan_attn_map', '')
        output_path = os.path.join(parent_dir, f"{base_name}_layer_attention_grid.png")
    
    grid_image.save(output_path)
    print(f"\nOK 已保存合并图像到: {output_path}")
    print(f"  合并了 {len(images)} 个层的注意力图")
    print(f"  网格大小: {rows}x{cols}")
    
    return output_path


def find_and_combine_all(base_dir, attention_type='avg_cls_attn_head0'):
    """
    在结果目录中查找所有scan_attn_map目录并合并
    """
    base_dir = os.path.abspath(base_dir)
    
    if not os.path.exists(base_dir):
        print(f"结果目录不存在: {base_dir}")
        return
    
    count = 0
    # 查找所有包含scan_attn_map的目录
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'scan_attn_map' in dir_name:
                scan_attn_dir = os.path.join(root, dir_name)
                print(f"\n{'='*70}")
                print(f"处理: {dir_name}")
                print(f"{'='*70}")
                
                result = combine_layer_attention(scan_attn_dir, attention_type=attention_type)
                if result:
                    count += 1
    
    print(f"\n{'='*70}")
    print(f"总计处理了 {count} 个图像")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并不同层的注意力可视化')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='结果目录（如 scan_attn_map/）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图像路径')
    parser.add_argument('--base_dir', type=str, 
                        default=r'D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit',
                        help='结果基础目录，自动查找所有需要合并的图像')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[2, 3],
                        help='网格大小 (rows cols)，默认 2 3')
    parser.add_argument('--attention_type', type=str, default='avg_cls_attn_head0',
                        help='注意力图类型（avg_cls_attn_head0, top0_patch等）')
    
    args = parser.parse_args()
    
    if args.result_dir:
        # 合并指定目录
        combine_layer_attention(args.result_dir, args.output, 
                               grid_size=tuple(args.grid_size),
                               attention_type=args.attention_type)
    else:
        # 自动查找并合并
        print("自动查找并合并所有层的注意力图...")
        find_and_combine_all(args.base_dir, attention_type=args.attention_type)

