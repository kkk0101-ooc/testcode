"""
将tem0目录中的0-9.png合并成一个网格图
这些图通常是模型处理过程中不同步骤的可视化
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import argparse

def combine_tem0_to_grid(tem0_dir, output_path=None, grid_size=(2, 5)):
    """
    将tem0目录中的0-9.png合并成网格图
    
    Args:
        tem0_dir: tem0目录路径
        output_path: 输出图像路径
        grid_size: 网格大小 (rows, cols)
    """
    
    if not os.path.exists(tem0_dir):
        print(f"错误: 目录不存在 {tem0_dir}")
        return False
    
    # 查找所有数字命名的PNG文件
    image_files = []
    for i in range(10):  # 0-9
        img_path = os.path.join(tem0_dir, f"{i}.png")
        if os.path.exists(img_path):
            image_files.append((i, img_path))
    
    if not image_files:
        print(f"未找到0-9.png文件")
        return False
    
    print(f"找到 {len(image_files)} 个图像: {[i for i, _ in image_files]}")
    
    rows, cols = grid_size
    max_images = rows * cols
    image_files = image_files[:max_images]
    
    # 加载所有图像
    images = []
    labels = []
    for idx, img_path in image_files:
        try:
            img = Image.open(img_path)
            # 调整到统一尺寸（如果需要）
            target_size = (256, 256)  # 可以根据需要调整
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(img)
            labels.append(idx)
            print(f"  {idx}.png: ✓")
        except Exception as e:
            print(f"  {idx}.png: ✗ ({e})")
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
    draw = ImageDraw.Draw(grid_image)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid_image.paste(img, (x, y))
        
        # 添加标签
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
        
        text = f"{label}.png"
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
        parent_dir = os.path.dirname(tem0_dir)
        base_name = os.path.basename(parent_dir)
        output_path = os.path.join(parent_dir, f"{base_name}_tem0_grid.png")
    
    grid_image.save(output_path)
    print(f"\n✓ 已保存合并图像到: {output_path}")
    print(f"  合并了 {len(images)} 个图像")
    print(f"  网格大小: {rows}×{cols}")
    
    return output_path


def find_and_combine_all(base_dir):
    """
    在结果目录中查找所有tem0目录并合并
    """
    base_dir = os.path.abspath(base_dir)
    
    if not os.path.exists(base_dir):
        print(f"结果目录不存在: {base_dir}")
        return
    
    count = 0
    # 查找所有tem0目录
    for root, dirs, files in os.walk(base_dir):
        if 'tem0' in dirs:
            tem0_dir = os.path.join(root, 'tem0')
            
            # 检查是否有0.png文件
            if os.path.exists(os.path.join(tem0_dir, '0.png')):
                print(f"\n{'='*70}")
                print(f"处理: {os.path.basename(root)}")
                print(f"{'='*70}")
                
                result = combine_tem0_to_grid(tem0_dir)
                if result:
                    count += 1
    
    print(f"\n{'='*70}")
    print(f"总计处理了 {count} 个图像")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并tem0目录中的0-9.png')
    parser.add_argument('--tem0_dir', type=str, default=None,
                        help='tem0目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图像路径')
    parser.add_argument('--base_dir', type=str, 
                        default=r'D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit',
                        help='结果基础目录，自动查找所有tem0目录')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[2, 5],
                        help='网格大小 (rows cols)，默认 2 5')
    
    args = parser.parse_args()
    
    if args.tem0_dir:
        # 合并指定目录
        combine_tem0_to_grid(args.tem0_dir, args.output, 
                           grid_size=tuple(args.grid_size))
    else:
        # 自动查找并合并
        print("自动查找并合并所有tem0目录...")
        find_and_combine_all(args.base_dir)

