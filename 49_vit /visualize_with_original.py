"""
使用depth=6模型生成可视化，并创建包含原图的网格布局
"""
import os
import sys
import torch
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'model'))
sys.path.append(os.path.join(current_dir, 'function'))
sys.path.append(os.path.join(current_dir, 'NNS'))

from models import VisionTransformer
from dataset import ScanDataSet
from Scannerpic1107_cls_nonet import NNS_NONET

def create_grid_with_original(scan_result_dir, original_image_path, output_path, num_layers=6):
    """
    创建包含原图的网格可视化
    布局: 2行6列 (原图 + 选择的层)
    """
    print(f"\n创建网格可视化...")
    print(f"  扫描结果目录: {scan_result_dir}")
    print(f"  原图路径: {original_image_path}")
    
    # 查找所有层目录
    layer_dirs = []
    if os.path.exists(scan_result_dir):
        for item in os.listdir(scan_result_dir):
            item_path = os.path.join(scan_result_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                layer_dirs.append((int(item), item_path))
        layer_dirs.sort()
    
    print(f"  找到 {len(layer_dirs)} 个层")
    
    if len(layer_dirs) == 0:
        print("  ⚠️  没有找到层目录")
        return
    
    # 选择要显示的层（均匀分布）
    if len(layer_dirs) <= 11:
        selected_layers = layer_dirs
    else:
        # 选择11层（原图占1格）
        indices = np.linspace(0, len(layer_dirs)-1, 11, dtype=int)
        selected_layers = [layer_dirs[i] for i in indices]
    
    # 创建图像
    fig = plt.figure(figsize=(24, 8))
    gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.2)
    
    # 第一个位置：原图
    ax = fig.add_subplot(gs[0, 0])
    if os.path.exists(original_image_path):
        original_img = Image.open(original_image_path).convert('RGB')
        ax.imshow(original_img)
        ax.set_title('Original Image', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center')
        ax.set_title('Original Image', fontsize=12)
    ax.axis('off')
    
    # 其余位置：层的注意力图
    positions = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                 (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    
    for idx, (layer_num, layer_path) in enumerate(selected_layers):
        if idx >= len(positions):
            break
        
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        
        # 查找avg_cls_attn图像
        img_path = os.path.join(layer_path, 'avg_cls_attn_head0.png')
        if not os.path.exists(img_path):
            # 尝试其他可能的文件名
            alternatives = ['avg_cls_attn.png', 'attention_map.png']
            for alt in alternatives:
                alt_path = os.path.join(layer_path, alt)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f'Layer {layer_num:02d}', fontsize=11)
        else:
            ax.text(0.5, 0.5, f'Layer {layer_num}\nNot Found', ha='center', va='center')
            ax.set_title(f'Layer {layer_num:02d}', fontsize=11)
        ax.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\n✓ 网格图像已保存: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--device', type=str, default='0', help='GPU设备')
    parser.add_argument('--depth', type=int, default=6, help='模型深度')
    parser.add_argument('--num_heads', type=int, default=3, help='注意力头数')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch大小')
    parser.add_argument('--classes', type=int, default=10, help='类别数')
    parser.add_argument('--img_size', type=int, default=32, help='图像大小')
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    print(f"模型配置: depth={args.depth}, heads={args.num_heads}, patch_size={args.patch_size}")
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=args.classes,
        embed_dim=192,
        depth=args.depth,
        num_heads=args.num_heads
    )
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ 模型加载成功")
    
    # 加载图像
    print(f"\n加载图像: {args.image_path}")
    dataset = ScanDataSet(os.path.dirname(args.image_path))
    
    # 找到目标图像
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    print(f"图像名称: {image_name}")
    
    # 处理图像
    for idx, (img_path, label) in enumerate(dataset.imgs):
        current_name = os.path.splitext(os.path.basename(img_path))[0]
        if current_name == image_name:
            print(f"\n找到图像: {img_path}")
            
            # 加载并转换图像
            image_tensor = dataset[idx][0]  # (C, H, W)
            image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
            
            # 运行NNS
            print("\n开始NNS处理...")
            params = {
                'cls_token': model.cls_token,
                'pos_embed': model.pos_embed
            }
            
            nns = NNS_NONET(
                model=model,
                imageTensor=image_tensor[0],
                img_name=image_name,
                name_i=idx,
                device=device,
                model_name='VisionTransformer',
                model_type='VisionTransformer',
                dataset='tiny200ILSVRC',
                patch_size=args.patch_size,
                num_heads=args.num_heads,
                depth=args.depth,
                params=params
            )
            
            # 检查是否已处理
            if not os.path.exists(nns.rootdir):
                print(f"  NNS目录不存在，开始处理...")
                LN = True  # 使用Layer Normalization
                nns.get_scanpic(LN)
                print(f"  ✓ NNS处理完成")
            else:
                print(f"  ✓ NNS目录已存在，跳过处理")
            
            # 创建网格图像
            scan_attn_dir = os.path.join(nns.rootdir, 'scan_attn_map')
            output_grid = os.path.join(nns.rootdir, f'{image_name}_grid_with_original.png')
            
            create_grid_with_original(
                scan_result_dir=scan_attn_dir,
                original_image_path=args.image_path,
                output_path=output_grid,
                num_layers=args.depth
            )
            
            print(f"\n✓ 完成！")
            print(f"  NNS结果目录: {nns.rootdir}")
            print(f"  网格图像: {output_grid}")
            break
    else:
        print(f"\n⚠️  未找到图像: {image_name}")

if __name__ == '__main__':
    main()

