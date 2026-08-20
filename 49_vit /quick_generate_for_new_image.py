"""快速为任意新图片生成NNS效果图"""
import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import zoom

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.vit_final import VisionTransformer
from NNS.Scannerpic1107_cls_nonet import NNS_NONET

def process_new_image(image_path, output_name=None):
    """处理新图片的完整流程"""
    
    print(f"{'='*80}")
    print(f"开始处理新图片: {os.path.basename(image_path)}")
    print(f"{'='*80}")
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"ERROR: 图片文件不存在: {image_path}")
        return None
    
    # 配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型参数
    depth = 12
    num_heads = 3
    patch_size = 4
    classes = 10
    
    # 模型路径
    model_path = r"d:\project\vit_114\NNS_model_and_dataset\models\vit_3channel_mutlihead\tiny200ILSVRC_VisionTransformer_32imgsize_10classes_45threshold_3heads_12depth_4patch_size_VisionTransformer_200_final.pth"
    
    if not os.path.exists(model_path):
        print(f"ERROR: 模型文件不存在: {model_path}")
        return None
    
    # 生成输出名称
    if output_name is None:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_name = f"{image_name}_auto"
    
    print(f"输出名称: {output_name}")
    
    # 步骤1: 加载模型
    print("\n[1/4] 加载模型...")
    model = VisionTransformer(
        img_size=[32], 
        patch_size=patch_size, 
        num_classes=classes, 
        embed_dim=192, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_ratio=4,
        qkv_bias=False
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("SUCCESS: Model loaded")
    
    # 步骤2: 预处理图片
    print("\n[2/4] 预处理图片...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"SUCCESS: Image preprocessing completed: {image.size} -> (32, 32)")
    except Exception as e:
        print(f"ERROR: 图片预处理失败: {e}")
        return None
    
    # 获取预测
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[pred].item()
        print(f"SUCCESS: Model prediction: 类别{pred} (置信度: {confidence:.2%})")
    
    # 步骤3: NNS扫描
    print("\n[3/4] 执行NNS扫描...")
    
    # 提取参数
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.clone()
    
    # 创建NNS对象
    model_name = f"VisionTransformer_auto_{output_name}"
    
    nns = NNS_NONET(
        model, image_tensor[0], output_name, 0, device, 
        model_name, "VisionTransformer", 'tiny200ILSVRC', 
        patch_size, num_heads, depth, params
    )
    
    print(f"SUCCESS: NNS输出目录: {nns.rootdir}")
    
    # 确保目录存在
    os.makedirs(nns.rootdir, exist_ok=True)
    os.makedirs(nns.scan_attn_map, exist_ok=True)
    
    # 运行扫描
    nns.get_scanpic(50)
    print("SUCCESS: NNS scan completed")
    
    # 步骤4: 生成优化网格图
    print("\n[4/4] 生成优化网格图...")
    
    grid_path = create_optimized_grid_for_image(
        nns.rootdir, 
        image_path, 
        output_name, 
        pred, 
        confidence
    )
    
    if grid_path:
        print(f"SUCCESS: 优化网格图已生成: {grid_path}")
        
        # 自动打开
        import platform
        if platform.system() == 'Windows':
            try:
                os.startfile(grid_path)
                print("SUCCESS: Grid generation completed")
            except:
                pass
        
        return grid_path
    else:
        print("ERROR: 网格图生成失败")
        return None

def create_optimized_grid_for_image(result_dir, original_image_path, image_name, pred_class, confidence):
    """为图片创建优化网格图"""
    
    scan_attn_dir = os.path.join(result_dir, 'scan_attn_map')
    if not os.path.exists(scan_attn_dir):
        print(f"ERROR: {scan_attn_dir} not found")
        return None
    
    output_path = os.path.join(result_dir, f"{image_name}_OPTIMIZED_12layers_grid.png")
    
    # 查找所有层
    layer_dirs = []
    for item in os.listdir(scan_attn_dir):
        item_path = os.path.join(scan_attn_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            layer_dirs.append((int(item), item_path))
    layer_dirs.sort()
    
    if len(layer_dirs) == 0:
        print("ERROR: 没有找到注意力层")
        return None
    
    # 选择11层
    if len(layer_dirs) >= 11:
        step = len(layer_dirs) / 11
        selected_layers = [layer_dirs[int(i * step)] for i in range(11)]
    else:
        selected_layers = layer_dirs
    
    print(f"选择的层: {[l[0] for l in selected_layers]}")
    
    # 创建3x4网格
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # 原图
    ax = fig.add_subplot(gs[0, 0])
    if os.path.exists(original_image_path):
        original_img = Image.open(original_image_path).convert('RGB')
        ax.imshow(original_img)
        ax.set_title(f'Original Image\n{image_name}\nClass {pred_class} ({confidence:.1%})', 
                     fontsize=16, fontweight='bold', pad=12)
    else:
        ax.text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center', fontsize=12)
        ax.set_title('Original Image', fontsize=16, pad=12)
    ax.axis('off')
    
    # 层注意力图
    positions = [(0, 1), (0, 2), (0, 3),
                 (1, 0), (1, 1), (1, 2), (1, 3),
                 (2, 0), (2, 1), (2, 2), (2, 3)]
    
    success_count = 0
    
    for idx, (layer_num, layer_path) in enumerate(selected_layers):
        if idx >= len(positions):
            break
        
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        
        # 尝试创建增强的注意力图
        enhanced_img = create_enhanced_attention_from_tensor(layer_path)
        
        if enhanced_img is not None:
            ax.imshow(enhanced_img)
            success_count += 1
        else:
            # 回退到PNG图像
            png_paths = [
                os.path.join(layer_path, 'input_attn', 'head0', 'avg_cls_attn_head0.png'),
                os.path.join(layer_path, 'input_attn', 'head1', 'avg_cls_attn_head1.png'),
                os.path.join(layer_path, 'input_attn', 'head2', 'avg_cls_attn_head2.png'),
            ]
            
            img_loaded = False
            for png_path in png_paths:
                if os.path.exists(png_path):
                    img = Image.open(png_path)
                    ax.imshow(img)
                    img_loaded = True
                    success_count += 1
                    break
            
            if not img_loaded:
                ax.text(0.5, 0.5, f'Layer {layer_num}\nNot Available', ha='center', va='center', fontsize=10)
        
        # 设置标题
        if layer_num <= 10:
            layer_desc = f'Layer {layer_num:02d}\n(Shallow)'
        elif layer_num <= 30:
            layer_desc = f'Layer {layer_num:02d}\n(Middle)'
        else:
            layer_desc = f'Layer {layer_num:02d}\n(Deep)'
        
        ax.set_title(layer_desc, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    # 保存网格图
    plt.suptitle(f'ViT Depth-12 NNS Visualization: {image_name}\nOptimized Attention Maps (Class {pred_class}, {confidence:.1%} confidence)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 添加说明
    info_text = f'Enhanced with Percentile Method | Best Head Selection | High Resolution\n'
    info_text += f'Red/Yellow = High Attention | Blue = Low Attention | {success_count}/11 layers processed'
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=12, style='italic')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()
    
    return output_path

def create_enhanced_attention_from_tensor(layer_path):
    """从tensor创建增强的注意力图"""
    
    tensor_path = os.path.join(layer_path, 'image_attn_tensor.pt')
    if not os.path.exists(tensor_path):
        return None
    
    try:
        # 加载注意力tensor
        attn_tensor = torch.load(tensor_path, map_location='cpu')
        
        if len(attn_tensor.shape) == 4:  # (batch, heads, seq_len, seq_len)
            cls_attn = attn_tensor[0, :, 0, 1:]  # 所有head的CLS注意力
            avg_cls_attn = torch.mean(cls_attn, dim=0)  # 平均所有head
        else:
            return None
        
        # 转换为numpy
        attn_np = avg_cls_attn.detach().cpu().numpy()
        
        # 计算patch数量
        num_patches = len(attn_np)
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size != num_patches:
            return None
        
        # 重塑为2D
        attn_2d = attn_np.reshape(patch_size, patch_size)
        
        # 百分位数增强
        p_low = np.percentile(attn_2d, 5)
        p_high = np.percentile(attn_2d, 95)
        attn_enhanced = np.clip(attn_2d, p_low, p_high)
        attn_enhanced = (attn_enhanced - p_low) / (p_high - p_low + 1e-8)
        
        # 上采样到512x512
        upscale_factor = 512 // patch_size
        attn_upscaled = zoom(attn_enhanced, upscale_factor, order=3)
        
        # 转换为PIL图像
        import matplotlib.cm as cm
        try:
            colormap = cm.get_cmap('hot')
        except AttributeError:
            colormap = plt.cm.hot
        
        colored_img = colormap(attn_upscaled)
        colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
        
        return Image.fromarray(colored_img)
        
    except Exception as e:
        print(f"处理tensor失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='快速为新图片生成NNS效果图')
    parser.add_argument('--image_path', type=str, required=True, help='输入图片路径')
    parser.add_argument('--output_name', type=str, default=None, help='输出名称 (可选)')
    
    args = parser.parse_args()
    
    # 处理图片
    result_path = process_new_image(args.image_path, args.output_name)
    
    if result_path:
        print(f"\n{'='*80}")
        print("")
        print(f"{'='*80}")
        print(f"SUCCESS: Final result saved: {result_path}")
        print("\n现在您可以查看生成的12层注意力网格图!")
    else:
        print(f"\n{'='*80}")
        print("")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
