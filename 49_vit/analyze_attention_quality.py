"""分析注意力质量并提供优化建议"""
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def analyze_attention_maps(result_dir, image_name):
    """分析注意力图的质量"""
    
    scan_attn_dir = os.path.join(result_dir, 'scan_attn_map')
    if not os.path.exists(scan_attn_dir):
        print(f"ERROR: {scan_attn_dir} not found")
        return
    
    print(f"分析注意力图质量: {image_name}")
    print("="*60)
    
    # 查找所有层
    layer_dirs = []
    for item in os.listdir(scan_attn_dir):
        item_path = os.path.join(scan_attn_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            layer_dirs.append((int(item), item_path))
    layer_dirs.sort()
    
    print(f"找到 {len(layer_dirs)} 个注意力层")
    
    # 分析每一层的注意力分布
    layer_stats = []
    
    for layer_num, layer_path in layer_dirs:
        # 查找注意力图文件
        possible_paths = [
            os.path.join(layer_path, 'input_attn', 'head0', 'avg_cls_attn_head0.png'),
            os.path.join(layer_path, 'input_attn', 'head1', 'avg_cls_attn_head1.png'),
            os.path.join(layer_path, 'input_attn', 'head2', 'avg_cls_attn_head2.png'),
        ]
        
        head_stats = []
        for head_idx, path in enumerate(possible_paths):
            if os.path.exists(path):
                # 加载图像并分析
                img = Image.open(path)
                img_array = np.array(img)
                
                # 转换为灰度值分析对比度
                if len(img_array.shape) == 3:
                    gray = np.mean(img_array, axis=2)
                else:
                    gray = img_array
                
                # 计算统计信息
                mean_val = np.mean(gray)
                std_val = np.std(gray)
                min_val = np.min(gray)
                max_val = np.max(gray)
                contrast = (max_val - min_val) / (max_val + min_val + 1e-8)
                
                head_stats.append({
                    'head': head_idx,
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'contrast': contrast,
                    'path': path
                })
        
        if head_stats:
            # 选择对比度最高的head
            best_head = max(head_stats, key=lambda x: x['contrast'])
            layer_stats.append({
                'layer': layer_num,
                'best_head': best_head,
                'all_heads': head_stats
            })
            
            print(f"Layer {layer_num:02d}: 最佳Head{best_head['head']} - 对比度:{best_head['contrast']:.3f}, 标准差:{best_head['std']:.1f}")
    
    return layer_stats

def create_enhanced_attention_map(tensor_path, output_path, enhancement_method='percentile'):
    """创建增强的注意力图"""
    
    if not os.path.exists(tensor_path):
        print(f"Tensor文件不存在: {tensor_path}")
        return False
    
    # 加载注意力tensor
    attn_tensor = torch.load(tensor_path, map_location='cpu')
    print(f"注意力tensor形状: {attn_tensor.shape}")
    
    # 提取CLS token的注意力 (假设第一个token是CLS)
    if len(attn_tensor.shape) == 4:  # (batch, heads, seq_len, seq_len)
        cls_attn = attn_tensor[0, :, 0, 1:]  # 所有head的CLS注意力，排除CLS自己
        # 平均所有head
        avg_cls_attn = torch.mean(cls_attn, dim=0)  # (seq_len-1,)
    elif len(attn_tensor.shape) == 3:  # (heads, seq_len, seq_len)
        cls_attn = attn_tensor[:, 0, 1:]  # 所有head的CLS注意力
        avg_cls_attn = torch.mean(cls_attn, dim=0)  # (seq_len-1,)
    else:
        print(f"未知的tensor形状: {attn_tensor.shape}")
        return False
    
    # 转换为numpy
    attn_np = avg_cls_attn.detach().cpu().numpy()
    
    # 计算patch数量 (假设是正方形)
    num_patches = len(attn_np)
    patch_size = int(np.sqrt(num_patches))
    
    if patch_size * patch_size != num_patches:
        print(f"注意力长度 {num_patches} 不是完全平方数")
        return False
    
    # 重塑为2D
    attn_2d = attn_np.reshape(patch_size, patch_size)
    
    # 应用不同的增强方法
    if enhancement_method == 'percentile':
        # 百分位数裁剪增强对比度
        p_low = np.percentile(attn_2d, 5)
        p_high = np.percentile(attn_2d, 95)
        attn_enhanced = np.clip(attn_2d, p_low, p_high)
        attn_enhanced = (attn_enhanced - p_low) / (p_high - p_low + 1e-8)
        
    elif enhancement_method == 'sigmoid':
        # Sigmoid增强
        attn_mean = np.mean(attn_2d)
        attn_enhanced = 1 / (1 + np.exp(-10 * (attn_2d - attn_mean)))
        
    elif enhancement_method == 'power':
        # 幂函数增强
        attn_normalized = (attn_2d - np.min(attn_2d)) / (np.max(attn_2d) - np.min(attn_2d) + 1e-8)
        attn_enhanced = np.power(attn_normalized, 0.5)  # 平方根增强
        
    elif enhancement_method == 'adaptive':
        # 自适应增强
        attn_std = np.std(attn_2d)
        if attn_std < 0.1:  # 低对比度
            # 使用更激进的增强
            p_low = np.percentile(attn_2d, 2)
            p_high = np.percentile(attn_2d, 98)
            attn_enhanced = np.clip(attn_2d, p_low, p_high)
            attn_enhanced = (attn_enhanced - p_low) / (p_high - p_low + 1e-8)
            # 再应用幂函数
            attn_enhanced = np.power(attn_enhanced, 0.3)
        else:
            # 标准增强
            p_low = np.percentile(attn_2d, 10)
            p_high = np.percentile(attn_2d, 90)
            attn_enhanced = np.clip(attn_2d, p_low, p_high)
            attn_enhanced = (attn_enhanced - p_low) / (p_high - p_low + 1e-8)
    
    # 上采样到更高分辨率
    from scipy.ndimage import zoom
    upscale_factor = 512 // patch_size  # 上采样到512x512
    attn_upscaled = zoom(attn_enhanced, upscale_factor, order=3)  # 三次插值
    
    # 创建热力图
    plt.figure(figsize=(8, 8))
    plt.imshow(attn_upscaled, cmap='hot', interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()
    
    print(f"增强注意力图已保存: {output_path}")
    print(f"  原始对比度: {np.std(attn_2d):.4f}")
    print(f"  增强对比度: {np.std(attn_enhanced):.4f}")
    print(f"  分辨率: {attn_upscaled.shape}")
    
    return True

def optimize_final_layer_attention(result_dir):
    """优化最后一层的注意力可视化"""
    
    print("\n" + "="*60)
    print("优化最后一层注意力可视化")
    print("="*60)
    
    scan_attn_dir = os.path.join(result_dir, 'scan_attn_map')
    
    # 找到最后一层
    layer_dirs = []
    for item in os.listdir(scan_attn_dir):
        item_path = os.path.join(scan_attn_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            layer_dirs.append((int(item), item_path))
    
    if not layer_dirs:
        print("没有找到注意力层")
        return
    
    layer_dirs.sort()
    final_layer_num, final_layer_path = layer_dirs[-1]
    
    print(f"最后一层: Layer {final_layer_num}")
    
    # 查找tensor文件
    tensor_files = []
    for root, dirs, files in os.walk(final_layer_path):
        for file in files:
            if file.endswith('.pt'):
                tensor_files.append(os.path.join(root, file))
    
    print(f"找到 {len(tensor_files)} 个tensor文件")
    
    # 为每个tensor创建增强版本
    enhancement_methods = ['percentile', 'sigmoid', 'power', 'adaptive']
    
    for tensor_file in tensor_files:
        tensor_name = os.path.basename(tensor_file).replace('.pt', '')
        print(f"\n处理tensor: {tensor_name}")
        
        for method in enhancement_methods:
            output_name = f"{tensor_name}_enhanced_{method}.png"
            output_path = os.path.join(final_layer_path, output_name)
            
            success = create_enhanced_attention_map(tensor_file, output_path, method)
            if success:
                print(f"  SUCCESS: {method} enhancement completed")

def create_multi_head_comparison(result_dir, layer_num=46):
    """创建多头注意力对比图"""
    
    print(f"\n创建Layer {layer_num}的多头注意力对比")
    
    layer_path = os.path.join(result_dir, 'scan_attn_map', f'{layer_num:02d}')
    if not os.path.exists(layer_path):
        print(f"Layer {layer_num} 不存在")
        return
    
    # 查找所有head的注意力图
    head_images = []
    for head_idx in range(3):  # 假设有3个head
        head_path = os.path.join(layer_path, 'input_attn', f'head{head_idx}', f'avg_cls_attn_head{head_idx}.png')
        if os.path.exists(head_path):
            head_images.append((head_idx, head_path))
    
    if len(head_images) < 2:
        print("没有足够的head图像进行对比")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(1, len(head_images), figsize=(6*len(head_images), 6))
    if len(head_images) == 1:
        axes = [axes]
    
    for idx, (head_idx, img_path) in enumerate(head_images):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f'Head {head_idx}', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle(f'Layer {layer_num} Multi-Head Attention Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(result_dir, f'layer_{layer_num}_multihead_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()
    
    print(f"多头对比图已保存: {output_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(output_path)
            print("多头对比图已自动打开")
        except:
            pass

if __name__ == "__main__":
    # 分析两个图像的注意力质量
    
    print("开始注意力质量分析和优化...")
    
    # 鱼图像
    fish_result_dir = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final_fixed\0_n01443537_24_fixed"
    
    # 新图像  
    new_result_dir = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final_new\0_n02085620_1_new"
    
    # 分析鱼图像
    if os.path.exists(fish_result_dir):
        print("\n1. 分析鱼图像注意力质量")
        fish_stats = analyze_attention_maps(fish_result_dir, "鱼图像")
        
        print("\n2. 优化鱼图像最后一层")
        optimize_final_layer_attention(fish_result_dir)
        
        print("\n3. 创建鱼图像多头对比")
        create_multi_head_comparison(fish_result_dir, 46)
    
    # 分析新图像
    if os.path.exists(new_result_dir):
        print("\n4. 分析新图像注意力质量")
        new_stats = analyze_attention_maps(new_result_dir, "新图像")
        
        print("\n5. 优化新图像最后一层")
        optimize_final_layer_attention(new_result_dir)
        
        print("\n6. 创建新图像多头对比")
        create_multi_head_comparison(new_result_dir, 46)
    
    print("\n" + "="*80)
    print("注意力分析和优化完成！")
    print("="*80)
    print("\n建议:")
    print("1. 查看多头对比图，选择对比度最好的head")
    print("2. 查看增强版本的注意力图，选择最清晰的增强方法")
    print("3. 可以基于最佳head和增强方法重新生成网格图")
    print("4. 考虑使用更大的patch size或更小的图像尺寸来提高分辨率")
