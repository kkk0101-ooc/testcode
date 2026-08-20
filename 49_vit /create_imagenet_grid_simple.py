"""简化的ImageNet网格图生成"""
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_imagenet_grid():
    """创建ImageNet 10层网格图"""
    print("Creating ImageNet 10-layer grid...")
    
    # 基础路径
    base_path = r".\scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained"
    
    # 查找第一个有效的图像目录
    image_dirs = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d)) and 'scan_attn_map' in d]
    
    if not image_dirs:
        print("ERROR: No scan directories found")
        return None
    
    # 使用第一个目录
    scan_dir = os.path.join(base_path, image_dirs[0], 'scan_attn_map')
    print(f"Using scan directory: {scan_dir}")
    
    # 查找层目录
    if not os.path.exists(scan_dir):
        print("ERROR: Scan directory does not exist")
        return None
    
    layer_dirs = [d for d in os.listdir(scan_dir) 
                 if os.path.isdir(os.path.join(scan_dir, d)) and d.isdigit()]
    layer_dirs.sort(key=int)
    
    print(f"Found {len(layer_dirs)} layers: {layer_dirs}")
    
    if len(layer_dirs) < 10:
        print(f"WARNING: Only {len(layer_dirs)} layers found, need at least 10")
        return None
    
    # 选择10个层
    selected_layers = layer_dirs[:10] if len(layer_dirs) >= 10 else layer_dirs
    
    # 创建2x5网格
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('ImageNet ViT-base-16: 10-Layer Attention Evolution\n(Blue-White Colormap)', 
                 fontsize=18, fontweight='bold')
    
    for idx, layer in enumerate(selected_layers):
        row = idx // 5
        col = idx % 5
        
        # 查找最佳注意力图
        layer_path = os.path.join(scan_dir, layer, 'input_attn')
        
        best_img_path = None
        best_contrast = 0
        
        # 尝试不同的头
        for head_idx in range(12):  # ViT-base-16有12个头
            head_dir = os.path.join(layer_path, f'head{head_idx}')
            if os.path.exists(head_dir):
                avg_cls_path = os.path.join(head_dir, f'avg_cls_attn_head{head_idx}.png')
                if os.path.exists(avg_cls_path):
                    try:
                        # 计算对比度
                        img = Image.open(avg_cls_path)
                        img_array = np.array(img.convert('L'))
                        contrast = np.std(img_array)
                        
                        if contrast > best_contrast:
                            best_contrast = contrast
                            best_img_path = avg_cls_path
                    except:
                        continue
        
        # 显示图像
        if best_img_path:
            img = Image.open(best_img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Layer {layer}\n(Contrast: {best_contrast:.1f})', 
                                   fontsize=12, fontweight='bold')
            print(f"  Layer {layer}: SUCCESS (contrast: {best_contrast:.1f})")
        else:
            axes[row, col].text(0.5, 0.5, f'Layer {layer}\nNot Available', 
                              ha='center', va='center')
            axes[row, col].set_title(f'Layer {layer}', fontsize=12)
            print(f"  Layer {layer}: FAILED")
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存网格图
    output_path = os.path.join(base_path, "imagenet_vitbase16_10layers_grid.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: Grid saved to {output_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(output_path)
            print("SUCCESS: Grid opened automatically")
        except:
            pass
    
    return output_path

def main():
    """主函数"""
    print("="*80)
    print("ImageNet ViT-base-16 Grid Generation")
    print("="*80)
    
    grid_path = create_imagenet_grid()
    
    if grid_path:
        print("\n" + "="*80)
        print("SUCCESS: ImageNet grid generation completed!")
        print("="*80)
        print(f"Grid file: {grid_path}")
        print("This shows the attention evolution across 10 layers")
        print("of the ViT-base-16 model on ImageNet data")
    else:
        print("\n" + "="*80)
        print("ERROR: Grid generation failed")
        print("="*80)

if __name__ == "__main__":
    main()
