"""调试NNS问题 - 检查为什么没有生成scan_attn_map"""
import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.vit_final import VisionTransformer

def debug_nns_paths():
    """调试NNS路径问题"""
    
    # 模拟NNS_NONET的路径设置
    model_type = "VisionTransformer"
    dataset = "tiny200ILSVRC"
    model_name = "VisionTransformer_200_final_debug"
    img_name = "n01443537_24_debug"
    id = 0
    
    # 自动检测输出路径（复制NNS代码逻辑）
    possible_output_bases = [
        '/hy-tmp/1_vit_final/datas/scan_vit/',  # Linux路径
        os.path.join(os.path.expanduser('~'), 'scan_results', 'scan_vit'),  # Windows用户目录
        os.path.join('.', 'scan_results', 'scan_vit'),  # 当前目录
    ]
    output_base = None
    for base in possible_output_bases:
        if os.path.exists(os.path.dirname(base)) or base.startswith('.'):
            output_base = base
            break
    if output_base is None:
        output_base = os.path.join('.', 'scan_results', 'scan_vit')
        os.makedirs(output_base, exist_ok=True)
    
    rootdir = os.path.join(output_base, '114' + str(model_type) + '_' + str(dataset), str(model_name), str(id) + '_' + str(img_name))
    scan_attn_map = rootdir + 'scan_attn_map/'
    
    print(f"Debug NNS paths:")
    print(f"  output_base: {output_base}")
    print(f"  rootdir: {rootdir}")
    print(f"  scan_attn_map: {scan_attn_map}")
    
    # 创建目录结构
    os.makedirs(rootdir, exist_ok=True)
    os.makedirs(scan_attn_map, exist_ok=True)
    
    # 创建测试层目录
    test_layers = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
    for layer_num in test_layers:
        layer_dir = os.path.join(scan_attn_map, str(layer_num).rjust(2, '0'))
        input_attn_dir = os.path.join(layer_dir, 'input_attn', 'head0')
        os.makedirs(input_attn_dir, exist_ok=True)
        
        # 创建一个测试图像文件
        test_file = os.path.join(input_attn_dir, 'avg_cls_attn_head0.png')
        
        # 创建一个简单的测试图像
        from PIL import Image, ImageDraw
        import numpy as np
        
        # 创建32x32的测试热力图
        img_array = np.random.rand(32, 32) * 255
        img_array = img_array.astype(np.uint8)
        
        # 转换为热力图颜色
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # 使用热力图colormap
        colormap = cm.get_cmap('hot')
        colored_img = colormap(img_array / 255.0)
        colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
        
        # 保存图像
        pil_img = Image.fromarray(colored_img)
        pil_img = pil_img.resize((512, 512), Image.BICUBIC)  # 放大到512x512
        pil_img.save(test_file)
        
        print(f"  Created test layer {layer_num:02d}: {test_file}")
    
    print(f"\nTest structure created at: {rootdir}")
    return rootdir

def create_test_grid(result_dir):
    """使用测试数据创建网格图"""
    
    fish_image = r"d:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG"
    output_path = os.path.join(result_dir, "test_12layers_grid.png")
    
    # 查找scan_attn_map目录
    scan_attn_dir = result_dir + 'scan_attn_map/'
    
    if not os.path.exists(scan_attn_dir):
        print(f"ERROR: {scan_attn_dir} not found")
        return
    
    print(f"Using test data from: {scan_attn_dir}")
    
    # 查找所有层目录
    layer_dirs = []
    for item in os.listdir(scan_attn_dir):
        item_path = os.path.join(scan_attn_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            layer_dirs.append((int(item), item_path))
    layer_dirs.sort()
    
    print(f"Found {len(layer_dirs)} test layers")
    
    # 选择11层
    if len(layer_dirs) >= 11:
        step = len(layer_dirs) / 11
        selected_layers = [layer_dirs[int(i * step)] for i in range(11)]
    else:
        selected_layers = layer_dirs
    
    # 创建网格图
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25)
    
    # 原图
    ax = fig.add_subplot(gs[0, 0])
    if os.path.exists(fish_image):
        original_img = Image.open(fish_image).convert('RGB')
        ax.imshow(original_img)
        ax.set_title('Original Fish Image', fontsize=16, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center', fontsize=12)
        ax.set_title('Original Image', fontsize=16)
    ax.axis('off')
    
    # 层图像
    positions = [(0, 1), (0, 2), (0, 3),
                 (1, 0), (1, 1), (1, 2), (1, 3),
                 (2, 0), (2, 1), (2, 2), (2, 3)]
    
    for idx, (layer_num, layer_path) in enumerate(selected_layers):
        if idx >= len(positions):
            break
        
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        
        # 查找测试图像
        test_img_path = os.path.join(layer_path, 'input_attn', 'head0', 'avg_cls_attn_head0.png')
        
        if os.path.exists(test_img_path):
            img = Image.open(test_img_path)
            ax.imshow(img)
            ax.set_title(f'Test Layer {layer_num:02d}', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'Layer {layer_num}\nTest Failed', ha='center', va='center', fontsize=10)
            ax.set_title(f'Layer {layer_num:02d}', fontsize=14)
        ax.axis('off')
    
    plt.suptitle('ViT Depth-12 NNS Test Grid: Verifying 12-Layer Structure', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()
    
    print(f"Test grid saved: {output_path}")
    
    # 尝试打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(output_path)
            print("Test grid opened!")
        except:
            pass
    
    return output_path

if __name__ == "__main__":
    print("="*80)
    print("NNS DEBUG: Creating test 12-layer structure")
    print("="*80)
    
    # 创建测试结构
    test_dir = debug_nns_paths()
    
    # 创建测试网格图
    test_grid = create_test_grid(test_dir)
    
    print(f"\n{'='*80}")
    print("DEBUG COMPLETE!")
    print(f"{'='*80}")
    print(f"Test directory: {test_dir}")
    print(f"Test grid: {test_grid}")
    print("\nThis demonstrates the expected 12-layer structure.")
    print("Now we need to fix the actual NNS code to generate real attention maps.")
