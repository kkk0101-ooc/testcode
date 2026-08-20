"""
使用训练完成的depth=12模型生成可视化
"""
import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'model'))
sys.path.append(os.path.join(current_dir, 'function'))
sys.path.append(os.path.join(current_dir, 'NNS'))

from model.vit_final import VisionTransformer
from function.dataset32 import ScanDataSet
from NNS.Scannerpic1107_cls_nonet import NNS_NONET

# 配置
MODEL_PATH = r"D:\project\vit_114\NNS_model_and_dataset\models\vit_3channel_mutlihead\tiny200ILSVRC_VisionTransformer_32imgsize_10classes_0threshold_3heads_12depth_4patch_size_VisionTransformer_50_final.pth"
IMAGE_PATH = r"D:\project\vit_114\vitnns\49_vit\datas\tiny-imagenet-5-scanner\n01443537\n01443537_24.JPEG"
OUTPUT_DIR = r"D:\project\vit_114\vitnns\49_vit\training_visualizations"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 模型参数
DEPTH = 12
NUM_HEADS = 3
PATCH_SIZE = 4
CLASSES = 10
IMG_SIZE = 32

print(f"\n{'='*60}")
print("使用最终模型生成可视化")
print(f"{'='*60}")
print(f"模型: {MODEL_PATH}")
print(f"图像: {IMAGE_PATH}")
print(f"设备: {DEVICE}")
print(f"配置: depth={DEPTH}, heads={NUM_HEADS}, patch_size={PATCH_SIZE}")
print(f"{'='*60}\n")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
print("加载模型...")
model = VisionTransformer(
    img_size=[IMG_SIZE],  # 需要是列表
    patch_size=PATCH_SIZE,
    num_classes=CLASSES,
    embed_dim=192,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    mlp_ratio=4,
    qkv_bias=False
)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model = model.to(DEVICE)
model.eval()
print("✓ 模型加载成功")

# 加载图像
print(f"\n加载图像: {IMAGE_PATH}")
image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
print(f"图像名称: {image_name}")

# 直接加载图像并转换
from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

from PIL import Image
img = Image.open(IMAGE_PATH).convert('RGB')
image_tensor = transform(img).unsqueeze(0).to(DEVICE)  # (1, C, H, W)
print(f"✓ 图像加载成功，尺寸: {image_tensor.shape}")

found = True
if True:  # 直接处理
        
        # 运行NNS
        print("\n开始NNS处理...")
        params = {
            'cls_token': model.cls_token,
            'pos_embed': model.pos_embed
        }
        
        nns = NNS_NONET(
            network=model,
            imageTensor=image_tensor[0],
            img_name=f"{image_name}_final",
            id=0,  # 使用0作为索引
            device=DEVICE,
            model_name='VisionTransformer_final',
            model_type='VisionTransformer',
            dataset='tiny200ILSVRC',
            patch_size=PATCH_SIZE,
            num_heads=NUM_HEADS,
            depth=DEPTH,
            params=params
        )
        
        # 强制重新生成
        LN = True
        nns.get_scanpic(LN)
        print("✓ NNS处理完成")
        
        # 创建网格图像
        scan_attn_dir = os.path.join(nns.rootdir, 'scan_attn_map')
        output_path = os.path.join(OUTPUT_DIR, "final_model_grid_with_original.png")
        
        print(f"\n创建网格可视化...")
        print(f"扫描结果目录: {scan_attn_dir}")
        
        # 查找所有层目录
        layer_dirs = []
        if os.path.exists(scan_attn_dir):
            for item in os.listdir(scan_attn_dir):
                item_path = os.path.join(scan_attn_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    layer_dirs.append((int(item), item_path))
            layer_dirs.sort()
        
        print(f"找到 {len(layer_dirs)} 个层")
        
        if len(layer_dirs) == 0:
            print("错误：没有找到层目录")
            sys.exit(1)
        
        # 选择11层（因为原图占1格）
        if len(layer_dirs) > 11:
            indices = np.linspace(0, len(layer_dirs)-1, 11, dtype=int)
            selected_layers = [layer_dirs[i] for i in indices]
        else:
            selected_layers = layer_dirs
        
        print(f"选择 {len(selected_layers)} 层用于显示")
        
        # 创建 2x6 网格
        fig = plt.figure(figsize=(24, 8))
        gs = GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.25)
        
        # 添加标题
        fig.suptitle('Final Model (Epoch 50) - Vision Transformer Attention Maps (Depth=12)', 
                     fontsize=16, fontweight='bold')
        
        # 第一个位置：原图
        print("\n添加原图...")
        ax = fig.add_subplot(gs[0, 0])
        if os.path.exists(IMAGE_PATH):
            original_img = Image.open(IMAGE_PATH).convert('RGB')
            ax.imshow(original_img)
            ax.set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
            print("  ✓ 原图添加成功")
        else:
            ax.text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center')
            ax.set_title('Original Image', fontsize=14, pad=10)
            print("  ⚠️  原图未找到")
        ax.axis('off')
        
        # 其余位置：层的注意力图
        positions = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                     (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
        
        print("\n添加层注意力图...")
        for idx, (layer_num, layer_path) in enumerate(selected_layers):
            if idx >= len(positions):
                break
            
            row, col = positions[idx]
            ax = fig.add_subplot(gs[row, col])
            
            # 查找注意力图
            possible_paths = [
                os.path.join(layer_path, 'input_attn', 'avg_cls_attn_head0.png'),
                os.path.join(layer_path, 'avg_cls_attn_head0.png'),
                os.path.join(layer_path, 'input_attn', 'avg_cls_attn.png'),
            ]
            
            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    break
            
            if img_path:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f'Layer {layer_num:02d}', fontsize=13, fontweight='bold', pad=8)
                print(f"  ✓ Layer {layer_num:02d} 添加成功")
            else:
                ax.text(0.5, 0.5, f'Layer {layer_num}\nNot Found', 
                       ha='center', va='center', fontsize=10)
                ax.set_title(f'Layer {layer_num:02d}', fontsize=13, pad=8)
                print(f"  ⚠️  Layer {layer_num:02d} 图像未找到")
            ax.axis('off')
        
        print(f"\n保存网格图像: {output_path}")
        plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
        plt.close()
        
        print(f"\n✅ 完成！")
        print(f"  网格图像: {output_path}")
        print(f"  布局: 2行×6列")
        print(f"  包含: 原图 + {len(selected_layers)} 层注意力图")

