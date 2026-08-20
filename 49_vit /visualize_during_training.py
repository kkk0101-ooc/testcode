"""
训练过程中自动生成可视化
每个epoch结束后检查新模型并生成12层注意力图+原图的网格
"""
import os
import sys
import time
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

# 使用与train_scanner_cls.py相同的导入方式
from model.vit_final import VisionTransformer
from function.dataset32 import ScanDataSet
from NNS.Scannerpic1107_cls_nonet import NNS_NONET

# 配置
MODEL_DIR = r"D:\project\vit_114\NNS_model_and_dataset\models\vit_3channel_mutlihead\tiny200ILSVRC_VisionTransformer_32imgsize_10classes_0threshold_3heads_12depth_4patch_size_"
IMAGE_PATH = r"D:\project\vit_114\vitnns\49_vit\datas\tiny-imagenet-5-scanner\n01443537\n01443537_24.JPEG"
OUTPUT_DIR = r"D:\project\vit_114\vitnns\49_vit\training_visualizations"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 模型参数
DEPTH = 12
NUM_HEADS = 3
PATCH_SIZE = 4
CLASSES = 10
IMG_SIZE = 32

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 记录已处理的模型
processed_models = set()

def create_visualization(model_path, epoch_num):
    """为单个模型创建可视化"""
    print(f"\n{'='*60}")
    print(f"处理模型: {os.path.basename(model_path)}")
    print(f"Epoch: {epoch_num}")
    print(f"{'='*60}")
    
    try:
        # 加载模型
        model = VisionTransformer(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            num_classes=CLASSES,
            embed_dim=192,
            depth=DEPTH,
            num_heads=NUM_HEADS
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(DEVICE)
        model.eval()
        print("✓ 模型加载成功")
        
        # 加载图像
        dataset = ScanDataSet(os.path.dirname(IMAGE_PATH))
        image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        
        # 找到目标图像
        for idx, (img_path, label) in enumerate(dataset.imgs):
            current_name = os.path.splitext(os.path.basename(img_path))[0]
            if current_name == image_name:
                # 加载并转换图像
                image_tensor = dataset[idx][0]  # (C, H, W)
                image_tensor = image_tensor.unsqueeze(0).to(DEVICE)  # (1, C, H, W)
                
                # 运行NNS
                print("开始NNS处理...")
                params = {
                    'cls_token': model.cls_token,
                    'pos_embed': model.pos_embed
                }
                
                nns = NNS_NONET(
                    model=model,
                    imageTensor=image_tensor[0],
                    img_name=f"{image_name}_epoch{epoch_num}",
                    name_i=idx,
                    device=DEVICE,
                    model_name=f'VisionTransformer_epoch{epoch_num}',
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
                output_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch_num:03d}_grid.png")
                
                create_grid(scan_attn_dir, IMAGE_PATH, output_path, epoch_num)
                
                print(f"✓ 可视化完成: {output_path}")
                break
        
        # 清理显存
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()

def create_grid(scan_result_dir, original_image_path, output_path, epoch_num):
    """创建包含原图的12层网格"""
    # 查找所有层目录
    layer_dirs = []
    if os.path.exists(scan_result_dir):
        for item in os.listdir(scan_result_dir):
            item_path = os.path.join(scan_result_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                layer_dirs.append((int(item), item_path))
        layer_dirs.sort()
    
    if len(layer_dirs) == 0:
        print("  ⚠️  没有找到层目录")
        return
    
    # 选择11层（因为原图占1格）
    if len(layer_dirs) > 11:
        indices = np.linspace(0, len(layer_dirs)-1, 11, dtype=int)
        selected_layers = [layer_dirs[i] for i in indices]
    else:
        selected_layers = layer_dirs
    
    # 创建 2x6 网格
    fig = plt.figure(figsize=(24, 8))
    gs = GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.25)
    
    # 添加标题
    fig.suptitle(f'Epoch {epoch_num} - Vision Transformer Attention Maps (Depth=12)', 
                 fontsize=16, fontweight='bold')
    
    # 第一个位置：原图
    ax = fig.add_subplot(gs[0, 0])
    if os.path.exists(original_image_path):
        original_img = Image.open(original_image_path).convert('RGB')
        ax.imshow(original_img)
        ax.set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
    else:
        ax.text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center')
        ax.set_title('Original Image', fontsize=14, pad=10)
    ax.axis('off')
    
    # 其余位置：层的注意力图
    positions = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                 (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    
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
        else:
            ax.text(0.5, 0.5, f'Layer {layer_num}\nProcessing...', 
                   ha='center', va='center', fontsize=10)
            ax.set_title(f'Layer {layer_num:02d}', fontsize=13, pad=8)
        ax.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()

def monitor_training():
    """监控训练过程并生成可视化"""
    print(f"\n{'='*60}")
    print("开始监控训练...")
    print(f"模型目录: {MODEL_DIR}")
    print(f"图像: {IMAGE_PATH}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    check_interval = 30  # 每30秒检查一次
    
    while True:
        try:
            # 查找所有模型文件
            if os.path.exists(MODEL_DIR):
                all_files = os.listdir(MODEL_DIR)
                model_files = [f for f in all_files if f.endswith('.pth')]
                
                if len(model_files) == 0:
                    print(f"  [DEBUG] 目录存在但未找到.pth文件")
                    print(f"  [DEBUG] 目录路径: {MODEL_DIR}")
                    print(f"  [DEBUG] 目录内容: {all_files[:10] if len(all_files) > 0 else '空目录'}")
                else:
                    print(f"  [DEBUG] 找到 {len(model_files)} 个.pth文件")
                
                model_files = sorted(
                    model_files,
                    key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x))
                )
                
                # 处理新模型
                new_count = 0
                for model_file in model_files:
                    if model_file not in processed_models:
                        model_path = os.path.join(MODEL_DIR, model_file)
                        print(f"  [INFO] 发现新模型: {model_file}")
                        
                        # 从文件名提取epoch
                        try:
                            # 格式: VisionTransformer_2_19.4.pth
                            parts = model_file.replace('.pth', '').split('_')
                            epoch_num = int(parts[1]) - 1  # epoch从0开始
                        except:
                            epoch_num = len(processed_models)
                        
                        create_visualization(model_path, epoch_num)
                        processed_models.add(model_file)
                        new_count += 1
                
                # 显示进度
                current_time = time.strftime('%H:%M:%S')
                if new_count > 0:
                    print(f"[{current_time}] ✓ 处理了 {new_count} 个新模型，总计: {len(processed_models)}")
                else:
                    print(f"[{current_time}] 已处理模型数: {len(processed_models)} (总文件数: {len(model_files)})")
            else:
                print(f"  [ERROR] 模型目录不存在: {MODEL_DIR}")
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(check_interval)

if __name__ == '__main__':
    print("训练可视化监控脚本")
    print("按 Ctrl+C 停止监控\n")
    monitor_training()

