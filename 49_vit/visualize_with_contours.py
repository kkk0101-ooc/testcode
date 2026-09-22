"""
优化版可视化 - 添加白色轮廓标记
在热力图上叠加目标的边缘轮廓
"""

import torch
import timm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import cv2
import matplotlib
matplotlib.use('Agg')

def detect_edges(image_array, low_threshold=50, high_threshold=150):
    """使用Canny边缘检测提取轮廓"""
    # 转换为灰度图
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Canny边缘检测
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # 膨胀操作，让边缘更明显
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges

def overlay_contours_on_heatmap(heatmap_array, edges, resize_shape=(14, 14)):
    """将边缘轮廓叠加到热力图上"""
    # 调整边缘图大小以匹配热力图
    edges_resized = cv2.resize(edges, resize_shape, interpolation=cv2.INTER_NEAREST)
    
    # 归一化边缘（0-1）
    edges_normalized = edges_resized.astype(float) / 255.0
    
    # 创建轮廓掩码（边缘处为1，其他为0）
    contour_mask = (edges_normalized > 0.3).astype(float)
    
    return contour_mask

def visualize_complete_with_contours(image_path, model, device, output_dir):
    """生成带轮廓的完整可视化"""
    
    image_path = Path(image_path)
    img_id = image_path.stem.split('_')[-1]
    
    print(f"\n处理图片: {img_id}")
    
    # 加载和预处理图片
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.230])
    ])
    
    original_image = Image.open(image_path).convert('RGB')
    
    # 同时准备224x224的numpy数组用于边缘检测
    original_224 = original_image.copy()
    original_224 = original_224.resize((224, 224), Image.LANCZOS)
    original_224_array = np.array(original_224)
    
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # 获取预测
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()
    
    print(f"  预测类别: {pred}")
    
    # 边缘检测
    print(f"  检测边缘...")
    edges = detect_edges(original_224_array, low_threshold=30, high_threshold=100)
    contour_mask = overlay_contours_on_heatmap(edges, edges, resize_shape=(14, 14))
    
    # 提取特征
    with torch.no_grad():
        features = model.forward_features(image_tensor)
    
    print(f"  提取特征: {features.shape}")
    
    # 获取最后一层的attention和V向量
    last_block = model.blocks[-1]
    with torch.no_grad():
        x_norm = last_block.norm1(features)
        B, N, C = x_norm.shape
        
        # Get Q, K, V
        qkv = last_block.attn.qkv(x_norm).reshape(B, N, 3, last_block.attn.num_heads, C // last_block.attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * last_block.attn.scale
        attn = attn.softmax(dim=-1)
        
        # Compute V magnitude
        v_mag = v.norm(dim=-1)
    
    print(f"  生成可视化...")
    
    # 创建图形
    fig = plt.figure(figsize=(24, 18), dpi=200)
    
    # 原图 (上方整行)
    ax_orig = plt.subplot2grid((3, 13), (0, 0), colspan=13, rowspan=1)
    ax_orig.imshow(original_image)
    ax_orig.set_title(f'Original Image (ID: {img_id}, Pred: {pred})', 
                     fontsize=16, fontweight='bold')
    ax_orig.axis('off')
    
    # V向量热力图 (第二行)
    for head_idx in range(12):
        ax = plt.subplot2grid((3, 13), (1, head_idx), colspan=1, rowspan=1)
        
        # V magnitude
        v_head = v_mag[0, head_idx, 1:].reshape(14, 14).cpu().numpy()
        
        im = ax.imshow(v_head, cmap='hot', interpolation='bilinear')
        
        # 叠加轮廓（白色）
        ax.contour(contour_mask, levels=[0.5], colors='white', linewidths=1.5, alpha=0.8)
        
        ax.set_title(f'H{head_idx}', fontsize=10)
        ax.axis('off')
        
        if head_idx == 11:
            cbar_ax = plt.subplot2grid((3, 13), (1, 12), colspan=1, rowspan=1)
            plt.colorbar(im, cax=cbar_ax, label='V-magnitude')
    
    # CLS attention (第三行) - 这里添加白色轮廓
    for head_idx in range(12):
        ax = plt.subplot2grid((3, 13), (2, head_idx), colspan=1, rowspan=1)
        
        # CLS attention
        cls_attn = attn[0, head_idx, 0, 1:].reshape(14, 14).cpu().numpy()
        
        im = ax.imshow(cls_attn, cmap='viridis', interpolation='bilinear')
        
        # 叠加轮廓（白色） - 这是关键！
        ax.contour(contour_mask, levels=[0.5], colors='white', linewidths=2.0, alpha=0.9)
        
        # 可选：添加散点标记在高attention区域
        high_attn_mask = (cls_attn > np.percentile(cls_attn, 80))
        high_attn_contour = high_attn_mask * contour_mask
        if high_attn_contour.sum() > 0:
            y_coords, x_coords = np.where(high_attn_contour > 0.5)
            ax.scatter(x_coords, y_coords, c='white', s=5, alpha=0.6, marker='o')
        
        ax.set_title(f'H{head_idx}', fontsize=10)
        ax.axis('off')
        
        if head_idx == 11:
            cbar_ax = plt.subplot2grid((3, 13), (2, 12), colspan=1, rowspan=1)
            plt.colorbar(im, cax=cbar_ax, label='Attention')
    
    plt.suptitle(f'NNS-ViT Enhanced Visualization with Contours: ImageNet {img_id}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / f"imagenet_{img_id}_combined_contours.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"  保存: {output_path}")
    
    return output_path

def main():
    """主函数"""
    
    print("="*80)
    print("优化版ImageNet可视化 - 添加轮廓")
    print("="*80)
    print()
    
    # 图片路径
    images = [
        "datas/ILSVRC2012_val/ILSVRC2012_val_00000441.JPEG",
        "datas/ILSVRC2012_val/ILSVRC2012_val_00006100.JPEG"
    ]
    
    # 输出目录
    output_dir = Path("ppt_results/imagenet_visualizations/combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 加载模型
    print("加载ViT-Base-16模型...")
    model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
    model.to(device)
    model.eval()
    print("  OK\n")
    
    # 处理每张图片
    results = []
    for img_path in images:
        if Path(img_path).exists():
            output_path = visualize_complete_with_contours(img_path, model, device, output_dir)
            results.append(output_path)
        else:
            print(f"错误: 图片不存在 - {img_path}")
    
    print("\n" + "="*80)
    print("完成!")
    print("="*80)
    print("\n生成的文件:")
    for path in results:
        print(f"  {path}")
    
    print("\n优化说明:")
    print("  - 在V向量热力图上添加了白色轮廓线")
    print("  - 在CLS Attention上添加了更粗的白色轮廓线")
    print("  - 在高attention区域添加了白色散点标记")
    print("  - 轮廓突出显示目标的身体边界")

if __name__ == "__main__":
    main()
