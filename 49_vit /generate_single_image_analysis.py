"""
为单张图片生成完整的ViT分析
支持自定义图片路径和输出目录
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import label
import cv2
import argparse

# 导入本地模型
from load_local_model import load_vit_model

def generate_analysis(image_path, output_dir, image_name=None):
    """
    为单张图片生成完整分析
    
    参数:
        image_path: 图片路径
        output_dir: 输出目录
        image_name: 可选的图片名称（用于标题）
    """
    
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    
    if not image_path.exists():
        print(f"错误: 图片不存在 {image_path}")
        return False
    
    if image_name is None:
        image_name = image_path.stem
    
    print("="*70)
    print(f" 为图片生成ViT完整分析: {image_name}")
    print("="*70)
    
    # 创建目录结构
    print("\n创建目录结构...")
    subdirs_attention = [
        "visualize_attention/layer_progression",
        "visualize_attention/head_specialization",
        "visualize_attention/hotspot_identification",
        "visualize_attention/attention_evolution",
        "visualize_attention/spatial_distribution",
        "visualize_attention/quantitative"
    ]
    
    subdirs_v = [
        "visualize_v/layer_progression",
        "visualize_v/head_specialization",
        "visualize_v/value_evolution",
        "visualize_v/quantitative"
    ]
    
    for subdir in subdirs_attention + subdirs_v:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"[OK] 已创建输出目录: {output_dir}")
    
    # 加载模型
    print("\n加载模型...")
    model, device = load_vit_model('cuda')
    print(f"[OK] 模型已加载到: {device}")
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 提取数据
    print("\n提取attention和V向量...")
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    attention_maps = []
    v_vectors = []
    
    with torch.no_grad():
        x = model.patch_embed(img_tensor)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        
        for block in model.blocks:
            x_norm = block.norm1(x)
            B, N, C = x_norm.shape
            
            qkv = block.attn.qkv(x_norm).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            
            attention_maps.append(attn.cpu())
            v_vectors.append(v.cpu())
            
            x = block(x)
        
        # 预测
        x = model.norm(x)
        out = model.head(x[:, 0])
        pred_class = out.argmax(1).item()
    
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].shape[1]
    
    print(f"[OK] 已提取 {num_layers}层, {num_heads}个头的数据")
    
    # ============================================================
    # PART 1: visualize_attention
    # ============================================================
    print("\n" + "="*70)
    print(" PART 1: 生成visualize_attention")
    print("="*70)
    
    # 1. Layer Progression
    print("\n1. Layer Progression...")
    for head_idx in [0, 5, 11]:
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        fig.suptitle(f'{image_name} - Attention Head {head_idx} - Layer Progression', 
                     fontsize=16, fontweight='bold')
        
        for layer_idx in range(num_layers):
            ax = axes[layer_idx // 6, layer_idx % 6]
            
            attn = attention_maps[layer_idx]
            cls_attn = attn[0, head_idx, 0, 1:].numpy().reshape(14, 14)
            
            im = ax.imshow(cls_attn, cmap='hot', interpolation='bilinear')
            ax.set_title(f'Layer {layer_idx}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"visualize_attention/layer_progression/head{head_idx}_progression.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Head Specialization
    print("2. Head Specialization...")
    representative_layers = [0, 2, 4, 6, 8, 10]
    
    for layer_idx in representative_layers:
        # All heads
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'{image_name} - Layer {layer_idx} - All Heads', fontsize=16)
        
        attn = attention_maps[layer_idx]
        cls_attn = attn[0, :, 0, 1:].numpy()
        
        for h in range(num_heads):
            ax = axes[h//4, h%4]
            attn_map = cls_attn[h].reshape(14, 14)
            im = ax.imshow(attn_map, cmap='viridis')
            ax.set_title(f'Head {h}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"visualize_attention/head_specialization/layer{layer_idx:02d}_heads.png", 
                    dpi=150)
        plt.close()
        
        # Similarity
        sim = cosine_similarity(cls_attn)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax.set_title(f'{image_name} - Layer {layer_idx} - Head Similarity', fontsize=14)
        plt.colorbar(im, ax=ax)
        
        for i in range(num_heads):
            for j in range(num_heads):
                ax.text(j, i, f'{sim[i,j]:.2f}', ha="center", va="center", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"visualize_attention/head_specialization/layer{layer_idx:02d}_similarity.png", 
                    dpi=150)
        plt.close()
    
    # 3. Hotspot Identification
    print("3. Hotspot Identification...")
    for layer_idx in representative_layers:
        for head_idx in [0, 5, 11]:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            attn = attention_maps[layer_idx]
            cls_attn = attn[0, head_idx, 0, 1:].numpy().reshape(14, 14)
            
            threshold = cls_attn.mean() + cls_attn.std()
            hotspots = cls_attn > threshold
            
            im = ax.imshow(cls_attn, cmap='hot', interpolation='bilinear')
            
            if hotspots.any():
                contours, _ = cv2.findContours((hotspots*255).astype(np.uint8), 
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) > 0:
                        cnt = cnt.squeeze()
                        if cnt.ndim == 2:
                            ax.plot(cnt[:, 0], cnt[:, 1], 'g-', linewidth=2)
            
            ax.set_title(f'{image_name} - Layer {layer_idx} Head {head_idx} - Hotspots', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"visualize_attention/hotspot_identification/layer{layer_idx:02d}_head{head_idx}_hotspots.png", 
                        dpi=150)
            plt.close()
    
    # 4. Attention Evolution
    print("4. Attention Evolution...")
    
    # Evolution across layers
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'{image_name} - Attention Evolution Across Layers', fontsize=16)
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx//4, layer_idx%4]
        
        attn = attention_maps[layer_idx]
        cls_attn = attn[0, :, 0, 1:].mean(0).numpy().reshape(14, 14)
        
        im = ax.imshow(cls_attn, cmap='hot', interpolation='bilinear')
        ax.set_title(f'Layer {layer_idx}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / "visualize_attention/attention_evolution/layer_evolution.png", dpi=150)
    plt.close()
    
    # 5. Spatial Distribution
    print("5. Spatial Distribution...")
    for layer_idx in [0, 6, 11]:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        attn = attention_maps[layer_idx]
        cls_attn = attn[0, :, 0, 1:].mean(0).numpy().reshape(14, 14)
        
        im = ax.imshow(cls_attn, cmap='hot', interpolation='bilinear')
        ax.set_title(f'{image_name} - Layer {layer_idx} - Spatial Distribution', fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"visualize_attention/spatial_distribution/layer{layer_idx:02d}_spatial.png", 
                    dpi=150)
        plt.close()
    
    # 6. Quantitative
    print("6. Quantitative Analysis...")
    
    stats = {
        'entropy': [],
        'concentration': [],
        'sparsity': []
    }
    
    for layer_idx in range(num_layers):
        attn = attention_maps[layer_idx]
        cls_attn = attn[0, :, 0, 1:].mean(0).numpy()
        cls_attn_norm = cls_attn / cls_attn.sum()
        
        ent = -np.sum(cls_attn_norm * np.log(cls_attn_norm + 1e-10))
        conc = cls_attn_norm.std()
        spar = (cls_attn_norm < 0.01).sum() / len(cls_attn_norm)
        
        stats['entropy'].append(float(ent))
        stats['concentration'].append(float(conc))
        stats['sparsity'].append(float(spar))
    
    # Save stats
    import json
    with open(output_dir / "visualize_attention/quantitative/statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Plot stats
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(stats['entropy'], 'o-', linewidth=2)
    axes[0].set_title('Entropy Evolution')
    axes[0].set_xlabel('Layer')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(stats['concentration'], 'o-', color='orange', linewidth=2)
    axes[1].set_title('Concentration Evolution')
    axes[1].set_xlabel('Layer')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(stats['sparsity'], 'o-', color='green', linewidth=2)
    axes[2].set_title('Sparsity Evolution')
    axes[2].set_xlabel('Layer')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "visualize_attention/quantitative/statistics_summary.png", dpi=150)
    plt.close()
    
    print("[OK] visualize_attention完成")
    
    # ============================================================
    # PART 2: visualize_v
    # ============================================================
    print("\n" + "="*70)
    print(" PART 2: 生成visualize_v")
    print("="*70)
    
    # 1. V Vector Layer Progression
    print("\n1. V Vector Layer Progression...")
    for head_idx in [0, 5, 11]:
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        fig.suptitle(f'{image_name} - V Vector Head {head_idx} - Layer Progression', 
                     fontsize=16, fontweight='bold')
        
        for layer_idx in range(num_layers):
            ax = axes[layer_idx // 6, layer_idx % 6]
            
            v = v_vectors[layer_idx]
            v_intensity = v[0, head_idx, 1:, :].norm(dim=-1).numpy().reshape(14, 14)
            
            im = ax.imshow(v_intensity, cmap='viridis', interpolation='bilinear')
            ax.set_title(f'Layer {layer_idx}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"visualize_v/layer_progression/head{head_idx}_progression.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. V Vector Evolution
    print("2. V Vector Evolution...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'{image_name} - V Vector Evolution Across Layers', fontsize=16)
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx//4, layer_idx%4]
        
        v = v_vectors[layer_idx]
        v_intensity = v[0, :, 1:, :].norm(dim=-1).mean(0).numpy().reshape(14, 14)
        
        im = ax.imshow(v_intensity, cmap='viridis', interpolation='bilinear')
        ax.set_title(f'Layer {layer_idx}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / "visualize_v/value_evolution/layer_evolution.png", dpi=150)
    plt.close()
    
    print("[OK] visualize_v完成")
    
    # ============================================================
    # 生成HTML报告
    # ============================================================
    print("\n生成HTML报告...")
    
    # Attention HTML
    html_attention = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ViT Attention Analysis - {image_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top: 30px; }}
        img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; border-radius: 5px; }}
        .section {{ margin-bottom: 40px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
        .info {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ViT Attention Analysis Report</h1>
        
        <div class="info">
            <p><strong>Image:</strong> {image_name}</p>
            <p><strong>Model:</strong> ViT-Base-16</p>
            <p><strong>Layers:</strong> {num_layers}</p>
            <p><strong>Heads:</strong> {num_heads}</p>
            <p><strong>Predicted Class:</strong> {pred_class}</p>
        </div>
        
        <div class="section">
            <h2>层级进展分析</h2>
            <p>展示每个attention head在12层中的演化模式</p>
            <div class="image-grid">
                <img src="layer_progression/head0_progression.png" />
                <img src="layer_progression/head5_progression.png" />
                <img src="layer_progression/head11_progression.png" />
            </div>
        </div>
        
        <div class="section">
            <h2>注意力头专业化</h2>
            <p>分析同一层中不同head的专业化特征</p>
            <div class="image-grid">
"""
    
    for layer in representative_layers:
        html_attention += f'                <img src="head_specialization/layer{layer:02d}_heads.png" />\n'
        html_attention += f'                <img src="head_specialization/layer{layer:02d}_similarity.png" />\n'
    
    html_attention += """
            </div>
        </div>
        
        <div class="section">
            <h2>注意力热点识别</h2>
            <p>识别高亮区域，绿色轮廓标记热点</p>
            <div class="image-grid">
"""
    
    for layer in representative_layers:
        for head in [0, 5, 11]:
            html_attention += f'                <img src="hotspot_identification/layer{layer:02d}_head{head}_hotspots.png" />\n'
    
    html_attention += f"""
            </div>
        </div>
        
        <div class="section">
            <h2>注意力演化</h2>
            <p>展示attention随层数的演化</p>
            <div class="image-grid">
                <img src="attention_evolution/layer_evolution.png" />
            </div>
        </div>
        
        <div class="section">
            <h2>空间分布</h2>
            <p>不同层的平均attention空间分布</p>
            <div class="image-grid">
                <img src="spatial_distribution/layer00_spatial.png" />
                <img src="spatial_distribution/layer06_spatial.png" />
                <img src="spatial_distribution/layer11_spatial.png" />
            </div>
        </div>
        
        <div class="section">
            <h2>量化分析</h2>
            <p>统计指标的演化</p>
            <div class="image-grid">
                <img src="quantitative/statistics_summary.png" />
            </div>
        </div>
        
        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #999;">
            Generated by ViT Single Image Analyzer | {image_name}
        </p>
    </div>
</body>
</html>
"""
    
    with open(output_dir / "visualize_attention/analysis_report.html", 'w', encoding='utf-8') as f:
        f.write(html_attention)
    
    print("[OK] visualize_attention/analysis_report.html")
    
    # V向量HTML
    html_v = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ViT Value Vector Analysis - {image_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top: 30px; }}
        img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; border-radius: 5px; }}
        .section {{ margin-bottom: 40px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
        .info {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ViT Value Vector Analysis Report</h1>
        
        <div class="info">
            <p><strong>Image:</strong> {image_name}</p>
            <p><strong>Model:</strong> ViT-Base-16</p>
        </div>
        
        <div class="section">
            <h2>V向量层级进展</h2>
            <p>展示V向量强度在12层中的演化</p>
            <div class="image-grid">
                <img src="layer_progression/head0_progression.png" />
                <img src="layer_progression/head5_progression.png" />
                <img src="layer_progression/head11_progression.png" />
            </div>
        </div>
        
        <div class="section">
            <h2>V向量演化</h2>
            <p>展示V向量随层数的整体演化</p>
            <div class="image-grid">
                <img src="value_evolution/layer_evolution.png" />
            </div>
        </div>
        
        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #999;">
            Generated by ViT Single Image Analyzer | {image_name}
        </p>
    </div>
</body>
</html>
"""
    
    with open(output_dir / "visualize_v/analysis_report.html", 'w', encoding='utf-8') as f:
        f.write(html_v)
    
    print("[OK] visualize_v/analysis_report.html")
    
    print("\n" + "="*70)
    print(f" 完成！结果保存在: {output_dir}")
    print("="*70)
    print(f"\n查看报告：")
    print(f"  start {output_dir}/visualize_attention/analysis_report.html")
    print(f"  start {output_dir}/visualize_v/analysis_report.html")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为单张图片生成ViT完整分析')
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录名')
    parser.add_argument('--name', type=str, default=None, help='图片显示名称（可选）')
    
    args = parser.parse_args()
    
    success = generate_analysis(args.image, args.output, args.name)
    
    if success:
        print("\n分析生成成功！")
    else:
        print("\n分析生成失败！")
        sys.exit(1)
