"""
基于20张图片生成完整的ViT分析 - 类似do_images结构
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

# 导入本地模型
from load_local_model import load_vit_model

print("="*70)
print(" ViT完整分析 - 20张ImageNet图片")
print("="*70)

# 配置
IMAGE_DIR = Path("datas/ILSVRC2012_val")
OUTPUT_DIR = Path("analysis_results_20images")
NUM_IMAGES = 20

# 创建输出目录
subdirs = [
    "visualize_attention/attention_evolution",
    "visualize_attention/head_specialization", 
    "visualize_attention/spatial_distribution",
    "visualize_attention/quantitative",
    "visualize_v/value_evolution",
    "visualize_v/quantitative",
    "raw_data/attention",
    "raw_data/values"
]

for subdir in subdirs:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

# 选择图片
all_images = sorted(list(IMAGE_DIR.glob("*.JPEG")))
indices = np.linspace(0, len(all_images)-1, NUM_IMAGES, dtype=int)
selected_images = [all_images[i] for i in indices]

print(f"\n从{len(all_images)}张图片中选择了{NUM_IMAGES}张:")
for i, img in enumerate(selected_images, 1):
    print(f"  {i:2d}. {img.name}")

# 加载模型
print("\n加载模型...")
model, device = load_vit_model('cuda')
print(f"[OK] 模型已加载到: {device}")

# 预处理函数
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 提取数据
print("\n提取attention和V向量...")
all_data = []

for img_path in tqdm(selected_images):
    # 加载图片
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 手动提取attention（参考generate_all_ppt_figures.py的方法）
    attention_maps = []
    
    with torch.no_grad():
        # Patch embedding
        x = model.patch_embed(img_tensor)
        
        # 添加cls token
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 添加position embedding
        x = model.pos_drop(x + model.pos_embed)
        
        # 逐层提取
        for block in model.blocks:
            x_norm = block.norm1(x)
            B, N, C = x_norm.shape
            
            # Get Q, K, V
            qkv = block.attn.qkv(x_norm).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            
            attention_maps.append(attn.cpu())
            
            # Forward through block
            x = block(x)
        
        # 最后的norm和预测
        x = model.norm(x)
        out = model.head(x[:, 0])
        pred = out.argmax(1).item()
    
    # 保存
    img_id = img_path.stem
    for idx, attn in enumerate(attention_maps):
        save_path = OUTPUT_DIR / "raw_data" / "attention" / img_id
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / f"layer{idx:02d}.npy", attn.cpu().numpy())
    
    all_data.append({
        'name': img_path.name,
        'id': img_id,
        'attention': attention_maps,
        'pred': pred
    })

print(f"[OK] 已处理{len(all_data)}张图片\n")

# 生成分析
print("生成可视化分析...")

# 继续运行分析脚本
exec(open("generate_analysis_visualizations.py", encoding='utf-8').read())

print("\n" + "="*70)
print(f" 分析完成！结果保存在: {OUTPUT_DIR}")
print("="*70)
