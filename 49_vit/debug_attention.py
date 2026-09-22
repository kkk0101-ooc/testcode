"""
调试attention提取
"""

import torch
import timm
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
model = model.to(device)
model.eval()

# 打印模型结构
print("模型结构:")
print(model)

# 检查第一个block的结构
print("\n第一个Block的结构:")
print(model.blocks[0])

# 检查attention的结构
print("\n第一个Block的Attention结构:")
print(model.blocks[0].attn)

# 加载一张测试图片
scan_dir = Path("scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained")
img_dirs = sorted([d for d in scan_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
test_dir = img_dirs[0]
img_files = list((test_dir / "tem0").glob("*.JPEG"))
img_path = img_files[0]

# 预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.405])
])

img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

print(f"\n输入shape: {img_tensor.shape}")

# 前向传播第一层
with torch.no_grad():
    x = model.patch_embed(img_tensor)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = model.pos_drop(x + model.pos_embed)
    
    print(f"Patch embedding后shape: {x.shape}")
    
    # 第一个block
    block = model.blocks[0]
    B, N, C = x.shape
    print(f"B={B}, N={N}, C={C}")
    print(f"num_heads={block.attn.num_heads}")
    
    # QKV
    qkv = block.attn.qkv(block.norm1(x))
    print(f"QKV shape: {qkv.shape}")
    
    qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
    print(f"QKV reshaped: {qkv.shape}")
    
    qkv = qkv.permute(2, 0, 3, 1, 4)
    print(f"QKV permuted: {qkv.shape}")
    
    q, k, v = qkv[0], qkv[1], qkv[2]
    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"V shape: {v.shape}")
    
    # Attention
    attn = (q @ k.transpose(-2, -1)) * block.attn.scale
    print(f"Attention (before softmax) shape: {attn.shape}")
    print(f"Attention (before softmax) min/max: {attn.min().item():.4f} / {attn.max().item():.4f}")
    
    attn = attn.softmax(dim=-1)
    print(f"Attention (after softmax) shape: {attn.shape}")
    print(f"Attention (after softmax) min/max: {attn.min().item():.4f} / {attn.max().item():.4f}")
    print(f"Attention sum along last dim: {attn.sum(dim=-1)[0, 0, :5]}")
    
    # CLS attention
    cls_attn = attn[0, :, 0, :]  # (num_heads, seq_len)
    print(f"\nCLS attention shape: {cls_attn.shape}")
    print(f"CLS attention (head 0) min/max: {cls_attn[0].min().item():.6f} / {cls_attn[0].max().item():.6f}")
    print(f"CLS attention (head 0) sum: {cls_attn[0].sum().item():.6f}")
    print(f"CLS attention (head 0) first 10 values: {cls_attn[0, :10]}")
    
    # 计算熵
    cls_attn_normalized = cls_attn[0] / (cls_attn[0].sum() + 1e-10)
    cls_attn_safe = cls_attn_normalized + 1e-10
    entropy = -torch.sum(cls_attn_safe * torch.log2(cls_attn_safe))
    
    print(f"\nHead 0 entropy: {entropy.item():.4f} bits")
