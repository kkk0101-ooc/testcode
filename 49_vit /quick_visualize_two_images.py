"""
快速为两张图片生成可视化
直接使用timm ViT-Base模型提取特征并生成热力图
"""

import torch
import timm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')

def extract_attention_and_v(model, image_tensor, device):
    """提取attention和V向量"""
    
    attention_maps = []
    v_vectors = []
    
    def hook_fn_attn(module, input, output):
        # output is tuple: (attn_output, attn_weights)
        # attn_weights shape: (batch, num_heads, seq_len, seq_len)
        attention_maps.append(output[1].detach().cpu())
    
    def hook_fn_v(module, input, output):
        # V向量 shape: (batch, num_heads, seq_len, head_dim)
        v_vectors.append(output.detach().cpu())
    
    # 注册hooks
    hooks = []
    for i, block in enumerate(model.blocks):
        # 注册attention hook
        hook = block.attn.register_forward_hook(hook_fn_attn)
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps, v_vectors

def visualize_complete(image_path, model, device, output_dir):
    """为一张图片生成完整可视化"""
    
    image_path = Path(image_path)
    img_id = image_path.stem.split('_')[-1]  # 提取00000441
    
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
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # 获取预测
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()
    
    print(f"  预测类别: {pred}")
    
    # 提取中间特征（使用forward_features）
    with torch.no_grad():
        features = model.forward_features(image_tensor)
        # features shape: (1, 197, 768) for ViT-Base
    
    print(f"  提取特征: {features.shape}")
    
    # 获取最后一层的attention (使用简化方法)
    last_block = model.blocks[-1]
    with torch.no_grad():
        x_norm = last_block.norm1(features)
        B, N, C = x_norm.shape
        
        # Get Q, K, V
        qkv = last_block.attn.qkv(x_norm).reshape(B, N, 3, last_block.attn.num_heads, C // last_block.attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * last_block.attn.scale
        attn = attn.softmax(dim=-1)
        
        # Compute V magnitude
        v_mag = v.norm(dim=-1)  # (B, num_heads, N)
    
    print(f"  Attention shape: {attn.shape}")
    print(f"  V magnitude shape: {v_mag.shape}")
    
    # 生成组合可视化
    fig = plt.figure(figsize=(24, 18), dpi=200)
    
    # 原图 (占据上方整行)
    ax_orig = plt.subplot2grid((3, 13), (0, 0), colspan=13, rowspan=1)
    ax_orig.imshow(original_image)
    ax_orig.set_title(f'Original Image (ID: {img_id}, Pred: {pred})', fontsize=16, fontweight='bold')
    ax_orig.axis('off')
    
    # V向量热力图 (12个head，第二行)
    for head_idx in range(12):
        ax = plt.subplot2grid((3, 13), (1, head_idx), colspan=1, rowspan=1)
        
        # 取CLS token的V magnitude
        v_head = v_mag[0, head_idx, 1:].reshape(14, 14).cpu().numpy()
        
        im = ax.imshow(v_head, cmap='hot', interpolation='bilinear')
        ax.set_title(f'H{head_idx}', fontsize=10)
        ax.axis('off')
        
        # 添加colorbar
        if head_idx == 11:
            cbar_ax = plt.subplot2grid((3, 13), (1, 12), colspan=1, rowspan=1)
            plt.colorbar(im, cax=cbar_ax, label='V-magnitude')
    
    # CLS attention演化 (12层，第三行)
    # 这里简化为只显示最后一层的12个head的CLS attention
    for head_idx in range(12):
        ax = plt.subplot2grid((3, 13), (2, head_idx), colspan=1, rowspan=1)
        
        # CLS token对其他tokens的attention
        cls_attn = attn[0, head_idx, 0, 1:].reshape(14, 14).cpu().numpy()
        
        im = ax.imshow(cls_attn, cmap='viridis', interpolation='bilinear')
        ax.set_title(f'H{head_idx}', fontsize=10)
        ax.axis('off')
        
        # 添加colorbar
        if head_idx == 11:
            cbar_ax = plt.subplot2grid((3, 13), (2, 12), colspan=1, rowspan=1)
            plt.colorbar(im, cax=cbar_ax, label='Attention')
    
    plt.suptitle(f'NNS-ViT Visualization: ImageNet {img_id}', fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / f"imagenet_{img_id}_combined.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"  保存: {output_path}")
    
    return output_path

def main():
    """主函数"""
    
    print("="*80)
    print("快速ImageNet可视化生成")
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
            output_path = visualize_complete(img_path, model, device, output_dir)
            results.append(output_path)
        else:
            print(f"错误: 图片不存在 - {img_path}")
    
    print("\n" + "="*80)
    print("完成!")
    print("="*80)
    print("\n生成的文件:")
    for path in results:
        print(f"  {path}")

if __name__ == "__main__":
    main()
