"""
生成PPT需要的实验图
包括：完整可视化、V向量网格、各层CLS演化等
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

def extract_all_features(model, image_tensor, device):
    """提取所有层的特征、attention和V向量"""
    
    all_attentions = []  # 每层的attention
    all_v_vectors = []   # 每层的V向量
    all_features = []    # 每层的特征
    
    with torch.no_grad():
        # 先过patch embedding
        x = model.patch_embed(image_tensor)
        
        # 添加cls token
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 添加position embedding
        x = model.pos_drop(x + model.pos_embed)
        
        # 逐层提取
        for i, block in enumerate(model.blocks):
            # 保存输入特征
            all_features.append(x.clone())
            
            # 提取attention和V
            x_norm = block.norm1(x)
            B, N, C = x_norm.shape
            
            # Get Q, K, V
            qkv = block.attn.qkv(x_norm).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            
            all_attentions.append(attn.cpu())
            all_v_vectors.append(v.cpu())
            
            # Forward through block
            x = block(x)
        
        # 最后的norm
        x = model.norm(x)
        all_features.append(x.clone())
    
    return all_attentions, all_v_vectors, all_features

def visualize_combined(image_path, model, device, output_dir, img_id):
    """生成完整可视化（原图 + 12头V向量 + 12层CLS）"""
    
    print(f"  生成完整可视化...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.230])
    ])
    
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()
    
    # 提取特征
    all_attentions, all_v_vectors, all_features = extract_all_features(model, image_tensor, device)
    
    # 最后一层的V向量（12个head）
    last_v = all_v_vectors[-1]  # (B, num_heads, N, head_dim)
    v_mag = last_v.norm(dim=-1)[0, :, 1:].reshape(12, 14, 14).numpy()
    
    # 各层的CLS attention（取head 0）
    cls_attns = []
    for attn in all_attentions:
        cls_attn = attn[0, 0, 0, 1:].reshape(14, 14).numpy()  # Head 0的CLS attention
        cls_attns.append(cls_attn)
    
    # 创建图形
    fig = plt.figure(figsize=(24, 18), dpi=200)
    
    # 原图
    ax_orig = plt.subplot2grid((3, 13), (0, 0), colspan=13, rowspan=1)
    ax_orig.imshow(original_image)
    ax_orig.set_title(f'Original Image (ID: {img_id}, Pred: {pred})', 
                     fontsize=16, fontweight='bold')
    ax_orig.axis('off')
    
    # V向量热力图（12个head）
    for head_idx in range(12):
        ax = plt.subplot2grid((3, 13), (1, head_idx), colspan=1, rowspan=1)
        
        v_head = v_mag[head_idx]
        
        # 使用hot colormap，高值自然显示为白色/黄色
        im = ax.imshow(v_head, cmap='hot', interpolation='bilinear', vmin=0)
        ax.set_title(f'H{head_idx}', fontsize=10)
        ax.axis('off')
        
        if head_idx == 11:
            cbar_ax = plt.subplot2grid((3, 13), (1, 12), colspan=1, rowspan=1)
            plt.colorbar(im, cax=cbar_ax, label='V-magnitude')
    
    # CLS attention（12层）
    for layer_idx in range(12):
        ax = plt.subplot2grid((3, 13), (2, layer_idx), colspan=1, rowspan=1)
        
        cls_attn = cls_attns[layer_idx]
        
        # 使用viridis colormap，高值自然显示为黄色/亮色
        im = ax.imshow(cls_attn, cmap='viridis', interpolation='bilinear', vmin=0)
        ax.set_title(f'L{layer_idx}', fontsize=10)
        ax.axis('off')
        
        if layer_idx == 11:
            cbar_ax = plt.subplot2grid((3, 13), (2, 12), colspan=1, rowspan=1)
            plt.colorbar(im, cax=cbar_ax, label='Attention')
    
    plt.suptitle(f'NNS-ViT Visualization: ImageNet {img_id}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    output_path = output_dir / f"imagenet_{img_id}_combined.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"    保存: {output_path}")
    return output_path

def visualize_v_vectors(image_path, model, device, output_dir, img_id):
    """生成V向量网格（原图 + 12个head的V向量）"""
    
    print(f"  生成V向量网格...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.230])
    ])
    
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    all_attentions, all_v_vectors, all_features = extract_all_features(model, image_tensor, device)
    
    last_v = all_v_vectors[-1]
    v_mag = last_v.norm(dim=-1)[0, :, 1:].reshape(12, 14, 14).numpy()
    
    # 创建图形（4行4列，第一个是原图）
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), dpi=150)
    
    # 原图
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 12个head
    for idx in range(12):
        row = (idx + 1) // 4
        col = (idx + 1) % 4
        
        im = axes[row, col].imshow(v_mag[idx], cmap='hot', interpolation='bilinear')
        axes[row, col].set_title(f'Head {idx}', fontsize=10)
        axes[row, col].axis('off')
    
    # 最后3个空位
    for idx in range(13, 16):
        row = idx // 4
        col = idx % 4
        axes[row, col].axis('off')
    
    plt.suptitle(f'V-magnitude Visualization: {img_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"v_vectors_{img_id}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"    保存: {output_path}")
    return output_path

def visualize_cls_layers(image_path, model, device, output_dir, img_id):
    """生成各层CLS演化（原图 + 12层的CLS attention）"""
    
    print(f"  生成CLS演化...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.230])
    ])
    
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    all_attentions, all_v_vectors, all_features = extract_all_features(model, image_tensor, device)
    
    cls_attns = []
    for attn in all_attentions:
        cls_attn = attn[0, 0, 0, 1:].reshape(14, 14).numpy()
        cls_attns.append(cls_attn)
    
    # 创建图形
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), dpi=150)
    
    # 原图
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 12层
    for idx in range(12):
        row = (idx + 1) // 4
        col = (idx + 1) % 4
        
        im = axes[row, col].imshow(cls_attns[idx], cmap='viridis', interpolation='bilinear')
        axes[row, col].set_title(f'Layer {idx}', fontsize=10)
        axes[row, col].axis('off')
    
    # 最后3个空位
    for idx in range(13, 16):
        row = idx // 4
        col = idx % 4
        axes[row, col].axis('off')
    
    plt.suptitle(f'CLS Attention Evolution: {img_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"cls_layers_{img_id}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"    保存: {output_path}")
    return output_path

def exp1_module_evolution(image_path, model, device, output_dir, img_id):
    """实验1: 模块演化(12层x12头)"""
    print(f"  实验1: 模块演化...")
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.230])
    ])
    
    original = Image.open(image_path).convert('RGB')
    tensor = transform(original).unsqueeze(0).to(device)
    all_attentions, _, _ = extract_all_features(model, tensor, device)
    
    fig = plt.figure(figsize=(26, 24), dpi=150)
    ax_orig = plt.subplot2grid((13, 12), (0, 4), colspan=4, rowspan=1)
    ax_orig.imshow(original)
    ax_orig.set_title(f'Original: {img_id}', fontsize=14, fontweight='bold')
    ax_orig.axis('off')
    
    for layer_idx in range(12):
        for head_idx in range(12):
            ax = plt.subplot2grid((13, 12), (layer_idx+1, head_idx))
            attn = all_attentions[layer_idx][0, head_idx, 0, 1:].reshape(14, 14).numpy()
            ax.imshow(attn, cmap='viridis', interpolation='bilinear')
            if head_idx == 0:
                ax.set_ylabel(f'L{layer_idx}', fontsize=9)
            if layer_idx == 0:
                ax.set_title(f'H{head_idx}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle(f'Exp1: Module Evolution - {img_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / f"exp1_module_{img_id}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    保存: {output_path}")
    return output_path

def exp2_v_vector_grid(image_path, model, device, output_dir, img_id):
    """实验2: V向量网格(12层x12头)"""
    print(f"  实验2: V向量网格...")
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.230])
    ])
    
    original = Image.open(image_path).convert('RGB')
    tensor = transform(original).unsqueeze(0).to(device)
    _, all_v_vectors, _ = extract_all_features(model, tensor, device)
    
    fig = plt.figure(figsize=(26, 24), dpi=150)
    ax_orig = plt.subplot2grid((13, 12), (0, 4), colspan=4, rowspan=1)
    ax_orig.imshow(original)
    ax_orig.set_title(f'Original: {img_id}', fontsize=14, fontweight='bold')
    ax_orig.axis('off')
    
    for layer_idx in range(12):
        for head_idx in range(12):
            ax = plt.subplot2grid((13, 12), (layer_idx+1, head_idx))
            v = all_v_vectors[layer_idx][0, head_idx, 1:, :]
            v_mag = v.norm(dim=-1).reshape(14, 14).numpy()
            ax.imshow(v_mag, cmap='hot', interpolation='bilinear')
            if head_idx == 0:
                ax.set_ylabel(f'L{layer_idx}', fontsize=9)
            if layer_idx == 0:
                ax.set_title(f'H{head_idx}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle(f'Exp2: V-vector Grid - {img_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / f"exp2_v_grid_{img_id}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    保存: {output_path}")
    return output_path

def exp3_entropy_analysis(image_paths, model, device, output_dir):
    """实验3: 信息熵分析"""
    print("  实验3: 信息熵分析...")
    import json
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.230])
    ])
    
    entropy_data = {}
    for img_id, img_path in image_paths.items():
        original = Image.open(img_path).convert('RGB')
        tensor = transform(original).unsqueeze(0).to(device)
        all_attentions, _, _ = extract_all_features(model, tensor, device)
        
        layer_entropies = []
        for attn in all_attentions:
            head_entropies = []
            for head_idx in range(12):
                attn_w = attn[0, head_idx, 0, 1:]
                entropy = -(attn_w * torch.log(attn_w + 1e-10)).sum().item()
                head_entropies.append(entropy)
            layer_entropies.append(head_entropies)
        entropy_data[img_id] = layer_entropies
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for idx, (img_id, entropies) in enumerate(entropy_data.items()):
        ax = axes[idx]
        im = ax.imshow(np.array(entropies).T, cmap='coolwarm', aspect='auto')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Head', fontsize=12)
        ax.set_title(f'Entropy: {img_id}', fontsize=13, fontweight='bold')
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        plt.colorbar(im, ax=ax, label='Entropy')
    
    plt.suptitle('Exp3: Attention Entropy Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "exp3_entropy.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    with open(output_dir / "exp3_data.json", 'w') as f:
        json.dump({k: [[float(e) for e in r] for r in v] for k, v in entropy_data.items()}, f, indent=2)
    
    print(f"    保存: {output_path}")
    return output_path

def exp4_layer_changes(image_paths, model, device, output_dir):
    """实验4: 层间变化"""
    print("  实验4: 层间变化...")
    import json
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.230])
    ])
    
    changes_data = {}
    for img_id, img_path in image_paths.items():
        original = Image.open(img_path).convert('RGB')
        tensor = transform(original).unsqueeze(0).to(device)
        all_attentions, _, _ = extract_all_features(model, tensor, device)
        
        layer_changes = []
        for i in range(len(all_attentions) - 1):
            attn1 = all_attentions[i][0, :, 0, 1:]
            attn2 = all_attentions[i+1][0, :, 0, 1:]
            diff = (attn1 - attn2).norm(dim=-1).mean().item()
            layer_changes.append(diff)
        changes_data[img_id] = layer_changes
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for img_id, changes in changes_data.items():
        ax.plot(range(len(changes)), changes, marker='o', linewidth=2, label=img_id)
    ax.set_xlabel('Layer Transition', fontsize=12)
    ax.set_ylabel('Change Magnitude', fontsize=12)
    ax.set_title('Exp4: Layer-to-Layer Changes', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / "exp4_changes.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    with open(output_dir / "exp4_data.json", 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in changes_data.items()}, f, indent=2)
    
    print(f"    保存: {output_path}")
    return output_path

def exp5_distance_decay(image_paths, model, device, output_dir):
    """实验5: 距离衰减"""
    print("  实验5: 距离衰减...")
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.230])
    ])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    for idx, (img_id, img_path) in enumerate(image_paths.items()):
        original = Image.open(img_path).convert('RGB')
        tensor = transform(original).unsqueeze(0).to(device)
        all_attentions, _, _ = extract_all_features(model, tensor, device)
        
        last_attn = all_attentions[-1][0, 0, :, :]
        cls_attn = last_attn[0, 1:].numpy()
        
        center = 7
        distances, attentions = [], []
        for i in range(14):
            for j in range(14):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                distances.append(dist)
                attentions.append(cls_attn[i * 14 + j])
        
        ax = axes[idx]
        ax.scatter(distances, attentions, alpha=0.5)
        z = np.polyfit(distances, attentions, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(0, max(distances), 100)
        ax.plot(x_trend, p(x_trend), 'r-', linewidth=2, label='Trend')
        ax.set_xlabel('Distance from Center', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title(f'Distance Decay: {img_id}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Exp5: Distance Decay Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "exp5_decay.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    保存: {output_path}")
    return output_path

def exp6_concentration(image_paths, model, device, output_dir):
    """实验6: 注意力集中度"""
    print("  实验6: 注意力集中度...")
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.230])
    ])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    for idx, (img_id, img_path) in enumerate(image_paths.items()):
        original = Image.open(img_path).convert('RGB')
        tensor = transform(original).unsqueeze(0).to(device)
        all_attentions, _, _ = extract_all_features(model, tensor, device)
        
        head_concentrations = []
        for head_idx in range(3):
            layer_conc = []
            for attn in all_attentions:
                attn_w = attn[0, head_idx, 0, 1:].numpy()
                sorted_attn = np.sort(attn_w)
                n = len(sorted_attn)
                gini = (2 * np.sum((np.arange(n) + 1) * sorted_attn) / (n * np.sum(sorted_attn))) - (n + 1) / n
                layer_conc.append(gini)
            head_concentrations.append(layer_conc)
        
        ax = axes[idx]
        for head_idx in range(3):
            ax.plot(range(12), head_concentrations[head_idx], marker='o', linewidth=2, label=f'Head {head_idx}')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Concentration (Gini)', fontsize=12)
        ax.set_title(f'Concentration: {img_id}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Exp6: Attention Concentration Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "exp6_concentration.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    保存: {output_path}")
    return output_path

def main():
    """主函数"""
    
    print("="*80)
    print("为新图片生成所有PPT实验图")
    print("="*80)
    print()
    
    # 图片路径
    images = {
        '00000441': 'datas/ILSVRC2012_val/ILSVRC2012_val_00000441.JPEG',
        '00006100': 'datas/ILSVRC2012_val/ILSVRC2012_val_00006100.JPEG'
    }
    
    # 输出目录
    output_dirs = {
        'combined': Path('ppt_results/imagenet_visualizations/combined'),
        'v_vectors': Path('ppt_results/imagenet_visualizations/v_vectors'),
        'cls_layers': Path('ppt_results/imagenet_visualizations/cls_layers')
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
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
    results = {}
    
    for img_id, img_path in images.items():
        print(f"处理图片: {img_id}")
        
        if not Path(img_path).exists():
            print(f"  错误: 图片不存在")
            continue
        
        results[img_id] = {}
        
        # 1. 完整可视化
        results[img_id]['combined'] = visualize_combined(
            img_path, model, device, output_dirs['combined'], img_id
        )
        
        # 2. V向量网格
        results[img_id]['v_vectors'] = visualize_v_vectors(
            img_path, model, device, output_dirs['v_vectors'], img_id
        )
        
        # 3. CLS演化
        results[img_id]['cls_layers'] = visualize_cls_layers(
            img_path, model, device, output_dirs['cls_layers'], img_id
        )
        
        print()
    
    print("="*80)
    print("完成!")
    print("="*80)
    print()
    
    print("生成的文件:")
    for img_id, files in results.items():
        print(f"\n图片 {img_id}:")
        for file_type, file_path in files.items():
            print(f"  {file_type}: {file_path}")
    
    print("\n说明:")
    print("  - 使用自然的colormap效果")
    print("  - hot colormap: 高值自然显示为白色/黄色")
    print("  - viridis colormap: 高值自然显示为黄色/亮色")
    print("  - 无人工添加的轮廓和标记")

if __name__ == "__main__":
    main()
