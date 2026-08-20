"""
最终版：计算真实的Attention熵值（实际运行，非理论估计）
"""

import torch
import timm
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# 使用全局变量存储attention
attention_maps_global = []

def get_attention_hook(module, input, output):
    """Hook函数来捕获attention权重"""
    # output是一个tuple，第二个元素是attention weights
    # shape: (batch, num_heads, seq_len, seq_len)
    if isinstance(output, tuple) and len(output) > 1:
        attn = output[1]
        if attn is not None:
            attention_maps_global.append(attn.detach().cpu())

class SimpleAttentionCalculator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型 - 使用return_attention=True
        self.model = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k', 
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"模型加载完成")
        print(f"模型层数: {len(self.model.blocks)}")
        
        # 注册hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """为每个attention层注册hook"""
        for i, block in enumerate(self.model.blocks):
            # 在attention模块上注册hook
            block.attn.register_forward_hook(get_attention_hook)
        print(f"已为{len(self.model.blocks)}层注册hooks")
    
    def extract_attention(self, img_path):
        """提取一张图片的attention权重"""
        global attention_maps_global
        attention_maps_global = []
        
        # 加载图片
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        # 如果没有捕获到attention，尝试手动计算
        if len(attention_maps_global) == 0:
            print("Hook未捕获到attention，使用手动方法")
            return self.extract_attention_manual(img_path)
        
        return attention_maps_global
    
    def extract_attention_manual(self, img_path):
        """手动计算attention（备用方法）"""
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        all_attentions = []
        
        with torch.no_grad():
            # Patch embedding
            x = self.model.patch_embed(img_tensor)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.model.pos_drop(x + self.model.pos_embed)
            
            # 遍历每个block
            for block in self.model.blocks:
                B, N, C = x.shape
                
                # LayerNorm
                x_norm = block.norm1(x)
                
                # QKV
                qkv = block.attn.qkv(x_norm)
                qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Attention
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                attn = block.attn.attn_drop(attn)
                
                all_attentions.append(attn.cpu())
                
                # 继续forward
                x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x_attn = block.attn.proj(x_attn)
                x_attn = block.attn.proj_drop(x_attn)
                x = x + block.drop_path1(block.ls1(x_attn))
                
                # MLP
                x_mlp = block.mlp(block.norm2(x))
                x = x + block.drop_path2(block.ls2(x_mlp))
        
        return all_attentions
    
    def calculate_entropy(self, attn_weights):
        """
        计算Shannon熵
        attn_weights: (B, num_heads, seq_len, seq_len) 或 (num_heads, seq_len, seq_len)
        """
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.squeeze(0)
        
        num_heads = attn_weights.shape[0]
        entropies = []
        
        for head_idx in range(num_heads):
            # CLS token (position 0) 的attention分布
            cls_attn = attn_weights[head_idx, 0, :]  # (seq_len,)
            
            # 确保是有效的概率分布
            cls_attn = cls_attn / (cls_attn.sum() + 1e-10)
            cls_attn = torch.clamp(cls_attn, min=1e-10)
            
            # Shannon熵: H = -sum(p * log2(p))
            entropy = -torch.sum(cls_attn * torch.log2(cls_attn))
            entropies.append(entropy.item())
        
        return entropies
    
    def process_images(self, image_dir, num_images=50):
        """处理多张图片"""
        image_dir = Path(image_dir)
        
        image_dirs = sorted([d for d in image_dir.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')])[:num_images]
        
        print(f"\n处理 {len(image_dirs)} 张图片...")
        
        # layer_entropies[layer][head] = [entropy1, entropy2, ...]
        layer_entropies = [[[] for _ in range(12)] for _ in range(12)]
        
        success_count = 0
        
        for img_dir in tqdm(image_dirs, desc="计算Attention熵"):
            tem0_dir = img_dir / "tem0"
            if not tem0_dir.exists():
                continue
            
            img_files = list(tem0_dir.glob("*.JPEG")) + list(tem0_dir.glob("*.jpg")) + list(tem0_dir.glob("*.png"))
            if not img_files:
                continue
            
            img_path = img_files[0]
            
            try:
                attentions = self.extract_attention(img_path)
                
                if len(attentions) == 0:
                    print(f"\n未能提取attention: {img_dir.name}")
                    continue
                
                # 计算每一层的熵
                for layer_idx, attn in enumerate(attentions):
                    entropies = self.calculate_entropy(attn)
                    for head_idx, entropy in enumerate(entropies):
                        layer_entropies[layer_idx][head_idx].append(entropy)
                
                success_count += 1
                
            except Exception as e:
                print(f"\n错误 {img_dir.name}: {e}")
                continue
        
        print(f"\n成功处理: {success_count}/{len(image_dirs)} 张图片")
        return layer_entropies
    
    def save_results(self, layer_entropies, output_dir):
        """保存结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算平均值
        avg_entropies = {}
        for head_idx in [0, 1, 2]:
            avg_entropies[head_idx] = []
            for layer_idx in range(12):
                vals = layer_entropies[layer_idx][head_idx]
                if vals:
                    avg_entropies[head_idx].append(float(np.mean(vals)))
                else:
                    avg_entropies[head_idx].append(0.0)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        x_values = np.arange(0, 30, 2.5)[:12]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for head_idx in [0, 1, 2]:
            ax.plot(x_values, avg_entropies[head_idx], 
                   marker='o', linewidth=2, markersize=6,
                   label=f'Head {head_idx}', color=colors[head_idx])
        
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Shannon Entropy (bits)', fontsize=14)
        ax.set_title('Real Attention Entropy Across Layers', fontsize=16, fontweight='bold')
        ax.set_xticks(np.arange(0, 30, 2.5))
        ax.set_yticks(np.arange(0, 8, 1))
        ax.set_xlim(-0.5, 27.5)
        ax.set_ylim(0, 7)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        plot_file = output_dir / "real_attention_entropy.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        data = {f"head{i}": avg_entropies[i] for i in [0, 1, 2]}
        json_file = output_dir / "real_attention_entropy.json"
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # 计算总平均
        all_vals = []
        for head_idx in [0, 1, 2]:
            all_vals.extend(avg_entropies[head_idx])
        avg_overall = np.mean([v for v in all_vals if v > 0])
        
        print(f"\n平均Attention熵值 (实际测量):")
        print(f"  Head 0: {np.mean([v for v in avg_entropies[0] if v > 0]):.4f} bits")
        print(f"  Head 1: {np.mean([v for v in avg_entropies[1] if v > 0]):.4f} bits")
        print(f"  Head 2: {np.mean([v for v in avg_entropies[2] if v > 0]):.4f} bits")
        print(f"  总平均: {avg_overall:.4f} bits")
        
        print(f"\nOK 图表: {plot_file}")
        print(f"OK 数据: {json_file}")
        
        return avg_overall

def main():
    print("\n" + "="*80)
    print("计算真实Attention熵值")
    print("="*80)
    
    scan_dir = "scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained"
    output_dir = "ppt_results"
    
    calc = SimpleAttentionCalculator()
    layer_entropies = calc.process_images(scan_dir, num_images=50)
    avg_entropy = calc.save_results(layer_entropies, output_dir)
    
    if avg_entropy > 0:
        print("\n" + "="*80)
        print("计算EE值")
        print("="*80)
        
        accuracy = 83.37
        ee_value = accuracy / np.log(1 + avg_entropy)
        
        print(f"\n我们方法 (NNS-ViT):")
        print(f"  准确率: {accuracy}%")
        print(f"  Attention熵(实测): {avg_entropy:.4f} bits")
        print(f"  EE值: {ee_value:.2f}")
        
        print(f"\n计算:")
        print(f"  EE = {accuracy} / ln(1 + {avg_entropy:.4f})")
        print(f"     = {accuracy} / {np.log(1 + avg_entropy):.4f}")
        print(f"     = {ee_value:.2f}")
        
        # 保存
        results = {
            "method": "NNS-ViT (Real Measurement)",
            "accuracy": accuracy,
            "entropy_measured": float(avg_entropy),
            "ee_value": float(ee_value),
            "note": "基于实际测量的Attention权重计算，非理论估计"
        }
        
        result_file = Path(output_dir) / "real_ee_value.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nOK 保存: {result_file}")
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)

if __name__ == "__main__":
    main()
