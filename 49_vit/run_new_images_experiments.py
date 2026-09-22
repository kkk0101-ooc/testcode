"""
为两张新图片(狗和螃蟹)运行所有PPT实验
生成所有需要的实验图表
"""

import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import euclidean
import json
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class NewImagesExperimentRunner:
    """新图片实验运行器"""
    
    def __init__(self, output_dir='ppt_results_new'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        self.data_dir = self.output_dir / 'data'
        self.data_dir.mkdir(exist_ok=True)
        
        # 图片路径
        self.images = {
            '00000441': 'datas/ILSVRC2012_val/ILSVRC2012_val_00000441.JPEG',
            '00006100': 'datas/ILSVRC2012_val/ILSVRC2012_val_00006100.JPEG'
        }
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型（从本地）
        print("加载ViT-Base-16模型（本地文件）...")
        from load_local_model import load_vit_model
        self.model, self.device = load_vit_model('cuda' if torch.cuda.is_available() else 'cpu')
        print("  模型加载完成\n")
        
        # 存储提取的特征
        self.features_cache = {}
    
    def extract_all_features(self, image_tensor, img_id):
        """提取所有层的特征、attention和V向量"""
        
        if img_id in self.features_cache:
            return self.features_cache[img_id]
        
        all_attentions = []
        all_v_vectors = []
        all_features = []
        
        with torch.no_grad():
            # Patch embedding
            x = self.model.patch_embed(image_tensor)
            
            # Add cls token
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
            # Add position embedding
            x = self.model.pos_drop(x + self.model.pos_embed)
            
            # Extract from each layer
            for block in self.model.blocks:
                all_features.append(x.clone())
                
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
                
                # Forward
                x = block(x)
            
            x = self.model.norm(x)
            all_features.append(x.clone())
        
        result = {
            'attentions': all_attentions,
            'v_vectors': all_v_vectors,
            'features': all_features
        }
        
        self.features_cache[img_id] = result
        return result
    
    def load_image(self, image_path):
        """加载图片"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.230])
        ])
        
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = transform(original_image).unsqueeze(0).to(self.device)
        
        return original_image, image_tensor
    
    def exp1_module_visualization(self):
        """实验1: 各模块的可视化演化"""
        print("实验1: 模块可视化演化...")
        
        for img_id, img_path in self.images.items():
            print(f"  处理图片: {img_id}")
            
            original_image, image_tensor = self.load_image(img_path)
            features = self.extract_all_features(image_tensor, img_id)
            
            # 创建图形
            fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=150)
            
            # 原图
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # 12层的CLS attention (head 0)
            for layer_idx in range(12):
                row = (layer_idx + 1) // 4
                col = (layer_idx + 1) % 4
                
                attn = features['attentions'][layer_idx]
                cls_attn = attn[0, 0, 0, 1:].reshape(14, 14).numpy()
                
                im = axes[row, col].imshow(cls_attn, cmap='viridis', interpolation='bilinear')
                axes[row, col].set_title(f'Layer {layer_idx}', fontsize=10)
                axes[row, col].axis('off')
            
            # 隐藏多余位置
            for idx in range(13, 16):
                row = idx // 4
                col = idx % 4
                axes[row, col].axis('off')
            
            plt.suptitle(f'Module Evolution: {img_id}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = self.figures_dir / f'exp1_module_evolution_{img_id}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"    保存: {output_path}")
        
        print("  实验1完成\n")
    
    def exp2_v_vector_grid(self):
        """实验2: V向量网格"""
        print("实验2: V向量网格...")
        
        for img_id, img_path in self.images.items():
            print(f"  处理图片: {img_id}")
            
            original_image, image_tensor = self.load_image(img_path)
            features = self.extract_all_features(image_tensor, img_id)
            
            # 最后一层的V向量
            last_v = features['v_vectors'][-1]
            v_mag = last_v.norm(dim=-1)[0, :, 1:].reshape(12, 14, 14).numpy()
            
            # 创建图形
            fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=150)
            
            # 原图
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # 12个head
            for head_idx in range(12):
                row = (head_idx + 1) // 4
                col = (head_idx + 1) % 4
                
                im = axes[row, col].imshow(v_mag[head_idx], cmap='hot', interpolation='bilinear')
                axes[row, col].set_title(f'Head {head_idx}', fontsize=10)
                axes[row, col].axis('off')
            
            # 隐藏多余位置
            for idx in range(13, 16):
                row = idx // 4
                col = idx % 4
                axes[row, col].axis('off')
            
            plt.suptitle(f'V-Vector Grid: {img_id}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = self.figures_dir / f'exp2_v_vector_grid_{img_id}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"    保存: {output_path}")
        
        print("  实验2完成\n")
    
    def exp3_entropy_analysis(self):
        """实验3: 信息熵分析"""
        print("实验3: 信息熵分析...")
        
        # 计算每张图片每层每个head的熵
        entropy_data = {}
        
        for img_id, img_path in self.images.items():
            print(f"  处理图片: {img_id}")
            
            original_image, image_tensor = self.load_image(img_path)
            features = self.extract_all_features(image_tensor, img_id)
            
            layer_entropies = []
            
            for layer_idx, attn in enumerate(features['attentions']):
                head_entropies = []
                
                for head_idx in range(12):
                    # CLS attention distribution
                    cls_attn = attn[0, head_idx, 0, 1:].numpy()
                    cls_attn = cls_attn / (cls_attn.sum() + 1e-10)
                    
                    # Compute entropy
                    entropy = -np.sum(cls_attn * np.log(cls_attn + 1e-10))
                    head_entropies.append(entropy)
                
                layer_entropies.append(np.mean(head_entropies))
            
            entropy_data[img_id] = layer_entropies
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        
        for img_id, entropies in entropy_data.items():
            ax.plot(range(12), entropies, marker='o', linewidth=2, 
                   markersize=8, label=f'Image {img_id}')
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Average Entropy (bits)', fontsize=12)
        ax.set_title('Information Entropy Across Layers', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        output_path = self.figures_dir / 'exp3_entropy_analysis.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"    保存: {output_path}")
        
        # 保存数据（转换为Python原生类型）
        entropy_data_save = {k: [float(x) for x in v] for k, v in entropy_data.items()}
        data_path = self.data_dir / 'exp3_entropy_data.json'
        with open(data_path, 'w') as f:
            json.dump(entropy_data_save, f, indent=2)
        
        print("  实验3完成\n")
    
    def exp4_layer_changes(self):
        """实验4: 层间变化分析"""
        print("实验4: 层间变化分析...")
        
        change_data = {}
        
        for img_id, img_path in self.images.items():
            print(f"  处理图片: {img_id}")
            
            original_image, image_tensor = self.load_image(img_path)
            features = self.extract_all_features(image_tensor, img_id)
            
            layer_changes = []
            
            for layer_idx in range(11):  # 0-10, 对比11个相邻层对
                attn_curr = features['attentions'][layer_idx][0, :, 0, 1:].numpy()
                attn_next = features['attentions'][layer_idx + 1][0, :, 0, 1:].numpy()
                
                # Compute L2 distance
                change = np.mean([euclidean(attn_curr[h], attn_next[h]) 
                                 for h in range(12)])
                layer_changes.append(change)
            
            change_data[img_id] = layer_changes
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        
        for img_id, changes in change_data.items():
            ax.plot(range(11), changes, marker='s', linewidth=2, 
                   markersize=8, label=f'Image {img_id}')
        
        ax.set_xlabel('Layer Transition', fontsize=12)
        ax.set_ylabel('Average Change (L2 distance)', fontsize=12)
        ax.set_title('Layer-to-Layer Changes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        output_path = self.figures_dir / 'exp4_layer_changes.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"    保存: {output_path}")
        
        # 保存数据（转换为Python原生类型）
        change_data_save = {k: [float(x) for x in v] for k, v in change_data.items()}
        data_path = self.data_dir / 'exp4_change_data.json'
        with open(data_path, 'w') as f:
            json.dump(change_data_save, f, indent=2)
        
        print("  实验4完成\n")
    
    def exp5_distance_decay(self):
        """实验5: 距离衰减分析"""
        print("实验5: 距离衰减分析...")
        
        decay_data = {}
        
        for img_id, img_path in self.images.items():
            print(f"  处理图片: {img_id}")
            
            original_image, image_tensor = self.load_image(img_path)
            features = self.extract_all_features(image_tensor, img_id)
            
            # 使用最后一层
            last_attn = features['attentions'][-1][0, :, :, :].numpy()  # (12, 197, 197)
            
            # 计算每个patch到CLS的距离衰减
            distances = []
            attentions = []
            
            for patch_idx in range(1, 197):  # 跳过CLS token
                # 计算空间距离 (14x14 grid)
                row = (patch_idx - 1) // 14
                col = (patch_idx - 1) % 14
                center_row, center_col = 7, 7
                
                spatial_dist = np.sqrt((row - center_row)**2 + (col - center_col)**2)
                
                # CLS对该patch的平均attention
                avg_attn = np.mean([last_attn[h, 0, patch_idx] for h in range(12)])
                
                distances.append(spatial_dist)
                attentions.append(avg_attn)
            
            decay_data[img_id] = {
                'distances': distances,
                'attentions': attentions
            }
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        
        for img_id, data in decay_data.items():
            ax.scatter(data['distances'], data['attentions'], 
                      alpha=0.5, s=30, label=f'Image {img_id}')
        
        ax.set_xlabel('Spatial Distance from Center', fontsize=12)
        ax.set_ylabel('CLS Attention', fontsize=12)
        ax.set_title('Distance Decay Pattern', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        output_path = self.figures_dir / 'exp5_distance_decay.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"    保存: {output_path}")
        
        print("  实验5完成\n")
    
    def exp6_concentration_evolution(self):
        """实验6: 注意力集中度演化"""
        print("实验6: 注意力集中度演化...")
        
        for img_id, img_path in self.images.items():
            print(f"  处理图片: {img_id}")
            
            original_image, image_tensor = self.load_image(img_path)
            features = self.extract_all_features(image_tensor, img_id)
            
            # 为3个代表性的head绘制演化
            heads_to_plot = [0, 5, 11]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
            
            for idx, head_idx in enumerate(heads_to_plot):
                concentrations = []
                
                for layer_idx in range(12):
                    attn = features['attentions'][layer_idx]
                    cls_attn = attn[0, head_idx, 0, 1:].numpy()
                    
                    # 计算集中度 (标准差或Gini系数)
                    concentration = np.std(cls_attn)
                    concentrations.append(concentration)
                
                axes[idx].plot(range(12), concentrations, marker='o', 
                             linewidth=2, markersize=8, color='blue')
                axes[idx].set_xlabel('Layer', fontsize=11)
                axes[idx].set_ylabel('Concentration (std)', fontsize=11)
                axes[idx].set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
            
            plt.suptitle(f'Attention Concentration Evolution: {img_id}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_path = self.figures_dir / f'exp6_concentration_{img_id}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"    保存: {output_path}")
        
        print("  实验6完成\n")
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("="*80)
        print("开始运行新图片的所有实验")
        print("="*80)
        print()
        
        self.exp1_module_visualization()
        self.exp2_v_vector_grid()
        self.exp3_entropy_analysis()
        self.exp4_layer_changes()
        self.exp5_distance_decay()
        self.exp6_concentration_evolution()
        
        print("="*80)
        print("所有实验完成!")
        print(f"结果保存在: {self.output_dir}")
        print(f"图表保存在: {self.figures_dir}")
        print(f"数据保存在: {self.data_dir}")
        print("="*80)
        
        # 生成总结
        self.generate_summary()
    
    def generate_summary(self):
        """生成实验总结"""
        summary = f"""# 新图片实验结果总结

## 图片信息

- 图片1: 00000441 (狗)
- 图片2: 00006100 (螃蟹)

## 完成的实验

1. 实验1: 模块可视化演化 - 展示12层的CLS attention演化
2. 实验2: V向量网格 - 展示12个head的V向量幅值
3. 实验3: 信息熵分析 - 分析各层attention的信息熵变化
4. 实验4: 层间变化分析 - 计算相邻层之间的attention变化
5. 实验5: 距离衰减分析 - 分析空间距离对attention的影响
6. 实验6: 集中度演化 - 展示attention集中度随层数的变化

## 输出文件

### 图表文件 (figures/)

- exp1_module_evolution_00000441.png
- exp1_module_evolution_00006100.png
- exp2_v_vector_grid_00000441.png
- exp2_v_vector_grid_00006100.png
- exp3_entropy_analysis.png
- exp4_layer_changes.png
- exp5_distance_decay.png
- exp6_concentration_00000441.png
- exp6_concentration_00006100.png

### 数据文件 (data/)

- exp3_entropy_data.json
- exp4_change_data.json

## 使用说明

1. 所有图表均为高分辨率PNG格式
2. 可直接用于PPT演示
3. 数据文件包含原始数值，可用于进一步分析

## 目录结构

```
{self.output_dir}/
├── figures/          # 所有实验图表
├── data/            # 实验数据JSON文件
└── SUMMARY.md       # 本文件
```
"""
        
        summary_path = self.output_dir / 'SUMMARY.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n总结报告已保存: {summary_path}")

def main():
    """主函数"""
    runner = NewImagesExperimentRunner(output_dir='ppt_results_new')
    runner.run_all_experiments()

if __name__ == "__main__":
    main()
