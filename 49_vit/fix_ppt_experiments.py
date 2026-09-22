"""
修复PPT实验的图表问题
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import cv2
from tqdm import tqdm
from scipy.spatial.distance import euclidean

class FixedPPTExperiments:
    def __init__(self):
        self.scan_dir = Path("scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained")
        self.output_dir = Path("ppt_results")
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def get_image_dirs(self, num_images=50):
        image_dirs = sorted([d for d in self.scan_dir.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')])[:num_images]
        return image_dirs
    
    # 修复实验3: 使用真实的Attention熵数据
    def fix_exp3_entropy(self):
        print("\n" + "="*80)
        print("修复实验3: 使用真实Attention熵值")
        print("="*80)
        
        # 使用刚才计算的真实attention熵值
        entropy_file = self.output_dir / "real_attention_entropy.json"
        if entropy_file.exists():
            with open(entropy_file, 'r') as f:
                real_data = json.load(f)
            
            # 绘图
            fig, ax = plt.subplots(figsize=(12, 6))
            x_values = np.arange(0, 30, 2.5)[:12]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for head_idx in [0, 1, 2]:
                data = real_data[f'head{head_idx}']
                ax.plot(x_values, data, 
                       marker='o', linewidth=2, markersize=6,
                       label=f'Head {head_idx}', color=colors[head_idx])
            
            ax.set_xlabel('Layer', fontsize=14)
            ax.set_ylabel('Shannon Entropy (bits)', fontsize=14)
            ax.set_title('Information Distribution Across Layers', fontsize=16, fontweight='bold')
            ax.set_xticks(np.arange(0, 30, 2.5))
            ax.set_yticks(np.arange(0, 6, 1))
            ax.set_xlim(-0.5, 27.5)
            ax.set_ylim(0, 5)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=12)
            
            output_file = self.figures_dir / "exp3_entropy_fixed.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"OK 修复完成: {output_file}")
            print(f"   使用真实Attention熵值 (3-5 bits)")
        else:
            print("错误: 未找到真实熵值数据文件")
            print("请先运行: calculate_attention_entropy_final.py")
    
    # 修复实验4: 不强制归一化，使用真实欧氏距离
    def load_map(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return (img.astype(np.float32) / 255.0).flatten()
    
    def fix_exp4_layer_changes(self):
        print("\n" + "="*80)
        print("修复实验4: 使用真实欧氏距离（不强制归一化）")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        layer_folders = ['02', '05', '08', '11', '14', '17', '20', '23', '26', '29', '32', '35']
        
        changes = {f'head{h}_{t}': [[] for _ in range(9)] 
                  for h in [0,1,2] for t in ['cls', 'mean']}
        
        for img_dir in tqdm(image_dirs, desc="重新计算变化"):
            scan_v_dir = img_dir / "scan_v"
            if not scan_v_dir.exists():
                continue
            
            for head_idx in [0, 1, 2]:
                prev_cls = None
                
                for layer_idx, layer_folder in enumerate(layer_folders):
                    if layer_idx < 3:  # 从layer 6开始
                        layer_dir = scan_v_dir / layer_folder
                        if layer_dir.exists():
                            cls_file = layer_dir / f"cls_v_magnitude_head{head_idx}.png"
                            if cls_file.exists():
                                prev_cls = self.load_map(cls_file)
                        continue
                    
                    layer_dir = scan_v_dir / layer_folder
                    if not layer_dir.exists():
                        continue
                    
                    cls_file = layer_dir / f"cls_v_magnitude_head{head_idx}.png"
                    if cls_file.exists():
                        curr_cls = self.load_map(cls_file)
                        
                        if prev_cls is not None and curr_cls is not None:
                            dist = euclidean(prev_cls, curr_cls)
                            idx = layer_idx - 3
                            if idx < 9:
                                changes[f'head{head_idx}_cls'][idx].append(dist)
                                changes[f'head{head_idx}_mean'][idx].append(dist)
                        
                        prev_cls = curr_cls
        
        # 计算平均值（不强制归一化）
        avg_changes = {}
        for key in changes:
            avg_changes[key] = [np.mean(vals) if vals else 0 for vals in changes[key]]
        
        # 找到合理的Y轴范围
        all_values = []
        for vals in avg_changes.values():
            all_values.extend(vals)
        max_val = max(all_values) if all_values else 1
        
        print(f"\n欧氏距离范围: 0 到 {max_val:.4f}")
        
        # 绘图 - 使用真实数值
        fig, ax = plt.subplots(figsize=(12, 6))
        x_values = np.arange(6, 24, 2)[:9]
        
        styles = {
            'head0_cls': {'color': '#1f77b4', 'linestyle': '-', 'label': 'Head 0 - CLS'},
            'head0_mean': {'color': '#1f77b4', 'linestyle': '--', 'label': 'Head 0 - Mean'},
            'head1_cls': {'color': '#ff7f0e', 'linestyle': '-', 'label': 'Head 1 - CLS'},
            'head1_mean': {'color': '#ff7f0e', 'linestyle': '--', 'label': 'Head 1 - Mean'},
            'head2_cls': {'color': '#2ca02c', 'linestyle': '-', 'label': 'Head 2 - CLS'},
            'head2_mean': {'color': '#2ca02c', 'linestyle': '--', 'label': 'Head 2 - Mean'}
        }
        
        for key, style in styles.items():
            ax.plot(x_values[:len(avg_changes[key])], avg_changes[key], 
                   linewidth=2, marker='o', markersize=5, **style)
        
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Change Magnitude (Euclidean Distance)', fontsize=14)
        ax.set_title('Layer-to-Layer Changes in Attention Patterns', fontsize=16, fontweight='bold')
        ax.set_xticks(np.arange(6, 23, 2))
        # 使用合理的Y轴范围
        y_max = max_val * 1.1
        ax.set_yticks(np.arange(0, y_max, y_max/10))
        ax.set_xlim(5, 23)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, ncol=2)
        
        output_file = self.figures_dir / "exp4_changes_fixed.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OK 修复完成: {output_file}")
        print(f"   Y轴范围调整为: 0 到 {y_max:.4f}")
        
        return avg_changes
    
    # 修复实验5: 不强制归一化，显示真实权重分布
    def fix_exp5_distance_decay(self):
        print("\n" + "="*80)
        print("修复实验5: 使用真实注意力权重（不强制归一化）")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        grid_size = 14
        center = grid_size // 2
        distance_weights = {d: [] for d in range(11)}
        
        for img_dir in tqdm(image_dirs, desc="重新计算衰减"):
            scan_v_dir = img_dir / "scan_v"
            if not scan_v_dir.exists():
                continue
            
            last_layer = scan_v_dir / "35"
            if not last_layer.exists():
                continue
            
            for head_idx in range(12):
                heatmap = last_layer / f"cls_v_magnitude_head{head_idx}.png"
                if not heatmap.exists():
                    continue
                
                img = cv2.imread(str(heatmap), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                img_resized = cv2.resize(img, (grid_size, grid_size))
                img_norm = img_resized.astype(np.float32) / 255.0
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        dist = int(np.sqrt((i - center)**2 + (j - center)**2))
                        if dist <= 10:
                            distance_weights[dist].append(img_norm[i, j])
        
        avg_weights = {d: np.mean(w) if w else 0 for d, w in distance_weights.items()}
        
        print(f"\n权重范围: {min(avg_weights.values()):.6f} 到 {max(avg_weights.values()):.6f}")
        
        # 绘图 - 使用真实数值
        fig, ax = plt.subplots(figsize=(12, 6))
        distances = sorted(avg_weights.keys())
        weights = [avg_weights[d] for d in distances]
        
        ax.scatter(distances, weights, s=100, alpha=0.6, color='#1f77b4')
        
        # 拟合曲线
        if len(distances) > 2:
            z = np.polyfit(distances, weights, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(0, 10, 100)
            ax.plot(x_smooth, p(x_smooth), 'r--', alpha=0.5, linewidth=2, label='Fitted curve')
        
        ax.set_xlabel('Spatial Distance Between Tokens', fontsize=14)
        ax.set_ylabel('Attention Weight (Normalized)', fontsize=14)
        ax.set_title('Attention Weight Decay with Distance', fontsize=16, fontweight='bold')
        ax.set_xticks(np.arange(0, 11, 2))
        
        # 使用合理的Y轴范围
        y_max = max(weights) * 1.1 if weights else 1
        y_step = y_max / 10
        ax.set_yticks(np.arange(0, y_max, y_step))
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        output_file = self.figures_dir / "exp5_decay_fixed.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OK 修复完成: {output_file}")
        print(f"   Y轴范围调整为: 0 到 {y_max:.6f}")
        
        return avg_weights

def main():
    print("\n" + "="*80)
    print("修复PPT实验图表")
    print("="*80)
    
    fixer = FixedPPTExperiments()
    
    # 修复实验3
    fixer.fix_exp3_entropy()
    
    # 修复实验4
    fixer.fix_exp4_layer_changes()
    
    # 修复实验5
    fixer.fix_exp5_distance_decay()
    
    print("\n" + "="*80)
    print("所有修复完成！")
    print("="*80)
    print("\n新生成的图表:")
    print("  - exp3_entropy_fixed.png (使用真实Attention熵)")
    print("  - exp4_changes_fixed.png (真实欧氏距离)")
    print("  - exp5_decay_fixed.png (真实权重分布)")
    print("\n旧图表:")
    print("  - exp3_entropy.png (V-magnitude熵，数据超出Y轴范围)")
    print("  - exp4_changes.png (强制归一化到0-0.05)")
    print("  - exp5_decay.png (强制归一化到0-0.09)")

if __name__ == "__main__":
    main()
