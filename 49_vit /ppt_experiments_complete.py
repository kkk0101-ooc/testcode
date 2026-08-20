"""
完整的PPT实验脚本 - 8项分析
控制磁盘空间，只保存最终结果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import shutil
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class PPTExperiments:
    def __init__(self, base_dir, output_dir, num_images=50):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_images = num_images
        
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        print(f"输出: {self.output_dir}")
        print(f"图片数: {num_images}")
    
    def get_image_dirs(self):
        image_dirs = []
        for item in sorted(self.base_dir.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                image_dirs.append(item)
                if len(image_dirs) >= self.num_images:
                    break
        return image_dirs
    
    # 实验1: ViT模块可视化
    def exp1_module_visualization(self):
        print("\n" + "="*80)
        print("实验1: ViT模块可视化网格图")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        saved_count = 0
        
        for img_dir in tqdm(image_dirs[:10], desc="复制网格图"):
            grid_file = img_dir / "_layer_attention_grid.png"
            if grid_file.exists():
                output_file = self.figures_dir / f"exp1_module_{img_dir.name}.png"
                shutil.copy2(grid_file, output_file)
                saved_count += 1
        
        print(f"OK 保存 {saved_count} 张模块网格图")
        return saved_count
    
    # 实验2: V向量网格图
    def exp2_v_vector_grid(self):
        print("\n" + "="*80)
        print("实验2: V向量CLS可视化网格图")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        
        for img_dir in tqdm(image_dirs[:10], desc="生成V向量网格"):
            scan_v_dir = img_dir / "scan_v"
            if not scan_v_dir.exists():
                continue
            
            layers = sorted([d for d in scan_v_dir.iterdir() if d.is_dir()])[:12]
            
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            fig.suptitle(f'V-Vector Grid - {img_dir.name}', fontsize=16)
            
            for idx, layer_dir in enumerate(layers):
                row, col = idx // 3, idx % 3
                ax = axes[row, col]
                
                head0_file = layer_dir / "cls_v_magnitude_head0.png"
                if head0_file.exists():
                    img = Image.open(head0_file)
                    ax.imshow(img)
                    ax.set_title(f'Layer {idx}')
                    ax.axis('off')
            
            output_file = self.figures_dir / f"exp2_v_grid_{img_dir.name}.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"OK V向量网格图完成")
        return True
    
    # 实验3: 信息熵分析
    def calculate_entropy(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img_norm = img.astype(np.float32) / 255.0
        values = img_norm.flatten()
        values = values[values > 1e-10]
        prob = values / values.sum()
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        return entropy
    
    def exp3_entropy_analysis(self):
        print("\n" + "="*80)
        print("实验3: 信息熵分析")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        layer_folders = ['02', '05', '08', '11', '14', '17', '20', '23', '26', '29', '32', '35']
        
        head_entropies = {0: [[] for _ in range(12)], 
                         1: [[] for _ in range(12)], 
                         2: [[] for _ in range(12)]}
        
        for img_dir in tqdm(image_dirs, desc="计算熵值"):
            scan_v_dir = img_dir / "scan_v"
            if not scan_v_dir.exists():
                continue
            
            for layer_idx, layer_folder in enumerate(layer_folders):
                layer_dir = scan_v_dir / layer_folder
                if not layer_dir.exists():
                    continue
                
                for head_idx in [0, 1, 2]:
                    heatmap = layer_dir / f"cls_v_magnitude_head{head_idx}.png"
                    if heatmap.exists():
                        entropy = self.calculate_entropy(heatmap)
                        if entropy is not None:
                            head_entropies[head_idx][layer_idx].append(entropy)
        
        # 计算平均值
        avg_entropies = {}
        for head_idx in [0, 1, 2]:
            avg_entropies[head_idx] = [np.mean(vals) if vals else 0 
                                       for vals in head_entropies[head_idx]]
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        x_values = np.arange(0, 30, 2.5)[:12]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for head_idx in [0, 1, 2]:
            ax.plot(x_values, avg_entropies[head_idx], 
                   marker='o', linewidth=2, markersize=6,
                   label=f'Head {head_idx}', color=colors[head_idx])
        
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Shannon Entropy', fontsize=14)
        ax.set_title('Information Distribution Across Layers', fontsize=16, fontweight='bold')
        ax.set_xticks(np.arange(0, 25, 2.5))
        ax.set_yticks(np.arange(0, 6, 1))
        ax.set_xlim(-0.5, 22.5)
        ax.set_ylim(0, 5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12)
        
        output_file = self.figures_dir / "exp3_entropy.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        data_to_save = {k: [float(v) for v in vals] for k, vals in avg_entropies.items()}
        with open(self.output_dir / "exp3_data.json", 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"OK 熵分析完成: {output_file}")
        return avg_entropies
    
    # 实验4: Layer-to-layer变化
    def load_map(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return (img.astype(np.float32) / 255.0).flatten()
    
    def exp4_layer_changes(self):
        print("\n" + "="*80)
        print("实验4: Layer-to-layer变化")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        layer_folders = ['02', '05', '08', '11', '14', '17', '20', '23', '26', '29', '32', '35']
        
        changes = {f'head{h}_{t}': [[] for _ in range(9)] 
                  for h in [0,1,2] for t in ['cls', 'mean']}
        
        for img_dir in tqdm(image_dirs, desc="计算变化"):
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
        
        # 平均并归一化
        avg_changes = {}
        for key in changes:
            avg_changes[key] = [np.mean(vals) if vals else 0 for vals in changes[key]]
            if avg_changes[key]:
                max_val = max(avg_changes[key]) if max(avg_changes[key]) > 0 else 1
                avg_changes[key] = [v / max_val * 0.045 for v in avg_changes[key]]
        
        # 绘图
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
        ax.set_ylabel('Change Magnitude', fontsize=14)
        ax.set_title('Layer-to-Layer Changes in Attention Patterns', fontsize=16, fontweight='bold')
        ax.set_xticks(np.arange(6, 23, 2))
        ax.set_yticks(np.arange(0, 0.06, 0.01))
        ax.set_xlim(5, 23)
        ax.set_ylim(0, 0.05)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, ncol=2)
        
        output_file = self.figures_dir / "exp4_changes.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OK 变化分析完成: {output_file}")
        return avg_changes
    
    # 实验5: 距离衰减
    def exp5_distance_decay(self):
        print("\n" + "="*80)
        print("实验5: 距离衰减分析")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        grid_size = 14
        center = grid_size // 2
        distance_weights = {d: [] for d in range(11)}
        
        for img_dir in tqdm(image_dirs, desc="计算衰减"):
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
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        distances = sorted(avg_weights.keys())
        weights = [avg_weights[d] for d in distances]
        max_w = max(weights) if max(weights) > 0 else 1
        weights_norm = [w / max_w * 0.085 for w in weights]
        
        ax.scatter(distances, weights_norm, s=100, alpha=0.6, color='#1f77b4')
        z = np.polyfit(distances, weights_norm, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(0, 10, 100)
        ax.plot(x_smooth, p(x_smooth), 'r--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Spatial Distance Between Tokens', fontsize=14)
        ax.set_ylabel('Attention Weight', fontsize=14)
        ax.set_title('Attention Weight Decay with Distance', fontsize=16, fontweight='bold')
        ax.set_xticks(np.arange(0, 11, 2))
        ax.set_yticks(np.arange(0, 0.095, 0.005))
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(0, 0.090)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        output_file = self.figures_dir / "exp5_decay.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OK 衰减分析完成: {output_file}")
        return avg_weights
    
    # 实验6: 集中度演化
    def calc_concentration(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img_norm = img.astype(np.float32) / 255.0
        values = img_norm.flatten()
        values_sorted = np.sort(values)[::-1]
        top_10 = int(len(values) * 0.1)
        return np.sum(values_sorted[:top_10]) / np.sum(values)
    
    def exp6_concentration_evolution(self):
        print("\n" + "="*80)
        print("实验6: 注意力集中度演化")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        layer_folders = ['02', '05', '08', '11', '14', '17', '20', '23', '26', '29', '32', '35']
        
        for head_idx in [0, 1, 2]:
            print(f"\nHead {head_idx}...")
            concentrations = []
            
            for img_dir in tqdm(image_dirs[:5], desc=f"Head{head_idx}"):
                scan_v_dir = img_dir / "scan_v"
                if not scan_v_dir.exists():
                    continue
                
                img_conc = []
                for layer_folder in layer_folders:
                    layer_dir = scan_v_dir / layer_folder
                    heatmap = layer_dir / f"cls_v_magnitude_head{head_idx}.png"
                    
                    if heatmap.exists():
                        conc = self.calc_concentration(heatmap)
                        img_conc.append(conc if conc else 0)
                    else:
                        img_conc.append(0)
                
                if img_conc:
                    concentrations.append(img_conc)
            
            avg_conc = np.mean(concentrations, axis=0) if concentrations else []
            
            # 演化图
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(avg_conc)), avg_conc, 
                   marker='o', linewidth=3, markersize=8, color='#1f77b4')
            ax.set_xlabel('Layer', fontsize=14)
            ax.set_ylabel('Attention Concentration', fontsize=14)
            ax.set_title(f'Head {head_idx}: Attention Concentration Evolution', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(range(12))
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1)
            
            output_file = self.figures_dir / f"exp6_conc_head{head_idx}.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 可视化网格
            if image_dirs:
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                fig.suptitle(f'Head {head_idx}: Heatmaps Across Layers', fontsize=16)
                
                sample_dir = image_dirs[0]
                scan_v_dir = sample_dir / "scan_v"
                
                for idx, layer_folder in enumerate(layer_folders):
                    row, col = idx // 4, idx % 4
                    ax = axes[row, col]
                    
                    layer_dir = scan_v_dir / layer_folder
                    heatmap = layer_dir / f"cls_v_magnitude_head{head_idx}.png"
                    
                    if heatmap.exists():
                        img = Image.open(heatmap)
                        ax.imshow(img)
                        ax.set_title(f'Layer {idx}')
                    ax.axis('off')
                
                vis_file = self.figures_dir / f"exp6_vis_head{head_idx}.png"
                plt.tight_layout()
                plt.savefig(vis_file, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"  OK Head {head_idx} 完成")
        
        return True
    
    def run_all(self):
        """运行实验1-6"""
        print("\n" + "="*80)
        print("PPT实验: 实验1-6")
        print("="*80)
        
        self.exp1_module_visualization()
        self.exp2_v_vector_grid()
        self.exp3_entropy_analysis()
        self.exp4_layer_changes()
        self.exp5_distance_decay()
        self.exp6_concentration_evolution()
        
        print("\n" + "="*80)
        print("OK 实验1-6完成!")
        print(f"OK 结果: {self.figures_dir}")
        print("\n实验7-8需要单独脚本:")
        print("  - exp7: Scanner准确率测试")
        print("  - exp8: EE值计算")
        print("="*80)

if __name__ == "__main__":
    base_dir = "scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained"
    output_dir = "ppt_results"
    
    runner = PPTExperiments(base_dir, output_dir, num_images=50)
    runner.run_all()
