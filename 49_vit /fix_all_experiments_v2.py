"""
全面修复所有PPT实验的问题
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import cv2
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from PIL import Image

class CompleteFix:
    def __init__(self):
        self.scan_dir = Path("scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained")
        self.output_dir = Path("ppt_results")
        self.figures_dir = self.output_dir / "figures_v2"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def get_image_dirs(self, num_images=50):
        image_dirs = sorted([d for d in self.scan_dir.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')])[:num_images]
        return image_dirs
    
    # =====================================================================
    # 修复实验3: 调整Y轴范围以适应真实熵值
    # =====================================================================
    def fix_exp3_entropy_v2(self):
        print("\n" + "="*80)
        print("修复实验3: 调整Y轴范围以适应真实熵值")
        print("="*80)
        
        entropy_file = self.output_dir / "real_attention_entropy.json"
        if not entropy_file.exists():
            print("错误: 未找到真实熵值数据")
            return
        
        with open(entropy_file, 'r') as f:
            real_data = json.load(f)
        
        # 找到数据范围
        all_values = []
        for head_idx in [0, 1, 2]:
            all_values.extend(real_data[f'head{head_idx}'])
        
        min_val = min(all_values)
        max_val = max(all_values)
        
        print(f"熵值范围: {min_val:.2f} 到 {max_val:.2f} bits")
        
        # 绘图 - 调整Y轴范围
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
        ax.set_yticks(np.arange(0, 8, 1))  # 扩大到0-8
        ax.set_xlim(-0.5, 27.5)
        ax.set_ylim(0, 8)  # 调整Y轴上限到8
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12)
        
        output_file = self.figures_dir / "exp3_entropy_v2.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OK 修复完成: {output_file}")
        print(f"   Y轴范围调整为: 0 到 8 bits")
    
    # =====================================================================
    # 修复实验5: 使用Attention权重而非V-magnitude
    # =====================================================================
    def fix_exp5_distance_decay_v2(self):
        print("\n" + "="*80)
        print("修复实验5: 分析真实的CLS Attention的距离衰减")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        grid_size = 14  # ViT-Base patch grid is 14x14
        center = grid_size // 2
        
        # 存储每个距离的attention权重
        distance_weights = {d: [] for d in range(11)}
        
        for img_dir in tqdm(image_dirs, desc="分析CLS attention"):
            # 查找layer 11的attention maps (最后一层)
            attn_map_dir = img_dir / "scan_attn_map" / "35" / "input_attn"
            if not attn_map_dir.exists():
                continue
            
            # 使用head 0-2
            for head_idx in range(3):
                head_dir = attn_map_dir / f"head{head_idx}"
                cls_attn_file = head_dir / f"avg_cls_attn_head{head_idx}.png"
                
                if not cls_attn_file.exists():
                    continue
                
                # 读取CLS attention heatmap
                img = cv2.imread(str(cls_attn_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Resize到14x14
                img_resized = cv2.resize(img, (grid_size, grid_size))
                img_norm = img_resized.astype(np.float32) / 255.0
                
                # 计算每个位置到中心的距离
                for i in range(grid_size):
                    for j in range(grid_size):
                        dist = int(np.sqrt((i - center)**2 + (j - center)**2))
                        if dist <= 10:
                            distance_weights[dist].append(img_norm[i, j])
        
        # 计算平均
        avg_weights = {}
        for d in range(11):
            if distance_weights[d]:
                avg_weights[d] = np.mean(distance_weights[d])
            else:
                avg_weights[d] = 0
        
        print(f"\n权重统计:")
        print(f"  最小值: {min(avg_weights.values()):.6f}")
        print(f"  最大值: {max(avg_weights.values()):.6f}")
        print(f"  距离0 (中心): {avg_weights.get(0, 0):.6f}")
        print(f"  距离10 (边缘): {avg_weights.get(10, 0):.6f}")
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        distances = sorted(avg_weights.keys())
        weights = [avg_weights[d] for d in distances]
        
        # 散点图
        ax.scatter(distances, weights, s=150, alpha=0.7, color='#1f77b4', 
                  edgecolors='black', linewidth=1.5, zorder=3)
        
        # 拟合曲线
        if len([w for w in weights if w > 0]) > 2:
            valid_points = [(d, w) for d, w in zip(distances, weights) if w > 0]
            if valid_points:
                d_valid, w_valid = zip(*valid_points)
                z = np.polyfit(d_valid, w_valid, 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(d_valid), max(d_valid), 100)
                ax.plot(x_smooth, p(x_smooth), 'r--', alpha=0.6, linewidth=2.5, 
                       label='Polynomial fit', zorder=2)
        
        ax.set_xlabel('Spatial Distance Between Tokens', fontsize=14)
        ax.set_ylabel('Attention Weight', fontsize=14)
        ax.set_title('Attention Weight Decay with Distance', fontsize=16, fontweight='bold')
        ax.set_xticks(np.arange(0, 11, 2))
        
        # 自动调整Y轴范围
        y_max = max(weights) * 1.15 if weights else 0.1
        y_step = y_max / 10
        ax.set_yticks(np.arange(0, y_max, y_step))
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
        ax.legend(fontsize=12)
        
        # 添加说明文字
        ax.text(0.98, 0.02, f'Mean weight: {np.mean(weights):.4f}\nStd: {np.std(weights):.4f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        
        output_file = self.figures_dir / "exp5_decay_v2.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OK 修复完成: {output_file}")
        print(f"   Y轴范围: 0 到 {y_max:.4f}")
    
    # =====================================================================
    # 修复实验6: 增加样本数量，改进可视化
    # =====================================================================
    def calc_concentration(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img_norm = img.astype(np.float32) / 255.0
        values = img_norm.flatten()
        values_sorted = np.sort(values)[::-1]
        top_10 = int(len(values) * 0.1)
        concentration = np.sum(values_sorted[:top_10]) / (np.sum(values) + 1e-10)
        return concentration
    
    def fix_exp6_concentration_v2(self):
        print("\n" + "="*80)
        print("修复实验6: 增加样本，改进可视化")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        layer_folders = ['02', '05', '08', '11', '14', '17', '20', '23', '26', '29', '32', '35']
        
        for head_idx in [0, 1, 2]:
            print(f"\nHead {head_idx}...")
            concentrations = []
            
            # 增加样本数量到30
            for img_dir in tqdm(image_dirs[:30], desc=f"Head{head_idx}", leave=False):
                scan_v_dir = img_dir / "scan_v"
                if not scan_v_dir.exists():
                    continue
                
                img_conc = []
                for layer_folder in layer_folders:
                    layer_dir = scan_v_dir / layer_folder
                    heatmap = layer_dir / f"cls_v_magnitude_head{head_idx}.png"
                    
                    if heatmap.exists():
                        conc = self.calc_concentration(heatmap)
                        img_conc.append(conc if conc is not None else 0)
                    else:
                        img_conc.append(0)
                
                if img_conc and max(img_conc) > 0:
                    concentrations.append(img_conc)
            
            if not concentrations:
                print(f"  警告: Head {head_idx} 无有效数据")
                continue
            
            avg_conc = np.mean(concentrations, axis=0)
            std_conc = np.std(concentrations, axis=0)
            
            print(f"  集中度范围: {min(avg_conc):.4f} 到 {max(avg_conc):.4f}")
            
            # 演化图
            fig, ax = plt.subplots(figsize=(12, 6))
            x_vals = range(len(avg_conc))
            
            # 绘制均值线和标准差区域
            ax.plot(x_vals, avg_conc, marker='o', linewidth=3, 
                   markersize=8, color='#1f77b4', label='Mean concentration')
            ax.fill_between(x_vals, avg_conc - std_conc, avg_conc + std_conc,
                          alpha=0.3, color='#1f77b4', label='±1 std')
            
            ax.set_xlabel('Layer', fontsize=14)
            ax.set_ylabel('Attention Concentration', fontsize=14)
            ax.set_title(f'Head {head_idx}: Attention Concentration Evolution', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(range(12))
            ax.set_xticklabels([f'L{i}' for i in range(12)])
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=11)
            
            output_file = self.figures_dir / f"exp6_conc_head{head_idx}_v2.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 改进的可视化网格 - 使用热力图而非直接显示PNG
            if image_dirs:
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                fig.suptitle(f'Head {head_idx}: Heatmaps Across Layers', fontsize=16, fontweight='bold')
                
                sample_dir = image_dirs[0]
                scan_v_dir = sample_dir / "scan_v"
                
                for idx, layer_folder in enumerate(layer_folders):
                    row, col = idx // 4, idx % 4
                    ax = axes[row, col]
                    
                    layer_dir = scan_v_dir / layer_folder
                    heatmap = layer_dir / f"cls_v_magnitude_head{head_idx}.png"
                    
                    if heatmap.exists():
                        img = cv2.imread(str(heatmap), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # 使用热力图显示，而非直接显示PNG
                            im = ax.imshow(img, cmap='hot', interpolation='bilinear')
                            ax.set_title(f'Layer {idx} (Conc: {avg_conc[idx]:.3f})', 
                                       fontsize=11, fontweight='bold')
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.axis('off')
                
                vis_file = self.figures_dir / f"exp6_vis_head{head_idx}_v2.png"
                plt.tight_layout()
                plt.savefig(vis_file, dpi=200, bbox_inches='tight')
                plt.close()
            
            print(f"  OK Head {head_idx} 完成")
    
    # =====================================================================
    # 修复实验1、2: 添加原图到网格
    # =====================================================================
    def fix_exp1_with_original(self):
        print("\n" + "="*80)
        print("修复实验1: 网格图添加原图")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        layer_folders = ['02', '05', '08', '11', '14', '17', '20', '23', '26', '29', '32', '35']
        
        saved_count = 0
        
        for img_dir in tqdm(image_dirs[:10], desc="生成模块网格"):
            # 找原图
            tem0_dir = img_dir / "tem0"
            if not tem0_dir.exists():
                continue
            
            img_files = list(tem0_dir.glob("*.JPEG")) + list(tem0_dir.glob("*.jpg"))
            if not img_files:
                continue
            
            original_img = Image.open(img_files[0]).convert('RGB')
            original_img = original_img.resize((224, 224))
            
            # 创建网格: 1行原图 + 4行×3列=12个layer
            fig = plt.figure(figsize=(18, 20))
            gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.2)
            
            # 第一行: 原图（占据全部3列）
            ax_orig = fig.add_subplot(gs[0, :])
            ax_orig.imshow(original_img)
            ax_orig.set_title('Original Image', fontsize=16, fontweight='bold')
            ax_orig.axis('off')
            
            # 后4行: 12个layer的网格图
            layer_grid_file = img_dir / f"{img_dir.name}_layer_attention_grid.png"
            
            if layer_grid_file.exists():
                for idx in range(12):
                    row = (idx // 3) + 1  # +1因为第0行是原图
                    col = idx % 3
                    
                    ax = fig.add_subplot(gs[row, col])
                    
                    # 读取对应layer的可视化
                    scan_rgb_dir = img_dir / "scan_RGB"
                    layer_rgb = scan_rgb_dir / layer_folders[idx] / "RGB.png"
                    
                    if layer_rgb.exists():
                        layer_img = Image.open(layer_rgb)
                        ax.imshow(layer_img)
                        ax.set_title(f'Layer {idx}', fontsize=12, fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                    
                    ax.axis('off')
            
            output_file = self.figures_dir / f"exp1_module_{img_dir.name}_v2.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            saved_count += 1
        
        print(f"OK 保存 {saved_count} 张模块网格图 (带原图)")
    
    def fix_exp2_with_original(self):
        print("\n" + "="*80)
        print("修复实验2: V向量网格图添加原图")
        print("="*80)
        
        image_dirs = self.get_image_dirs()
        layer_folders = ['02', '05', '08', '11', '14', '17', '20', '23', '26', '29', '32', '35']
        
        saved_count = 0
        
        for img_dir in tqdm(image_dirs[:10], desc="生成V向量网格"):
            # 找原图
            tem0_dir = img_dir / "tem0"
            if not tem0_dir.exists():
                continue
            
            img_files = list(tem0_dir.glob("*.JPEG")) + list(tem0_dir.glob("*.jpg"))
            if not img_files:
                continue
            
            original_img = Image.open(img_files[0]).convert('RGB')
            original_img = original_img.resize((224, 224))
            
            scan_v_dir = img_dir / "scan_v"
            if not scan_v_dir.exists():
                continue
            
            # 创建网格: 1行原图 + 4行×3列=12个layer (head 0的CLS V-magnitude)
            fig = plt.figure(figsize=(18, 20))
            gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.2)
            
            # 第一行: 原图
            ax_orig = fig.add_subplot(gs[0, :])
            ax_orig.imshow(original_img)
            ax_orig.set_title('Original Image', fontsize=16, fontweight='bold')
            ax_orig.axis('off')
            
            # 后4行: 12个layer的V-magnitude (head 0)
            for idx, layer_folder in enumerate(layer_folders):
                row = (idx // 3) + 1
                col = idx % 3
                
                ax = fig.add_subplot(gs[row, col])
                
                layer_dir = scan_v_dir / layer_folder
                v_file = layer_dir / "cls_v_magnitude_head0.png"
                
                if v_file.exists():
                    v_img = cv2.imread(str(v_file))
                    if v_img is not None:
                        v_img_rgb = cv2.cvtColor(v_img, cv2.COLOR_BGR2RGB)
                        ax.imshow(v_img_rgb)
                        ax.set_title(f'Layer {idx} - V mag (H0)', fontsize=11)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                
                ax.axis('off')
            
            output_file = self.figures_dir / f"exp2_v_grid_{img_dir.name}_v2.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            saved_count += 1
        
        print(f"OK 保存 {saved_count} 张V向量网格图 (带原图)")
    
    def run_all_fixes(self):
        print("\n" + "="*80)
        print("全面修复所有PPT实验")
        print("="*80)
        
        self.fix_exp3_entropy_v2()
        self.fix_exp5_distance_decay_v2()
        self.fix_exp6_concentration_v2()
        self.fix_exp1_with_original()
        self.fix_exp2_with_original()
        
        print("\n" + "="*80)
        print("所有修复完成！")
        print("="*80)
        print(f"\n新生成的图表保存在: {self.figures_dir}")

def main():
    fixer = CompleteFix()
    fixer.run_all_fixes()

if __name__ == "__main__":
    main()
