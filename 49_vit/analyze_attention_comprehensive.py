"""
全面的注意力分析脚本
分析所有层的注意力特征，包括统计指标、多头分析、层间演化等
"""
import os
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

class AttentionAnalyzer:
    def __init__(self, base_dir, output_dir):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储分析结果
        self.layer_stats = {}
        self.head_stats = {}
        
    def load_attention_map(self, layer_idx, head_idx, attn_type="avg_cls_attn"):
        """加载注意力图"""
        attn_path = self.base_dir / str(layer_idx) / "input_attn" / f"head{head_idx}" / f"{attn_type}_head{head_idx}.png"
        
        if not attn_path.exists():
            return None
            
        # 加载图像并转换为灰度
        img = cv2.imread(str(attn_path))
        if img is None:
            return None
        
        # 转换为灰度并归一化到 [0, 1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        
        return normalized
    
    def compute_statistics(self, attn_map):
        """计算注意力图的统计指标"""
        if attn_map is None:
            return None
        
        stats = {
            'mean': float(np.mean(attn_map)),
            'std': float(np.std(attn_map)),
            'min': float(np.min(attn_map)),
            'max': float(np.max(attn_map)),
            'median': float(np.median(attn_map)),
            'entropy': self.compute_entropy(attn_map),
            'sparsity': self.compute_sparsity(attn_map),
            'cv': float(np.std(attn_map) / (np.mean(attn_map) + 1e-8)),  # 变异系数
        }
        
        # 计算百分位数
        stats['p25'] = float(np.percentile(attn_map, 25))
        stats['p75'] = float(np.percentile(attn_map, 75))
        stats['p95'] = float(np.percentile(attn_map, 95))
        
        return stats
    
    def compute_entropy(self, attn_map):
        """计算注意力图的熵（衡量分布均匀程度）"""
        # 将注意力值作为概率分布
        attn_flat = attn_map.flatten()
        attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
        
        # 计算熵
        entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8))
        return float(entropy)
    
    def compute_sparsity(self, attn_map, threshold=0.1):
        """计算注意力的稀疏性（有多少区域被激活）"""
        # 计算高于阈值的像素比例
        active_ratio = np.mean(attn_map > threshold)
        return float(active_ratio)
    
    def analyze_all_layers(self, num_heads=3):
        """分析所有层的注意力"""
        print("\n[*] 分析所有层的注意力特征...")
        
        # 获取所有层目录
        layer_dirs = sorted([d for d in self.base_dir.iterdir() if d.is_dir() and d.name.isdigit()], 
                           key=lambda x: int(x.name))
        
        print(f"找到 {len(layer_dirs)} 个层")
        
        for layer_dir in layer_dirs:
            layer_idx = int(layer_dir.name)
            print(f"\n  Layer {layer_idx}:")
            
            layer_heads_stats = []
            
            for head_idx in range(num_heads):
                # 加载 avg_cls_attn
                attn_map = self.load_attention_map(layer_idx, head_idx, "avg_cls_attn")
                
                if attn_map is not None:
                    stats = self.compute_statistics(attn_map)
                    stats['layer'] = layer_idx
                    stats['head'] = head_idx
                    
                    layer_heads_stats.append(stats)
                    
                    print(f"    Head {head_idx}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                          f"entropy={stats['entropy']:.4f}, sparsity={stats['sparsity']:.4f}, cv={stats['cv']:.4f}")
            
            if layer_heads_stats:
                # 计算该层所有头的平均统计
                avg_stats = {
                    'layer': layer_idx,
                    'mean': np.mean([s['mean'] for s in layer_heads_stats]),
                    'std': np.mean([s['std'] for s in layer_heads_stats]),
                    'entropy': np.mean([s['entropy'] for s in layer_heads_stats]),
                    'sparsity': np.mean([s['sparsity'] for s in layer_heads_stats]),
                    'cv': np.mean([s['cv'] for s in layer_heads_stats]),
                    'heads': layer_heads_stats
                }
                
                self.layer_stats[layer_idx] = avg_stats
    
    def visualize_layer_evolution(self):
        """可视化层间演化趋势"""
        print("\n[*] 生成层间演化趋势图...")
        
        if not self.layer_stats:
            print("  ⚠️ 没有统计数据可供可视化")
            return
        
        layers = sorted(self.layer_stats.keys())
        
        metrics = {
            'mean': [self.layer_stats[l]['mean'] for l in layers],
            'std': [self.layer_stats[l]['std'] for l in layers],
            'entropy': [self.layer_stats[l]['entropy'] for l in layers],
            'sparsity': [self.layer_stats[l]['sparsity'] for l in layers],
            'cv': [self.layer_stats[l]['cv'] for l in layers],
        }
        
        # 创建多子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Attention Evolution Across Layers', fontsize=16, fontweight='bold')
        
        # 绘制每个指标
        ax = axes[0, 0]
        ax.plot(layers, metrics['mean'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Mean Attention', fontsize=12)
        ax.set_title('Mean Attention Value', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(layers, metrics['std'], 'o-', linewidth=2, markersize=8, color='#A23B72')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Std Attention', fontsize=12)
        ax.set_title('Standard Deviation', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.plot(layers, metrics['entropy'], 'o-', linewidth=2, markersize=8, color='#F18F01')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Entropy', fontsize=12)
        ax.set_title('Attention Entropy (Uniformity)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(layers, metrics['sparsity'], 'o-', linewidth=2, markersize=8, color='#C73E1D')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Sparsity', fontsize=12)
        ax.set_title('Attention Sparsity (Active Ratio)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(layers, metrics['cv'], 'o-', linewidth=2, markersize=8, color='#6A994E')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Coefficient of Variation', fontsize=12)
        ax.set_title('CV (Std/Mean)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 最后一个子图：综合对比
        ax = axes[1, 2]
        # 归一化所有指标到 [0, 1]
        for metric_name, values in metrics.items():
            values_norm = (np.array(values) - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
            ax.plot(layers, values_norm, 'o-', linewidth=2, markersize=6, label=metric_name, alpha=0.7)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.set_title('All Metrics (Normalized)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'layer_evolution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 层间演化图已保存: {output_path}")
    
    def analyze_multi_head(self, layer_idx, num_heads=3):
        """分析特定层的多头注意力"""
        print(f"\n[*] 分析 Layer {layer_idx} 的多头注意力...")
        
        head_maps = []
        head_stats = []
        
        for head_idx in range(num_heads):
            attn_map = self.load_attention_map(layer_idx, head_idx, "avg_cls_attn")
            if attn_map is not None:
                head_maps.append(attn_map)
                stats = self.compute_statistics(attn_map)
                stats['head'] = head_idx
                head_stats.append(stats)
                
                print(f"  Head {head_idx}: mean={stats['mean']:.4f}, entropy={stats['entropy']:.4f}")
        
        if not head_maps:
            print("  ⚠️ 没有找到注意力图")
            return
        
        # 可视化多头注意力
        self.visualize_multi_head(layer_idx, head_maps, head_stats)
    
    def visualize_multi_head(self, layer_idx, head_maps, head_stats):
        """可视化多头注意力对比"""
        num_heads = len(head_maps)
        
        fig, axes = plt.subplots(2, num_heads, figsize=(num_heads * 5, 10))
        fig.suptitle(f'Multi-Head Attention Analysis - Layer {layer_idx}', fontsize=16, fontweight='bold')
        
        # 第一行：热力图
        for i, (attn_map, stats) in enumerate(zip(head_maps, head_stats)):
            ax = axes[0, i] if num_heads > 1 else axes[0]
            
            # 应用 colormap
            colored = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
            ax.imshow(colored_rgb)
            ax.set_title(f"Head {i}\nEntropy: {stats['entropy']:.3f}", fontsize=12)
            ax.axis('off')
        
        # 第二行：直方图
        for i, (attn_map, stats) in enumerate(zip(head_maps, head_stats)):
            ax = axes[1, i] if num_heads > 1 else axes[1]
            
            ax.hist(attn_map.flatten(), bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.3f}")
            ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.3f}")
            ax.set_xlabel('Attention Value', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f"Head {i} Distribution", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f'layer_{layer_idx:02d}_multihead_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 多头分析图已保存: {output_path}")
    
    def analyze_last_layer_optimization(self, last_layer_idx=46):
        """分析最后一层的特征，提供优化建议"""
        print(f"\n[*] 分析最后一层 (Layer {last_layer_idx}) 的优化方向...")
        
        if last_layer_idx not in self.layer_stats:
            print(f"  ⚠️ Layer {last_layer_idx} 数据不存在")
            return
        
        last_stats = self.layer_stats[last_layer_idx]
        
        print(f"\n  当前最后一层统计:")
        print(f"    Mean: {last_stats['mean']:.4f}")
        print(f"    Std: {last_stats['std']:.4f}")
        print(f"    Entropy: {last_stats['entropy']:.4f}")
        print(f"    Sparsity: {last_stats['sparsity']:.4f}")
        print(f"    CV: {last_stats['cv']:.4f}")
        
        # 优化建议
        print(f"\n  [建议] 优化建议:")
        
        if last_stats['cv'] < 0.1:
            print(f"    ⚠️ 变异系数过低 (CV={last_stats['cv']:.4f} < 0.1)")
            print(f"       建议：使用更激进的归一化（百分位裁剪）或增强对比度")
        
        if last_stats['entropy'] > 5.0:
            print(f"    ⚠️ 熵过高 (Entropy={last_stats['entropy']:.4f} > 5.0)")
            print(f"       建议：注意力分布太均匀，需要更强的聚焦机制")
        
        if last_stats['sparsity'] > 0.8:
            print(f"    ⚠️ 稀疏性过高 (Sparsity={last_stats['sparsity']:.4f} > 0.8)")
            print(f"       建议：注意力过于分散，需要更强的选择性")
        
        # 对比第一层
        first_layer_idx = min(self.layer_stats.keys())
        first_stats = self.layer_stats[first_layer_idx]
        
        print(f"\n  [对比] 与第一层 (Layer {first_layer_idx}) 对比:")
        print(f"    Mean: {first_stats['mean']:.4f} -> {last_stats['mean']:.4f} "
              f"({'↑' if last_stats['mean'] > first_stats['mean'] else '↓'} {abs(last_stats['mean'] - first_stats['mean']):.4f})")
        print(f"    Entropy: {first_stats['entropy']:.4f} -> {last_stats['entropy']:.4f} "
              f"({'↑' if last_stats['entropy'] > first_stats['entropy'] else '↓'} {abs(last_stats['entropy'] - first_stats['entropy']):.4f})")
        print(f"    Sparsity: {first_stats['sparsity']:.4f} -> {last_stats['sparsity']:.4f} "
              f"({'↑' if last_stats['sparsity'] > first_stats['sparsity'] else '↓'} {abs(last_stats['sparsity'] - first_stats['sparsity']):.4f})")
    
    def save_statistics_json(self):
        """保存统计结果到JSON文件"""
        output_path = self.output_dir / 'attention_statistics.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.layer_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n[保存] 统计数据已保存: {output_path}")
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n[*] 生成综合分析报告...")
        
        report_path = self.output_dir / 'comprehensive_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Vision Transformer Attention Comprehensive Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Layers Analyzed: {len(self.layer_stats)}\n\n")
            
            # 总体趋势
            f.write("=" * 80 + "\n")
            f.write("1. Overall Trends\n")
            f.write("=" * 80 + "\n\n")
            
            layers = sorted(self.layer_stats.keys())
            
            # 计算趋势
            mean_trend = [self.layer_stats[l]['mean'] for l in layers]
            entropy_trend = [self.layer_stats[l]['entropy'] for l in layers]
            sparsity_trend = [self.layer_stats[l]['sparsity'] for l in layers]
            
            f.write(f"Mean Attention:\n")
            f.write(f"  First Layer: {mean_trend[0]:.4f}\n")
            f.write(f"  Last Layer: {mean_trend[-1]:.4f}\n")
            f.write(f"  Change: {mean_trend[-1] - mean_trend[0]:.4f}\n\n")
            
            f.write(f"Entropy:\n")
            f.write(f"  First Layer: {entropy_trend[0]:.4f}\n")
            f.write(f"  Last Layer: {entropy_trend[-1]:.4f}\n")
            f.write(f"  Change: {entropy_trend[-1] - entropy_trend[0]:.4f}\n\n")
            
            f.write(f"Sparsity:\n")
            f.write(f"  First Layer: {sparsity_trend[0]:.4f}\n")
            f.write(f"  Last Layer: {sparsity_trend[-1]:.4f}\n")
            f.write(f"  Change: {sparsity_trend[-1] - sparsity_trend[0]:.4f}\n\n")
            
            # 关键层分析
            f.write("=" * 80 + "\n")
            f.write("2. Key Layers Analysis\n")
            f.write("=" * 80 + "\n\n")
            
            for layer_idx in [layers[0], layers[len(layers)//4], layers[len(layers)//2], layers[3*len(layers)//4], layers[-1]]:
                stats = self.layer_stats[layer_idx]
                f.write(f"Layer {layer_idx}:\n")
                f.write(f"  Mean: {stats['mean']:.4f}\n")
                f.write(f"  Std: {stats['std']:.4f}\n")
                f.write(f"  Entropy: {stats['entropy']:.4f}\n")
                f.write(f"  Sparsity: {stats['sparsity']:.4f}\n")
                f.write(f"  CV: {stats['cv']:.4f}\n\n")
            
            # 优化建议
            f.write("=" * 80 + "\n")
            f.write("3. Optimization Recommendations\n")
            f.write("=" * 80 + "\n\n")
            
            last_stats = self.layer_stats[layers[-1]]
            
            f.write("For Last Layer:\n")
            if last_stats['cv'] < 0.1:
                f.write("  - Use percentile-based normalization (CV is too low)\n")
            if last_stats['entropy'] > 5.0:
                f.write("  - Attention is too uniform, consider stronger focusing\n")
            if last_stats['sparsity'] > 0.8:
                f.write("  - Attention is too sparse, need better selectivity\n")
            
            f.write("\nFor Visualization:\n")
            f.write("  - Use max pooling + bicubic interpolation for better fineness\n")
            f.write("  - Apply class-guided attention for better shape recognition\n")
            f.write("  - Use adaptive normalization based on CV value\n")
        
        print(f"  ✓ 报告已保存: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='全面的注意力分析')
    parser.add_argument('--base_dir', type=str, 
                       default=r'D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final\0_n01443537_24scan_attn_map',
                       help='注意力图基础目录')
    parser.add_argument('--output_dir', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\attention_analysis',
                       help='输出目录')
    parser.add_argument('--num_heads', type=int, default=3, help='注意力头数量')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Vision Transformer Attention Comprehensive Analysis")
    print("=" * 80)
    
    analyzer = AttentionAnalyzer(args.base_dir, args.output_dir)
    
    # 1. 分析所有层
    analyzer.analyze_all_layers(num_heads=args.num_heads)
    
    # 2. 可视化层间演化
    analyzer.visualize_layer_evolution()
    
    # 3. 多头分析（选择几个关键层）
    if analyzer.layer_stats:
        layers = sorted(analyzer.layer_stats.keys())
        key_layers = [layers[0], layers[len(layers)//4], layers[len(layers)//2], layers[-1]]
        
        for layer_idx in key_layers:
            analyzer.analyze_multi_head(layer_idx, num_heads=args.num_heads)
    
    # 4. 最后一层优化分析
    analyzer.analyze_last_layer_optimization()
    
    # 5. 保存统计数据
    analyzer.save_statistics_json()
    
    # 6. 生成综合报告
    analyzer.generate_comprehensive_report()
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print(f"所有结果已保存到: {args.output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()

