"""
Tiny-ImageNet vs ImageNet 完整对比分析
对比两个数据集上的ViT性能和注意力模式
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

class TinyVsImageNetComparator:
    def __init__(self, tiny_metrics_file, imagenet_metrics_file, output_dir="analysis_results/comparison"):
        """
        初始化对比分析器
        
        参数:
            tiny_metrics_file: Tiny-ImageNet的metrics JSON文件
            imagenet_metrics_file: ImageNet的metrics JSON文件
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        print("加载Tiny-ImageNet指标...")
        with open(tiny_metrics_file, 'r', encoding='utf-8') as f:
            self.tiny_data = json.load(f)
        
        print("加载ImageNet指标...")
        with open(imagenet_metrics_file, 'r', encoding='utf-8') as f:
            self.imagenet_data = json.load(f)
        
        print(f"Tiny-ImageNet: {len(self.tiny_data)} 张图片")
        print(f"ImageNet: {len(self.imagenet_data)} 张图片")
    
    def extract_metrics(self, data):
        """从数据中提取所有指标"""
        metrics = {
            'entropy': [],
            'peak_activation': [],
            'concentration_index': [],
            'head_diversity': [],
            'layer_entropy': {},  # 按层统计
            'layer_peak': {},
            'layer_concentration': {},
        }
        
        for result in data:
            for layer, layer_data in result['layers'].items():
                # 初始化层级列表
                if layer not in metrics['layer_entropy']:
                    metrics['layer_entropy'][layer] = []
                    metrics['layer_peak'][layer] = []
                    metrics['layer_concentration'][layer] = []
                
                # 收集每个头的指标
                for head, head_metrics in layer_data['heads'].items():
                    metrics['entropy'].append(head_metrics['entropy'])
                    metrics['peak_activation'].append(head_metrics['peak_activation'])
                    metrics['concentration_index'].append(head_metrics['concentration_index'])
                    
                    # 按层收集
                    metrics['layer_entropy'][layer].append(head_metrics['entropy'])
                    metrics['layer_peak'][layer].append(head_metrics['peak_activation'])
                    metrics['layer_concentration'][layer].append(head_metrics['concentration_index'])
                
                # 收集层级指标
                if 'layer_metrics' in layer_data and 'head_diversity' in layer_data['layer_metrics']:
                    metrics['head_diversity'].append(layer_data['layer_metrics']['head_diversity'])
        
        return metrics
    
    def compare_overall_metrics(self):
        """对比整体指标"""
        print("\n对比整体指标...")
        
        tiny_metrics = self.extract_metrics(self.tiny_data)
        imagenet_metrics = self.extract_metrics(self.imagenet_data)
        
        # 创建对比表
        comparison = {
            'Entropy': {
                'Tiny-ImageNet': {
                    'mean': np.mean(tiny_metrics['entropy']),
                    'std': np.std(tiny_metrics['entropy']),
                },
                'ImageNet': {
                    'mean': np.mean(imagenet_metrics['entropy']),
                    'std': np.std(imagenet_metrics['entropy']),
                }
            },
            'Peak Activation': {
                'Tiny-ImageNet': {
                    'mean': np.mean(tiny_metrics['peak_activation']),
                    'std': np.std(tiny_metrics['peak_activation']),
                },
                'ImageNet': {
                    'mean': np.mean(imagenet_metrics['peak_activation']),
                    'std': np.std(imagenet_metrics['peak_activation']),
                }
            },
            'Concentration Index': {
                'Tiny-ImageNet': {
                    'mean': np.mean(tiny_metrics['concentration_index']),
                    'std': np.std(tiny_metrics['concentration_index']),
                },
                'ImageNet': {
                    'mean': np.mean(imagenet_metrics['concentration_index']),
                    'std': np.std(imagenet_metrics['concentration_index']),
                }
            },
            'Head Diversity': {
                'Tiny-ImageNet': {
                    'mean': np.mean(tiny_metrics['head_diversity']),
                    'std': np.std(tiny_metrics['head_diversity']),
                },
                'ImageNet': {
                    'mean': np.mean(imagenet_metrics['head_diversity']),
                    'std': np.std(imagenet_metrics['head_diversity']),
                }
            }
        }
        
        # 保存对比结果
        output_file = self.output_dir / "overall_comparison.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 整体对比已保存到: {output_file}")
        
        return comparison, tiny_metrics, imagenet_metrics
    
    def visualize_metric_comparison(self, tiny_metrics, imagenet_metrics):
        """可视化指标对比"""
        print("\n生成对比可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Tiny-ImageNet vs ImageNet: Metrics Comparison', fontsize=18, fontweight='bold')
        
        metrics_to_plot = [
            ('entropy', 'Entropy', axes[0, 0]),
            ('peak_activation', 'Peak Activation', axes[0, 1]),
            ('concentration_index', 'Concentration Index', axes[1, 0]),
            ('head_diversity', 'Head Diversity', axes[1, 1])
        ]
        
        for metric_key, metric_name, ax in metrics_to_plot:
            tiny_vals = tiny_metrics[metric_key]
            imagenet_vals = imagenet_metrics[metric_key]
            
            # 箱线图对比
            data_to_plot = [tiny_vals, imagenet_vals]
            bp = ax.boxplot(data_to_plot, labels=['Tiny-ImageNet', 'ImageNet'],
                           patch_artist=True, showmeans=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加统计信息
            tiny_mean = np.mean(tiny_vals)
            imagenet_mean = np.mean(imagenet_vals)
            diff_pct = ((imagenet_mean - tiny_mean) / tiny_mean) * 100
            
            ax.text(0.5, 0.95, f'Difference: {diff_pct:+.1f}%',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_file = self.output_dir / "metrics_comparison_boxplot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 对比箱线图已保存到: {output_file}")
    
    def visualize_layer_comparison(self, tiny_metrics, imagenet_metrics):
        """对比层级演化"""
        print("\n生成层级演化对比...")
        
        # 计算每层的平均值
        tiny_layers = sorted(tiny_metrics['layer_entropy'].keys(), key=lambda x: int(x))
        imagenet_layers = sorted(imagenet_metrics['layer_entropy'].keys(), key=lambda x: int(x))
        
        # 确保层数一致（取较小的）
        common_layers = min(len(tiny_layers), len(imagenet_layers))
        tiny_layers = tiny_layers[:common_layers]
        imagenet_layers = imagenet_layers[:common_layers]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Tiny-ImageNet vs ImageNet: Layer Evolution Comparison', fontsize=18, fontweight='bold')
        
        layer_nums = [int(l) for l in tiny_layers]
        
        # Entropy演化对比
        tiny_entropy = [np.mean(tiny_metrics['layer_entropy'][l]) for l in tiny_layers]
        imagenet_entropy = [np.mean(imagenet_metrics['layer_entropy'][l]) for l in imagenet_layers]
        
        axes[0, 0].plot(layer_nums, tiny_entropy, 'o-', label='Tiny-ImageNet', linewidth=2, markersize=8)
        axes[0, 0].plot(layer_nums, imagenet_entropy, 's-', label='ImageNet', linewidth=2, markersize=8)
        axes[0, 0].set_title('Entropy Evolution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Average Entropy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Peak Activation演化对比
        tiny_peak = [np.mean(tiny_metrics['layer_peak'][l]) for l in tiny_layers]
        imagenet_peak = [np.mean(imagenet_metrics['layer_peak'][l]) for l in imagenet_layers]
        
        axes[0, 1].plot(layer_nums, tiny_peak, 'o-', label='Tiny-ImageNet', linewidth=2, markersize=8)
        axes[0, 1].plot(layer_nums, imagenet_peak, 's-', label='ImageNet', linewidth=2, markersize=8)
        axes[0, 1].set_title('Peak Activation Evolution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Average Peak Activation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Concentration Index演化对比
        tiny_conc = [np.mean(tiny_metrics['layer_concentration'][l]) for l in tiny_layers]
        imagenet_conc = [np.mean(imagenet_metrics['layer_concentration'][l]) for l in imagenet_layers]
        
        axes[1, 0].plot(layer_nums, tiny_conc, 'o-', label='Tiny-ImageNet', linewidth=2, markersize=8)
        axes[1, 0].plot(layer_nums, imagenet_conc, 's-', label='ImageNet', linewidth=2, markersize=8)
        axes[1, 0].set_title('Concentration Index Evolution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Average Concentration Index')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 差异热力图
        metrics_diff = np.array([
            np.array(imagenet_entropy) - np.array(tiny_entropy),
            np.array(imagenet_peak) - np.array(tiny_peak),
            np.array(imagenet_conc) - np.array(tiny_conc)
        ])
        
        im = axes[1, 1].imshow(metrics_diff, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[1, 1].set_title('Difference Heatmap (ImageNet - Tiny)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Metric')
        axes[1, 1].set_yticks([0, 1, 2])
        axes[1, 1].set_yticklabels(['Entropy', 'Peak', 'Concentration'])
        axes[1, 1].set_xticks(range(len(layer_nums)))
        axes[1, 1].set_xticklabels(layer_nums)
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        output_file = self.output_dir / "layer_evolution_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 层级演化对比已保存到: {output_file}")
    
    def generate_comparison_report(self, comparison):
        """生成文字对比报告"""
        print("\n生成对比报告...")
        
        report = []
        report.append("# Tiny-ImageNet vs ImageNet 对比分析报告\n")
        report.append("## 整体指标对比\n")
        
        for metric_name, metric_data in comparison.items():
            report.append(f"### {metric_name}\n")
            
            tiny_mean = metric_data['Tiny-ImageNet']['mean']
            tiny_std = metric_data['Tiny-ImageNet']['std']
            imagenet_mean = metric_data['ImageNet']['mean']
            imagenet_std = metric_data['ImageNet']['std']
            
            diff = imagenet_mean - tiny_mean
            diff_pct = (diff / tiny_mean) * 100 if tiny_mean != 0 else 0
            
            report.append(f"- **Tiny-ImageNet**: {tiny_mean:.4f} ± {tiny_std:.4f}\n")
            report.append(f"- **ImageNet**: {imagenet_mean:.4f} ± {imagenet_std:.4f}\n")
            report.append(f"- **差异**: {diff:+.4f} ({diff_pct:+.2f}%)\n")
            
            # 解释
            if 'Entropy' in metric_name:
                if diff > 0:
                    report.append(f"- **解释**: ImageNet上的注意力更分散，可能因为图片更复杂\n")
                else:
                    report.append(f"- **解释**: ImageNet上的注意力更集中\n")
            elif 'Peak' in metric_name:
                if diff > 0:
                    report.append(f"- **解释**: ImageNet上的关键区域更突出\n")
                else:
                    report.append(f"- **解释**: Tiny-ImageNet上的关键区域更突出\n")
            elif 'Concentration' in metric_name:
                if diff > 0:
                    report.append(f"- **解释**: ImageNet上的注意力更集中在少数区域\n")
                else:
                    report.append(f"- **解释**: Tiny-ImageNet上的注意力更集中\n")
            elif 'Diversity' in metric_name:
                if diff > 0:
                    report.append(f"- **解释**: ImageNet上的多头注意力差异更大，分工更明确\n")
                else:
                    report.append(f"- **解释**: Tiny-ImageNet上的多头注意力差异更大\n")
            
            report.append("\n")
        
        report.append("## 关键发现\n\n")
        report.append("1. **数据集规模影响**: ImageNet的更大规模和更高分辨率导致了不同的注意力模式\n")
        report.append("2. **模型差异**: ViT-base-16预训练模型 vs 自训练模型的性能差异\n")
        report.append("3. **注意力质量**: 通过量化指标客观评估两个数据集上的注意力质量\n")
        
        # 保存报告
        output_file = self.output_dir / "comparison_report.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"✓ 对比报告已保存到: {output_file}")
    
    def run_complete_comparison(self):
        """运行完整对比分析"""
        print("="*80)
        print("Tiny-ImageNet vs ImageNet 完整对比分析")
        print("="*80)
        
        # 1. 对比整体指标
        comparison, tiny_metrics, imagenet_metrics = self.compare_overall_metrics()
        
        # 2. 可视化对比
        self.visualize_metric_comparison(tiny_metrics, imagenet_metrics)
        self.visualize_layer_comparison(tiny_metrics, imagenet_metrics)
        
        # 3. 生成报告
        self.generate_comparison_report(comparison)
        
        print("\n" + "="*80)
        print("✓ 对比分析完成！")
        print(f"✓ 结果保存在: {self.output_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Tiny-ImageNet vs ImageNet 对比分析')
    parser.add_argument('--tiny_metrics', type=str,
                       default='do_images/visualize_attention/results_n01443537_24/quantitative/all_metrics.json',
                       help='Tiny-ImageNet指标文件')
    parser.add_argument('--imagenet_metrics', type=str,
                       default='analysis_results/imagenet_vitbase16/metrics/all_metrics.json',
                       help='ImageNet指标文件')
    parser.add_argument('--output_dir', type=str,
                       default='analysis_results/comparison',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.tiny_metrics):
        print(f"错误: Tiny-ImageNet指标文件不存在: {args.tiny_metrics}")
        print("请先运行Tiny-ImageNet的分析")
        return
    
    if not os.path.exists(args.imagenet_metrics):
        print(f"错误: ImageNet指标文件不存在: {args.imagenet_metrics}")
        print("请先运行: python analyze_imagenet_quantitative.py")
        return
    
    # 运行对比分析
    comparator = TinyVsImageNetComparator(args.tiny_metrics, args.imagenet_metrics, args.output_dir)
    comparator.run_complete_comparison()


if __name__ == "__main__":
    main()
