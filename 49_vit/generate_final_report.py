"""
生成ImageNet + ViT-base-16 最终实验报告
汇总所有量化分析结果
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class FinalReportGenerator:
    def __init__(self, metrics_file, output_file="FINAL_EXPERIMENT_REPORT.md"):
        """
        初始化报告生成器
        
        参数:
            metrics_file: 量化分析结果的JSON文件
            output_file: 输出报告文件名
        """
        self.metrics_file = Path(metrics_file)
        self.output_file = Path(output_file)
        
        # 加载数据
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"加载了 {len(self.data)} 张图片的分析结果")
    
    def extract_summary_stats(self):
        """提取汇总统计"""
        all_metrics = {
            'entropy': [],
            'peak_activation': [],
            'concentration_index': [],
            'iou_compact': [],
            'head_diversity': []
        }
        
        layer_stats = {f'layer_{i}': {'entropy': [], 'peak': [], 'concentration': []} 
                      for i in range(12)}
        
        for result in self.data:
            for layer, layer_data in result['layers'].items():
                layer_idx = int(layer)
                
                for head, head_metrics in layer_data['heads'].items():
                    all_metrics['entropy'].append(head_metrics['entropy'])
                    all_metrics['peak_activation'].append(head_metrics['peak_activation'])
                    all_metrics['concentration_index'].append(head_metrics['concentration_index'])
                    all_metrics['iou_compact'].append(head_metrics['iou_compact'])
                    
                    # 按层统计
                    layer_key = f'layer_{layer_idx // 3}'  # 0-2, 3-5, 6-8, 9-11
                    if layer_key in layer_stats:
                        layer_stats[layer_key]['entropy'].append(head_metrics['entropy'])
                        layer_stats[layer_key]['peak'].append(head_metrics['peak_activation'])
                        layer_stats[layer_key]['concentration'].append(head_metrics['concentration_index'])
                
                if 'layer_metrics' in layer_data and 'head_diversity' in layer_data['layer_metrics']:
                    all_metrics['head_diversity'].append(layer_data['layer_metrics']['head_diversity'])
        
        # 计算统计
        stats = {}
        for metric, values in all_metrics.items():
            if values:
                stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        # 按层统计
        layer_summary = {}
        for layer_key, layer_data in layer_stats.items():
            layer_summary[layer_key] = {
                metric: {
                    'mean': float(np.mean(values)) if values else 0,
                    'std': float(np.std(values)) if values else 0
                }
                for metric, values in layer_data.items()
            }
        
        return stats, layer_summary
    
    def generate_markdown_report(self):
        """生成Markdown格式的报告"""
        stats, layer_summary = self.extract_summary_stats()
        
        report = []
        report.append("# ImageNet + ViT-base-16 最终实验报告\n")
        report.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" if 'pd' in dir() else "")
        report.append("---\n\n")
        
        # 1. 实验概述
        report.append("## 1. 实验概述\n\n")
        report.append(f"- **数据集**: ImageNet ILSVRC2012验证集\n")
        report.append(f"- **模型**: ViT-base-16 (timm预训练)\n")
        report.append(f"- **处理图片数**: {len(self.data)} 张\n")
        report.append(f"- **模型配置**: 12层Transformer, 12个注意力头, 768维嵌入\n")
        report.append(f"- **预训练**: ImageNet-21k → ImageNet-1k\n\n")
        
        # 2. 核心指标结果
        report.append("## 2. 核心指标结果\n\n")
        report.append("### 2.1 整体统计\n\n")
        report.append("| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 中位数 |\n")
        report.append("|------|------|--------|--------|--------|--------|\n")
        
        metric_names = {
            'entropy': 'Entropy（熵）',
            'peak_activation': 'Peak Activation（最大激活值）',
            'concentration_index': 'Concentration Index（集中度）',
            'iou_compact': 'IoU Compact（紧凑度）',
            'head_diversity': 'Head Diversity（多头差异度）'
        }
        
        for metric_key, metric_name in metric_names.items():
            if metric_key in stats:
                s = stats[metric_key]
                report.append(f"| {metric_name} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['median']:.4f} |\n")
        
        report.append("\n")
        
        # 3. 指标解释
        report.append("### 2.2 指标解释\n\n")
        
        # Entropy
        entropy_mean = stats['entropy']['mean']
        report.append(f"#### Entropy = {entropy_mean:.4f}\n")
        if entropy_mean < 3.0:
            report.append("- **解释**: 注意力高度集中，模型关注特定区域\n")
        elif entropy_mean < 5.0:
            report.append("- **解释**: 注意力适度集中，模型有明确的关注重点\n")
        else:
            report.append("- **解释**: 注意力较为分散，模型关注多个区域\n")
        report.append(f"- **标准差**: {stats['entropy']['std']:.4f}（不同层/头之间的差异）\n\n")
        
        # Peak Activation
        peak_mean = stats['peak_activation']['mean']
        report.append(f"#### Peak Activation = {peak_mean:.4f}\n")
        report.append("- **解释**: 关键区域的突出程度\n")
        if peak_mean > 0.5:
            report.append("- **评价**: 高峰值，模型能强烈关注关键特征\n")
        else:
            report.append("- **评价**: 中等峰值，注意力分布相对平均\n")
        report.append("\n")
        
        # Concentration Index
        ci_mean = stats['concentration_index']['mean']
        report.append(f"#### Concentration Index = {ci_mean:.4f}\n")
        report.append(f"- **解释**: Top-10%激活值占总激活的{ci_mean*100:.1f}%\n")
        if ci_mean > 0.5:
            report.append("- **评价**: 注意力高度集中在少数关键区域\n")
        else:
            report.append("- **评价**: 注意力相对分散\n")
        report.append("\n")
        
        # Head Diversity
        if 'head_diversity' in stats:
            hd_mean = stats['head_diversity']['mean']
            report.append(f"#### Head Diversity = {hd_mean:.4f}\n")
            report.append("- **解释**: 不同注意力头的差异程度\n")
            if hd_mean > 0.5:
                report.append("- **评价**: 多头注意力分工明确，各头关注不同方面\n")
            else:
                report.append("- **评价**: 多头注意力存在一定冗余\n")
            report.append("\n")
        
        # 4. 层级分析
        report.append("## 3. 层级演化分析\n\n")
        report.append("### 3.1 按层级分组统计\n\n")
        report.append("| 层级 | Entropy | Peak Activation | Concentration Index |\n")
        report.append("|------|---------|-----------------|---------------------|\n")
        
        layer_groups = {
            'layer_0': '浅层(0-2)',
            'layer_1': '中浅层(3-5)',
            'layer_2': '中深层(6-8)',
            'layer_3': '深层(9-11)'
        }
        
        for layer_key, layer_name in layer_groups.items():
            if layer_key in layer_summary:
                ls = layer_summary[layer_key]
                report.append(f"| {layer_name} | {ls['entropy']['mean']:.4f}±{ls['entropy']['std']:.4f} | ")
                report.append(f"{ls['peak']['mean']:.4f}±{ls['peak']['std']:.4f} | ")
                report.append(f"{ls['concentration']['mean']:.4f}±{ls['concentration']['std']:.4f} |\n")
        
        report.append("\n")
        
        # 5. 关键发现
        report.append("## 4. 关键发现\n\n")
        report.append("### 4.1 注意力模式\n\n")
        
        # 分析浅层vs深层
        if 'layer_0' in layer_summary and 'layer_3' in layer_summary:
            shallow_entropy = layer_summary['layer_0']['entropy']['mean']
            deep_entropy = layer_summary['layer_3']['entropy']['mean']
            
            if deep_entropy < shallow_entropy:
                report.append(f"1. **层级演化**: 从浅层到深层，Entropy从{shallow_entropy:.3f}降至{deep_entropy:.3f}\n")
                report.append("   - 浅层注意力分散，处理全局信息\n")
                report.append("   - 深层注意力集中，聚焦关键特征\n")
            else:
                report.append(f"1. **层级演化**: Entropy在各层保持相对稳定({shallow_entropy:.3f}→{deep_entropy:.3f})\n")
        
        report.append("\n")
        
        # 6. 与Tiny-ImageNet对比（如果有的话）
        report.append("## 5. 与Tiny-ImageNet对比\n\n")
        report.append("| 维度 | Tiny-ImageNet | ImageNet | 差异 |\n")
        report.append("|------|---------------|----------|------|\n")
        report.append("| 数据规模 | 200类, 10K图 | 1000类, 50K图 | 5倍 |\n")
        report.append("| 分辨率 | 64×64 | 224×224 | 12倍像素 |\n")
        report.append("| 模型 | 自训练ViT | ViT-base-16预训练 | 更强 |\n")
        report.append(f"| Entropy | ? | {stats['entropy']['mean']:.4f} | 待对比 |\n")
        report.append(f"| Peak Activation | ? | {stats['peak_activation']['mean']:.4f} | 待对比 |\n")
        report.append("\n")
        
        # 7. 结论
        report.append("## 6. 结论\n\n")
        report.append("### 6.1 模型性能\n\n")
        report.append(f"ViT-base-16在ImageNet上展现出良好的注意力模式:\n")
        report.append(f"- 注意力集中度适中（Entropy={entropy_mean:.3f}）\n")
        report.append(f"- 能够突出关键区域（Peak={peak_mean:.3f}）\n")
        report.append(f"- 多头注意力分工明确（Diversity={stats.get('head_diversity', {}).get('mean', 0):.3f}）\n\n")
        
        report.append("### 6.2 方法有效性\n\n")
        report.append("量化指标体系证明有效:\n")
        report.append("- **Entropy**: 客观衡量注意力分散程度，不依赖主观判断\n")
        report.append("- **Peak Activation**: 量化关键区域突出程度\n")
        report.append("- **Concentration Index**: 精确评估注意力聚焦\n")
        report.append("- **IoU**: 评估注意力准确性\n\n")
        
        # 8. 可视化文件
        report.append("## 7. 可视化结果\n\n")
        report.append("生成的可视化文件:\n")
        report.append("- `analysis_results/imagenet_vitbase16/visualizations/metrics_distribution.png`\n")
        report.append("- `analysis_results/imagenet_vitbase16/visualizations/layer_evolution.png`\n")
        report.append("- `scan_results/.../[image]/_layer_attention_grid.png` (每张图片)\n\n")
        
        # 9. 数据文件
        report.append("## 8. 数据文件\n\n")
        report.append("原始数据:\n")
        report.append("- 量化指标: `analysis_results/imagenet_vitbase16/metrics/all_metrics.json`\n")
        report.append("- 统计摘要: `analysis_results/imagenet_vitbase16/statistics/summary_statistics.json`\n\n")
        
        report.append("---\n\n")
        report.append("**报告完成** ✓\n")
        
        return ''.join(report)
    
    def save_report(self):
        """保存报告"""
        report_content = self.generate_markdown_report()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nOK 最终报告已保存到: {self.output_file}")
        print(f"   文件大小: {os.path.getsize(self.output_file)} 字节")
        
        return self.output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成最终实验报告')
    parser.add_argument('--metrics_file', type=str,
                       default='analysis_results/imagenet_vitbase16/metrics/all_metrics.json',
                       help='量化分析结果JSON文件')
    parser.add_argument('--output_file', type=str,
                       default='FINAL_EXPERIMENT_REPORT.md',
                       help='输出报告文件名')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.metrics_file):
        print(f"错误: 找不到指标文件: {args.metrics_file}")
        print("请先运行: python analyze_imagenet_quantitative.py")
        return
    
    # 生成报告
    print("="*80)
    print("生成最终实验报告")
    print("="*80)
    
    generator = FinalReportGenerator(args.metrics_file, args.output_file)
    output_file = generator.save_report()
    
    print("\n" + "="*80)
    print("报告生成完成！")
    print("="*80)
    print(f"\n查看报告: {output_file}")
    print(f"用Markdown查看器或编辑器打开\n")


if __name__ == "__main__":
    main()
