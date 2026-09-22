"""
ImageNet + ViT-base-16 完整量化分析
实现所有关键指标：Entropy, IoU, Peak Activation, Concentration Index等
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from scipy.spatial.distance import cosine
from sklearn.metrics import jaccard_score

class ImageNetQuantitativeAnalyzer:
    def __init__(self, base_dir, output_dir="analysis_results/imagenet_vitbase16"):
        """
        初始化ImageNet量化分析器
        
        参数:
            base_dir: ViT-base-16扫描结果目录
            output_dir: 分析结果输出目录
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        for subdir in ["metrics", "visualizations", "statistics", "comparisons"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # 获取所有处理过的图片目录
        self.image_dirs = self._get_image_dirs()
        print(f"找到 {len(self.image_dirs)} 个已处理的图片")
        
        # 存储所有指标
        self.metrics = {
            'entropy': [],
            'peak_activation': [],
            'concentration_index': [],
            'layer_similarity': [],
            'head_diversity': [],
        }
    
    def _get_image_dirs(self):
        """获取所有已处理的图片目录"""
        dirs = [d for d in self.base_dir.iterdir() 
                if d.is_dir() and (d / "scan_attn_map").exists()]
        return sorted(dirs)
    
    def _load_attention_map(self, image_dir, layer, head):
        """加载注意力图"""
        attn_path = image_dir / "scan_attn_map" / layer / "input_attn" / f"head{head}" / f"avg_cls_attn_head{head}.png"
        
        if not attn_path.exists():
            return None
        
        img = cv2.imread(str(attn_path))
        if img is None:
            return None
        
        # 转换为灰度并归一化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.astype(float) / 255.0
    
    # ==================== 核心指标计算 ====================
    
    def calculate_entropy(self, attention_map):
        """
        计算熵 - 衡量注意力分散程度
        低熵 = 集中，高熵 = 分散
        """
        # 归一化为概率分布
        flat = attention_map.flatten()
        flat = flat / (flat.sum() + 1e-10)
        
        # 移除零值
        flat = flat[flat > 0]
        
        # 计算熵
        entropy = -np.sum(flat * np.log2(flat + 1e-10))
        return entropy
    
    def calculate_peak_activation(self, attention_map):
        """
        计算最大激活值 - 衡量关键区域突出程度
        高峰值 = 强烈关注某区域
        """
        return float(attention_map.max())
    
    def calculate_concentration_index(self, attention_map, k=0.1):
        """
        计算集中度指数 - Top-k激活值占总激活的比例
        高CI = 注意力集中在少数区域
        
        参数:
            k: Top-k比例（默认10%）
        """
        flat = attention_map.flatten()
        sorted_vals = np.sort(flat)[::-1]
        
        top_k_count = max(1, int(len(sorted_vals) * k))
        top_k_sum = sorted_vals[:top_k_count].sum()
        total_sum = sorted_vals.sum()
        
        return float(top_k_sum / (total_sum + 1e-10))
    
    def calculate_iou_with_threshold(self, attention_map, threshold=0.5):
        """
        计算IoU（使用阈值二值化）
        与高激活区域的重叠度
        
        参数:
            threshold: 二值化阈值
        """
        # 二值化
        binary = (attention_map > threshold).astype(int)
        
        # 计算与整个图像的IoU（作为基准）
        # 这里我们计算高激活区域的紧凑度
        if binary.sum() == 0:
            return 0.0
        
        # 计算边界框
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        if not rows.any() or not cols.any():
            return 0.0
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # IoU = 实际激活区域 / 边界框区域
        actual_area = binary.sum()
        bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
        
        return float(actual_area / bbox_area)
    
    def calculate_effective_resolution(self, attention_map, threshold=0.1):
        """
        计算有效分辨率 - 有意义的激活区域大小
        """
        binary = (attention_map > threshold).astype(int)
        return float(binary.sum() / attention_map.size)
    
    def calculate_layer_similarity(self, attn_map1, attn_map2):
        """
        计算两层之间的相似度（余弦相似度）
        """
        flat1 = attn_map1.flatten()
        flat2 = attn_map2.flatten()
        
        # 归一化
        flat1 = flat1 / (np.linalg.norm(flat1) + 1e-10)
        flat2 = flat2 / (np.linalg.norm(flat2) + 1e-10)
        
        # 余弦相似度
        similarity = 1 - cosine(flat1, flat2)
        return float(similarity)
    
    def calculate_head_diversity(self, attention_maps):
        """
        计算多头注意力的差异度
        
        参数:
            attention_maps: list of attention maps from different heads
        """
        if len(attention_maps) < 2:
            return 0.0
        
        diversities = []
        for i in range(len(attention_maps)):
            for j in range(i+1, len(attention_maps)):
                sim = self.calculate_layer_similarity(attention_maps[i], attention_maps[j])
                diversities.append(1 - sim)  # 差异度 = 1 - 相似度
        
        return float(np.mean(diversities))
    
    # ==================== 批量分析 ====================
    
    def analyze_single_image(self, image_dir):
        """分析单张图片的所有指标"""
        image_name = image_dir.name
        print(f"\n分析: {image_name}")
        
        # 获取所有层
        scan_attn_dir = image_dir / "scan_attn_map"
        layers = sorted([d.name for d in scan_attn_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        
        # 获取注意力头数量
        first_layer_dir = scan_attn_dir / layers[0] / "input_attn"
        num_heads = len([d for d in first_layer_dir.iterdir() if d.is_dir() and d.name.startswith("head")])
        
        results = {
            'image_name': image_name,
            'num_layers': len(layers),
            'num_heads': num_heads,
            'layers': {}
        }
        
        # 分析每一层
        for layer in tqdm(layers, desc="Layers"):
            layer_results = {
                'heads': {},
                'layer_metrics': {}
            }
            
            layer_attention_maps = []
            
            # 分析每个头
            for head in range(num_heads):
                attn_map = self._load_attention_map(image_dir, layer, head)
                
                if attn_map is None:
                    continue
                
                layer_attention_maps.append(attn_map)
                
                # 计算所有指标
                head_metrics = {
                    'entropy': self.calculate_entropy(attn_map),
                    'peak_activation': self.calculate_peak_activation(attn_map),
                    'concentration_index': self.calculate_concentration_index(attn_map),
                    'iou_compact': self.calculate_iou_with_threshold(attn_map, threshold=0.5),
                    'effective_resolution': self.calculate_effective_resolution(attn_map),
                }
                
                layer_results['heads'][f'head{head}'] = head_metrics
            
            # 计算层级指标
            if layer_attention_maps:
                layer_results['layer_metrics'] = {
                    'head_diversity': self.calculate_head_diversity(layer_attention_maps),
                    'avg_entropy': np.mean([h['entropy'] for h in layer_results['heads'].values()]),
                    'avg_peak': np.mean([h['peak_activation'] for h in layer_results['heads'].values()]),
                    'avg_concentration': np.mean([h['concentration_index'] for h in layer_results['heads'].values()]),
                }
            
            results['layers'][layer] = layer_results
        
        # 计算跨层相似度
        results['layer_similarities'] = self._calculate_cross_layer_similarities(image_dir, layers, num_heads)
        
        return results
    
    def _calculate_cross_layer_similarities(self, image_dir, layers, num_heads):
        """计算相邻层之间的相似度"""
        similarities = []
        
        for i in range(len(layers) - 1):
            layer_sims = []
            
            for head in range(num_heads):
                attn1 = self._load_attention_map(image_dir, layers[i], head)
                attn2 = self._load_attention_map(image_dir, layers[i+1], head)
                
                if attn1 is not None and attn2 is not None:
                    sim = self.calculate_layer_similarity(attn1, attn2)
                    layer_sims.append(sim)
            
            if layer_sims:
                similarities.append({
                    'from_layer': layers[i],
                    'to_layer': layers[i+1],
                    'avg_similarity': np.mean(layer_sims),
                    'std_similarity': np.std(layer_sims)
                })
        
        return similarities
    
    def analyze_all_images(self):
        """批量分析所有图片"""
        print(f"开始分析 {len(self.image_dirs)} 张图片...")
        
        all_results = []
        
        for image_dir in self.image_dirs:
            try:
                result = self.analyze_single_image(image_dir)
                all_results.append(result)
            except Exception as e:
                print(f"错误处理 {image_dir.name}: {e}")
                continue
        
        # 保存原始结果
        output_file = self.output_dir / "metrics" / "all_metrics.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nOK 原始指标已保存到: {output_file}")
        
        return all_results
    
    # ==================== 统计分析 ====================
    
    def generate_statistics(self, all_results):
        """生成统计报告"""
        print("\n生成统计报告...")
        
        stats = {
            'total_images': len(all_results),
            'metrics_summary': {},
            'layer_analysis': {},
            'head_analysis': {}
        }
        
        # 收集所有指标
        all_entropy = []
        all_peak = []
        all_concentration = []
        all_diversity = []
        
        for result in all_results:
            for layer, layer_data in result['layers'].items():
                for head, head_metrics in layer_data['heads'].items():
                    all_entropy.append(head_metrics['entropy'])
                    all_peak.append(head_metrics['peak_activation'])
                    all_concentration.append(head_metrics['concentration_index'])
                
                if 'head_diversity' in layer_data['layer_metrics']:
                    all_diversity.append(layer_data['layer_metrics']['head_diversity'])
        
        # 统计摘要
        stats['metrics_summary'] = {
            'entropy': {
                'mean': float(np.mean(all_entropy)),
                'std': float(np.std(all_entropy)),
                'min': float(np.min(all_entropy)),
                'max': float(np.max(all_entropy)),
            },
            'peak_activation': {
                'mean': float(np.mean(all_peak)),
                'std': float(np.std(all_peak)),
                'min': float(np.min(all_peak)),
                'max': float(np.max(all_peak)),
            },
            'concentration_index': {
                'mean': float(np.mean(all_concentration)),
                'std': float(np.std(all_concentration)),
                'min': float(np.min(all_concentration)),
                'max': float(np.max(all_concentration)),
            },
            'head_diversity': {
                'mean': float(np.mean(all_diversity)),
                'std': float(np.std(all_diversity)),
                'min': float(np.min(all_diversity)),
                'max': float(np.max(all_diversity)),
            }
        }
        
        # 保存统计结果
        stats_file = self.output_dir / "statistics" / "summary_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"OK 统计报告已保存到: {stats_file}")
        
        return stats
    
    # ==================== 可视化 ====================
    
    def visualize_metrics_distribution(self, all_results):
        """可视化指标分布"""
        print("\n生成指标分布图...")
        
        # 收集数据
        metrics_data = {
            'entropy': [],
            'peak_activation': [],
            'concentration_index': [],
            'head_diversity': []
        }
        
        for result in all_results:
            for layer, layer_data in result['layers'].items():
                for head, head_metrics in layer_data['heads'].items():
                    metrics_data['entropy'].append(head_metrics['entropy'])
                    metrics_data['peak_activation'].append(head_metrics['peak_activation'])
                    metrics_data['concentration_index'].append(head_metrics['concentration_index'])
                
                if 'head_diversity' in layer_data['layer_metrics']:
                    metrics_data['head_diversity'].append(layer_data['layer_metrics']['head_diversity'])
        
        # 创建4个子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ImageNet + ViT-base-16: Metrics Distribution', fontsize=16, fontweight='bold')
        
        # Entropy分布
        axes[0, 0].hist(metrics_data['entropy'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Entropy Distribution')
        axes[0, 0].set_xlabel('Entropy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(metrics_data['entropy']), color='red', linestyle='--', label=f'Mean: {np.mean(metrics_data["entropy"]):.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Peak Activation分布
        axes[0, 1].hist(metrics_data['peak_activation'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Peak Activation Distribution')
        axes[0, 1].set_xlabel('Peak Activation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(metrics_data['peak_activation']), color='red', linestyle='--', label=f'Mean: {np.mean(metrics_data["peak_activation"]):.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Concentration Index分布
        axes[1, 0].hist(metrics_data['concentration_index'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Concentration Index Distribution')
        axes[1, 0].set_xlabel('Concentration Index')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(metrics_data['concentration_index']), color='red', linestyle='--', label=f'Mean: {np.mean(metrics_data["concentration_index"]):.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Head Diversity分布
        axes[1, 1].hist(metrics_data['head_diversity'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Head Diversity Distribution')
        axes[1, 1].set_xlabel('Head Diversity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(metrics_data['head_diversity']), color='red', linestyle='--', label=f'Mean: {np.mean(metrics_data["head_diversity"]):.3f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "visualizations" / "metrics_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OK 指标分布图已保存到: {output_file}")
    
    def visualize_layer_evolution(self, all_results):
        """可视化层级演化"""
        print("\n生成层级演化图...")
        
        # 按层收集平均指标
        layer_metrics = {}
        
        for result in all_results:
            for layer, layer_data in result['layers'].items():
                if layer not in layer_metrics:
                    layer_metrics[layer] = {
                        'entropy': [],
                        'peak': [],
                        'concentration': [],
                        'diversity': []
                    }
                
                if 'layer_metrics' in layer_data:
                    lm = layer_data['layer_metrics']
                    if 'avg_entropy' in lm:
                        layer_metrics[layer]['entropy'].append(lm['avg_entropy'])
                    if 'avg_peak' in lm:
                        layer_metrics[layer]['peak'].append(lm['avg_peak'])
                    if 'avg_concentration' in lm:
                        layer_metrics[layer]['concentration'].append(lm['avg_concentration'])
                    if 'head_diversity' in lm:
                        layer_metrics[layer]['diversity'].append(lm['head_diversity'])
        
        # 计算平均值
        layers = sorted(layer_metrics.keys(), key=lambda x: int(x))
        avg_entropy = [np.mean(layer_metrics[l]['entropy']) for l in layers]
        avg_peak = [np.mean(layer_metrics[l]['peak']) for l in layers]
        avg_concentration = [np.mean(layer_metrics[l]['concentration']) for l in layers]
        avg_diversity = [np.mean(layer_metrics[l]['diversity']) for l in layers]
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ImageNet + ViT-base-16: Layer Evolution', fontsize=16, fontweight='bold')
        
        layer_nums = [int(l) for l in layers]
        
        # Entropy演化
        axes[0, 0].plot(layer_nums, avg_entropy, 'o-', linewidth=2, markersize=8, color='blue')
        axes[0, 0].set_title('Entropy Evolution Across Layers')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Average Entropy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Peak Activation演化
        axes[0, 1].plot(layer_nums, avg_peak, 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_title('Peak Activation Evolution')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Average Peak Activation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Concentration Index演化
        axes[1, 0].plot(layer_nums, avg_concentration, 'o-', linewidth=2, markersize=8, color='orange')
        axes[1, 0].set_title('Concentration Index Evolution')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Average Concentration Index')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Head Diversity演化
        axes[1, 1].plot(layer_nums, avg_diversity, 'o-', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_title('Head Diversity Evolution')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Average Head Diversity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "visualizations" / "layer_evolution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"OK 层级演化图已保存到: {output_file}")
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("="*80)
        print("ImageNet + ViT-base-16 完整量化分析")
        print("="*80)
        
        # 1. 分析所有图片
        all_results = self.analyze_all_images()
        
        # 2. 生成统计报告
        stats = self.generate_statistics(all_results)
        
        # 3. 生成可视化
        self.visualize_metrics_distribution(all_results)
        self.visualize_layer_evolution(all_results)
        
        print("\n" + "="*80)
        print("OK 分析完成！")
        print(f"OK 结果保存在: {self.output_dir}")
        print("="*80)
        
        return all_results, stats


def main():
    parser = argparse.ArgumentParser(description='ImageNet + ViT-base-16 量化分析')
    parser.add_argument('--base_dir', type=str, 
                       default=r'scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained',
                       help='ViT-base-16扫描结果目录')
    parser.add_argument('--output_dir', type=str,
                       default='analysis_results/imagenet_vitbase16',
                       help='分析结果输出目录')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = ImageNetQuantitativeAnalyzer(args.base_dir, args.output_dir)
    results, stats = analyzer.run_complete_analysis()
    
    # 打印关键统计
    print("\n关键指标摘要:")
    print(f"  Entropy: {stats['metrics_summary']['entropy']['mean']:.3f} ± {stats['metrics_summary']['entropy']['std']:.3f}")
    print(f"  Peak Activation: {stats['metrics_summary']['peak_activation']['mean']:.3f} ± {stats['metrics_summary']['peak_activation']['std']:.3f}")
    print(f"  Concentration Index: {stats['metrics_summary']['concentration_index']['mean']:.3f} ± {stats['metrics_summary']['concentration_index']['std']:.3f}")
    print(f"  Head Diversity: {stats['metrics_summary']['head_diversity']['mean']:.3f} ± {stats['metrics_summary']['head_diversity']['std']:.3f}")


if __name__ == "__main__":
    main()
