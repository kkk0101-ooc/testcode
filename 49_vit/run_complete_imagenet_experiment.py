"""
ImageNet + ViT-base-16 完整实验主控脚本
自动化运行所有分析步骤
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

class ImageNetExperimentRunner:
    def __init__(self, python_path=None):
        """
        初始化实验运行器
        
        参数:
            python_path: Python解释器路径
        """
        self.python_path = python_path or sys.executable
        self.base_dir = Path(__file__).parent
        
        print("="*80)
        print("ImageNet + ViT-base-16 完整实验流程")
        print("="*80)
        print(f"Python: {self.python_path}")
        print(f"工作目录: {self.base_dir}")
        print("="*80)
    
    def run_command(self, cmd, description):
        """运行命令并显示进度"""
        print(f"\n{'='*80}")
        print(f"步骤: {description}")
        print(f"命令: {' '.join(cmd)}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, cwd=self.base_dir, check=True,
                                   capture_output=False, text=True)
            elapsed = time.time() - start_time
            print(f"\n✓ 完成 ({elapsed:.1f}秒)")
            return True
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n✗ 失败 ({elapsed:.1f}秒)")
            print(f"错误: {e}")
            return False
    
    def step1_expand_imagenet_processing(self, count=200):
        """步骤1: 扩展ImageNet处理"""
        print("\n" + "="*80)
        print("步骤1: 扩展ImageNet图片处理")
        print(f"目标: 处理{count}张图片")
        print("="*80)
        
        cmd = [
            self.python_path,
            "train_scanner_cls.py",
            "--device", "0",
            "--count", str(count),
            "--dataset", "ImageNet",
            "--classes", "1000",
            "--imagenet_path", r"D:\project\vit_114\NNS_model_and_dataset\dataset\imagenet_full\val",
            "--process_wrong_classification", "True",
            "--nns_resize", "64"
        ]
        
        return self.run_command(cmd, f"处理{count}张ImageNet图片")
    
    def step2_generate_grid_images(self):
        """步骤2: 生成网格图"""
        print("\n" + "="*80)
        print("步骤2: 生成层注意力网格图")
        print("="*80)
        
        cmd = [
            self.python_path,
            "combine_layer_attention.py",
            "--base_dir", r"scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained",
            "--grid_size", "3", "4"
        ]
        
        return self.run_command(cmd, "生成网格图")
    
    def step3_quantitative_analysis(self):
        """步骤3: 量化分析"""
        print("\n" + "="*80)
        print("步骤3: 运行完整量化分析")
        print("="*80)
        
        cmd = [
            self.python_path,
            "analyze_imagenet_quantitative.py",
            "--base_dir", r"scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained",
            "--output_dir", "analysis_results/imagenet_vitbase16"
        ]
        
        return self.run_command(cmd, "量化分析")
    
    def step4_comparison_analysis(self):
        """步骤4: 对比分析"""
        print("\n" + "="*80)
        print("步骤4: Tiny-ImageNet vs ImageNet 对比分析")
        print("="*80)
        
        # 检查Tiny-ImageNet指标是否存在
        tiny_metrics = self.base_dir / "do_images" / "visualize_attention" / "results_n01443537_24" / "quantitative" / "all_metrics.json"
        
        if not tiny_metrics.exists():
            print(f"警告: Tiny-ImageNet指标文件不存在: {tiny_metrics}")
            print("跳过对比分析步骤")
            return True
        
        cmd = [
            self.python_path,
            "compare_tiny_vs_imagenet.py",
            "--tiny_metrics", str(tiny_metrics),
            "--imagenet_metrics", "analysis_results/imagenet_vitbase16/metrics/all_metrics.json",
            "--output_dir", "analysis_results/comparison"
        ]
        
        return self.run_command(cmd, "对比分析")
    
    def step5_generate_report(self):
        """步骤5: 生成最终报告"""
        print("\n" + "="*80)
        print("步骤5: 生成最终分析报告")
        print("="*80)
        
        # 这里可以添加报告生成脚本
        # 暂时只打印摘要
        print("\n报告生成位置:")
        print(f"  - 量化分析: analysis_results/imagenet_vitbase16/")
        print(f"  - 对比分析: analysis_results/comparison/")
        print(f"  - 可视化: scan_results/scan_vit/114VisionTransformer_ImageNet/")
        
        return True
    
    def run_full_experiment(self, skip_processing=False, image_count=200):
        """运行完整实验流程"""
        print("\n" + "="*80)
        print("开始完整实验流程")
        print("="*80)
        
        steps = []
        
        # 步骤1: 扩展ImageNet处理（可选跳过）
        if not skip_processing:
            steps.append(("扩展ImageNet处理", lambda: self.step1_expand_imagenet_processing(image_count)))
        else:
            print("\n跳过ImageNet处理步骤（使用现有结果）")
        
        # 步骤2-5
        steps.extend([
            ("生成网格图", self.step2_generate_grid_images),
            ("量化分析", self.step3_quantitative_analysis),
            ("对比分析", self.step4_comparison_analysis),
            ("生成报告", self.step5_generate_report),
        ])
        
        # 执行所有步骤
        results = []
        for step_name, step_func in steps:
            success = step_func()
            results.append((step_name, success))
            
            if not success:
                print(f"\n警告: {step_name} 失败，但继续执行后续步骤...")
        
        # 打印总结
        print("\n" + "="*80)
        print("实验流程完成总结")
        print("="*80)
        
        for step_name, success in results:
            status = "✓" if success else "✗"
            print(f"{status} {step_name}")
        
        print("\n" + "="*80)
        print("所有步骤执行完毕！")
        print("="*80)
        
        # 打印结果位置
        print("\n关键结果文件:")
        print("  1. 量化指标: analysis_results/imagenet_vitbase16/metrics/all_metrics.json")
        print("  2. 统计摘要: analysis_results/imagenet_vitbase16/statistics/summary_statistics.json")
        print("  3. 可视化图: analysis_results/imagenet_vitbase16/visualizations/")
        print("  4. 对比报告: analysis_results/comparison/comparison_report.md")
        print("\n")


def main():
    parser = argparse.ArgumentParser(description='ImageNet + ViT-base-16 完整实验流程')
    parser.add_argument('--python', type=str, default=None,
                       help='Python解释器路径（默认使用当前Python）')
    parser.add_argument('--skip_processing', action='store_true',
                       help='跳过ImageNet处理步骤，使用现有结果')
    parser.add_argument('--image_count', type=int, default=200,
                       help='要处理的ImageNet图片数量（默认200）')
    parser.add_argument('--step', type=str, choices=['1', '2', '3', '4', '5', 'all'], default='all',
                       help='运行特定步骤或全部步骤')
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = ImageNetExperimentRunner(args.python)
    
    # 根据参数运行
    if args.step == 'all':
        runner.run_full_experiment(args.skip_processing, args.image_count)
    elif args.step == '1':
        runner.step1_expand_imagenet_processing(args.image_count)
    elif args.step == '2':
        runner.step2_generate_grid_images()
    elif args.step == '3':
        runner.step3_quantitative_analysis()
    elif args.step == '4':
        runner.step4_comparison_analysis()
    elif args.step == '5':
        runner.step5_generate_report()


if __name__ == "__main__":
    main()
