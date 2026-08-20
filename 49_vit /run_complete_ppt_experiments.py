"""
完整的PPT实验脚本 - 8项分析
注意：控制磁盘空间，只保存必要的最终结果
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
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class PPTExperimentRunner:
    """8项PPT实验的完整运行器"""
    
    def __init__(self, base_dir, output_dir, num_images=50):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_images = num_images
        
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        print(f"输出目录: {self.output_dir}")
        print(f"处理图片数: {num_images}")
    
    def get_image_dirs(self):
        """获取图片目录"""
        image_dirs = []
        for item in sorted(self.base_dir.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                image_dirs.append(item)
                if len(image_dirs) >= self.num_images:
                    break
        return image_dirs
    
    def cleanup_temp(self):
        """清理临时文件"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(exist_ok=True)
    
    def run_all_experiments(self):
        """运行所有8项实验"""
        print("\n" + "="*80)
        print("开始运行完整PPT实验")
        print("="*80)
        
        # 实验1-6: 可视化和分析
        self.experiment_1_module_visualization()
        self.experiment_2_v_vector_grid()
        self.experiment_3_entropy_analysis()
        self.experiment_4_layer_changes()
        self.experiment_5_distance_decay()
        self.experiment_6_concentration_evolution()
        
        # 实验7-8在单独的脚本中
        print("\n" + "="*80)
        print("实验1-6完成！")
        print("实验7(Scanner准确率)和实验8(EE值)请运行单独的脚本")
        print("="*80)
        
        # 生成总结报告
        self.generate_summary_report()
        
        # 清理临时文件
        self.cleanup_temp()
        
        print(f"\n所有结果保存在: {self.output_dir}")
        print(f"图表保存在: {self.figures_dir}")
    
    def generate_summary_report(self):
        """生成总结报告"""
        report = f"""# PPT实验结果总结

## 实验完成情况

✅ 实验1: ViT模块可视化网格图
✅ 实验2: V向量CLS可视化网格图  
✅ 实验3: 信息熵分析
✅ 实验4: Layer-to-layer变化分析
✅ 实验5: 距离衰减分析
✅ 实验6: 注意力集中度演化（3个头）
⏳ 实验7: Scanner准确率（需单独运行）
⏳ 实验8: EE值计算（需实验7数据）

## 输出文件

所有图表保存在: {self.figures_dir}

- exp1_module_grid_*.png - 模块可视化
- exp2_v_grid_*.png - V向量网格
- exp3_entropy_analysis.png - 信息熵
- exp4_layer_changes.png - 层间变化
- exp5_distance_decay.png - 距离衰减
- exp6_concentration_head*.png - 集中度演化
- exp6_visualization_head*.png - 对应可视化

## 数据文件

- exp3_entropy_data.json
- exp4_changes_data.json

生成时间: {Path(__file__).name}
"""
        
        report_file = self.output_dir / "EXPERIMENT_SUMMARY.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ 总结报告: {report_file}")

# 导入实验函数
from ppt_experiments_part1 import *
from ppt_experiments_part2 import *

if __name__ == "__main__":
    base_dir = "scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained"
    output_dir = "ppt_experiment_results"
    
    runner = PPTExperimentRunner(base_dir, output_dir, num_images=50)
    runner.run_all_experiments()
