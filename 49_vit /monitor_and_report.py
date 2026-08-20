"""
监控实验进度并自动生成最终报告
"""

import time
import os
from pathlib import Path
import subprocess
import json

def count_processed_images(base_dir):
    """统计已处理的图片数量"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return 0
    dirs = [d for d in base_path.iterdir() if d.is_dir() and (d / "scan_attn_map").exists()]
    return len(dirs)

def check_analysis_complete(metrics_file):
    """检查量化分析是否完成"""
    if not os.path.exists(metrics_file):
        return False, 0
    
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return True, len(data)
    except:
        return False, 0

def monitor_experiment():
    """监控实验进度"""
    print("="*80)
    print("ImageNet + ViT-base-16 实验监控")
    print("="*80)
    
    base_dir = r"scan_results\scan_vit\114VisionTransformer_ImageNet\vit_base_patch16_224_pretrained"
    metrics_file = r"analysis_results\imagenet_vitbase16\metrics\all_metrics.json"
    
    last_count = 0
    check_interval = 30  # 每30秒检查一次
    
    while True:
        # 检查图片处理进度
        current_count = count_processed_images(base_dir)
        
        # 检查分析进度
        analysis_done, analyzed_count = check_analysis_complete(metrics_file)
        
        # 打印状态
        print(f"\n[{time.strftime('%H:%M:%S')}] 当前状态:")
        print(f"  - 已处理图片: {current_count} 张")
        if analysis_done:
            print(f"  - 量化分析: 完成 ({analyzed_count} 张)")
        else:
            print(f"  - 量化分析: 进行中...")
        
        # 检查是否有新图片
        if current_count > last_count:
            print(f"  → 新增 {current_count - last_count} 张图片")
            last_count = current_count
        
        # 如果达到目标，生成报告
        if current_count >= 200 and analysis_done:
            print("\n" + "="*80)
            print("实验完成！开始生成最终报告...")
            print("="*80)
            
            # 生成报告
            subprocess.run([
                "python", "generate_final_report.py",
                "--metrics_file", metrics_file,
                "--output_file", "FINAL_EXPERIMENT_REPORT.md"
            ])
            
            print("\n完成！查看报告: FINAL_EXPERIMENT_REPORT.md\n")
            break
        
        # 等待下一次检查
        time.sleep(check_interval)

if __name__ == "__main__":
    try:
        monitor_experiment()
    except KeyboardInterrupt:
        print("\n\n监控已停止")
