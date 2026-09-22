"""重新生成平滑的V向量可视化"""
import os
import sys
import subprocess

def regenerate_v_vectors():
    """重新生成V向量数据，使用新的平滑可视化"""
    print("重新生成平滑的V向量可视化...")
    
    # 使用鱼图像重新扫描
    scan_script = "run_nns_scan_fixed.py"
    
    if os.path.exists(scan_script):
        print("Step 1: 重新运行NNS扫描（生成平滑V向量）...")
        result = subprocess.run([
            r"D:\Program Files\miniconda3\envs\vit_114\python.exe", 
            scan_script
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("SUCCESS: NNS扫描完成，V向量已使用新的平滑方法生成")
        else:
            print(f"ERROR: NNS扫描失败: {result.stderr}")
            return False
    else:
        print(f"ERROR: 找不到扫描脚本: {scan_script}")
        return False
    
    # 重新运行V向量分析
    print("Step 2: 重新运行V向量分析...")
    base_dir = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final_fixed\0_n01443537_24_fixed"
    analysis_script = r"do_images\visualize_v\analyze_vit_values.py"
    output_dir = r"analysis_results\value_analysis_smooth"
    
    result = subprocess.run([
        r"D:\Program Files\miniconda3\envs\vit_114\python.exe",
        analysis_script,
        "--base_dir", base_dir,
        "--output_dir", output_dir
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    if result.returncode == 0:
        print("SUCCESS: 平滑V向量分析完成")
        print(f"结果保存在: {output_dir}")
        return True
    else:
        print(f"ERROR: V向量分析失败: {result.stderr}")
        return False

def create_v_vector_comparison():
    """创建V向量对比图（原始 vs 平滑）"""
    print("Step 3: 创建V向量对比图...")
    
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    # 对比路径
    original_dir = r"analysis_results\value_analysis"
    smooth_dir = r"analysis_results\value_analysis_smooth"
    
    # 创建对比图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('V-Vector Visualization Comparison: Original vs Smooth', fontsize=16, fontweight='bold')
    
    # 要对比的图像
    comparison_images = [
        ("layer_progression/head0_progression.png", "Head 0 Progression"),
        ("layer_progression/head1_progression.png", "Head 1 Progression"),
        ("layer_progression/head2_progression.png", "Head 2 Progression"),
        ("layer_progression/layer_changes.png", "Layer Changes")
    ]
    
    for i, (img_path, title) in enumerate(comparison_images):
        # 原始版本
        original_path = os.path.join(original_dir, img_path)
        if os.path.exists(original_path):
            img = Image.open(original_path)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Original\n{title}", fontsize=12)
        else:
            axes[0, i].text(0.5, 0.5, 'Original\nNot Available', ha='center', va='center')
            axes[0, i].set_title(f"Original\n{title}", fontsize=12)
        axes[0, i].axis('off')
        
        # 平滑版本
        smooth_path = os.path.join(smooth_dir, img_path)
        if os.path.exists(smooth_path):
            img = Image.open(smooth_path)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Smooth (VIRIDIS)\n{title}", fontsize=12)
        else:
            axes[1, i].text(0.5, 0.5, 'Smooth\nNot Available', ha='center', va='center')
            axes[1, i].set_title(f"Smooth (VIRIDIS)\n{title}", fontsize=12)
        axes[1, i].axis('off')
    
    # 添加说明
    fig.text(0.02, 0.5, 'Original\n(JET colormap)', rotation=90, va='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.25, 'Smooth\n(VIRIDIS + Gaussian)', rotation=90, va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = "analysis_results/V_VECTOR_COMPARISON.png"
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: V向量对比图已保存: {comparison_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(comparison_path)
            print("SUCCESS: 对比图已自动打开")
        except:
            pass
    
    return comparison_path

if __name__ == "__main__":
    print("="*80)
    print("Regenerating Smooth V-Vector Visualizations")
    print("="*80)
    
    # 重新生成V向量
    success = regenerate_v_vectors()
    
    if success:
        # 创建对比图
        comparison_path = create_v_vector_comparison()
        
        print("\n" + "="*80)
        print("SUCCESS: Smooth V-vector regeneration completed!")
        print("="*80)
        print("Changes made:")
        print("- VIRIDIS colormap (smoother than JET)")
        print("- Gaussian smoothing (sigma=0.5)")
        print("- Bilinear interpolation (smoother than LANCZOS4)")
        print(f"- Comparison image: {comparison_path}")
        print("\nThe grid-like artifacts should now be reduced!")
    else:
        print("\nERROR: Failed to regenerate V-vectors")
