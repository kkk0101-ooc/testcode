"""生成超平滑的V向量可视化"""
import os
import sys
import subprocess

def generate_ultra_smooth_v_vectors():
    """生成超平滑的V向量数据"""
    print("生成超平滑的V向量可视化...")
    
    # 使用鱼图像重新扫描
    scan_script = "run_nns_scan_fixed.py"
    
    if os.path.exists(scan_script):
        print("Step 1: 重新运行NNS扫描（生成超平滑V向量）...")
        result = subprocess.run([
            r"D:\Program Files\miniconda3\envs\vit_114\python.exe", 
            scan_script
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("SUCCESS: NNS扫描完成，V向量已使用超平滑方法生成")
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
    output_dir = r"analysis_results\value_analysis_ultra_smooth"
    
    result = subprocess.run([
        r"D:\Program Files\miniconda3\envs\vit_114\python.exe",
        analysis_script,
        "--base_dir", base_dir,
        "--output_dir", output_dir
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    if result.returncode == 0:
        print("SUCCESS: 超平滑V向量分析完成")
        print(f"结果保存在: {output_dir}")
        return True
    else:
        print(f"ERROR: V向量分析失败: {result.stderr}")
        return False

def create_smoothness_comparison():
    """创建平滑度对比图（原始 vs 平滑 vs 超平滑）"""
    print("Step 3: 创建平滑度对比图...")
    
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    # 对比路径
    original_dir = r"analysis_results\value_analysis"
    smooth_dir = r"analysis_results\value_analysis_smooth"
    ultra_smooth_dir = r"analysis_results\value_analysis_ultra_smooth"
    
    # 创建对比图
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('V-Vector Smoothness Comparison: Original → Smooth → Ultra-Smooth', fontsize=18, fontweight='bold')
    
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
            axes[0, i].set_title(f"Original (JET)\n{title}", fontsize=11)
        else:
            axes[0, i].text(0.5, 0.5, 'Original\nNot Available', ha='center', va='center')
            axes[0, i].set_title(f"Original\n{title}", fontsize=11)
        axes[0, i].axis('off')
        
        # 平滑版本
        smooth_path = os.path.join(smooth_dir, img_path)
        if os.path.exists(smooth_path):
            img = Image.open(smooth_path)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Smooth (VIRIDIS)\n{title}", fontsize=11)
        else:
            axes[1, i].text(0.5, 0.5, 'Smooth\nNot Available', ha='center', va='center')
            axes[1, i].set_title(f"Smooth\n{title}", fontsize=11)
        axes[1, i].axis('off')
        
        # 超平滑版本
        ultra_smooth_path = os.path.join(ultra_smooth_dir, img_path)
        if os.path.exists(ultra_smooth_path):
            img = Image.open(ultra_smooth_path)
            axes[2, i].imshow(img)
            axes[2, i].set_title(f"Ultra-Smooth (PLASMA)\n{title}", fontsize=11)
        else:
            axes[2, i].text(0.5, 0.5, 'Ultra-Smooth\nNot Available', ha='center', va='center')
            axes[2, i].set_title(f"Ultra-Smooth\n{title}", fontsize=11)
        axes[2, i].axis('off')
    
    # 添加说明
    fig.text(0.02, 0.83, 'Original\n(JET colormap\nNo smoothing)', rotation=90, va='center', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.5, 'Smooth\n(VIRIDIS colormap\nGaussian σ=0.5)', rotation=90, va='center', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.17, 'Ultra-Smooth\n(PLASMA colormap\nMulti-step filtering)', rotation=90, va='center', fontsize=12, fontweight='bold')
    
    # 添加技术说明
    fig.text(0.5, 0.02, 
             'Ultra-Smooth Processing: Gaussian(σ=1.0) → Median Filter(3×3) → Gaussian(σ=0.3) → PLASMA Colormap → Bilinear Interpolation',
             ha='center', fontsize=10, style='italic', color='#666666')
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = "analysis_results/V_VECTOR_SMOOTHNESS_COMPARISON.png"
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: 平滑度对比图已保存: {comparison_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(comparison_path)
            print("SUCCESS: 对比图已自动打开")
        except:
            pass
    
    return comparison_path

def create_single_layer_detail_comparison():
    """创建单层详细对比，展示平滑效果"""
    print("Step 4: 创建单层详细对比...")
    
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    # 检查是否有单层V向量图像可以对比
    base_dir = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final_fixed\0_n01443537_24_fixed"
    scan_v_dir = os.path.join(base_dir, "scan_v")
    
    if not os.path.exists(scan_v_dir):
        print("WARNING: 找不到scan_v目录，跳过单层对比")
        return None
    
    # 找到一个层目录
    layer_dirs = [d for d in os.listdir(scan_v_dir) if os.path.isdir(os.path.join(scan_v_dir, d)) and d.isdigit()]
    if not layer_dirs:
        print("WARNING: 找不到层目录，跳过单层对比")
        return None
    
    # 选择中间层进行对比
    layer_dirs.sort(key=int)
    mid_layer = layer_dirs[len(layer_dirs)//2]
    layer_path = os.path.join(scan_v_dir, mid_layer)
    
    # 创建详细对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'V-Vector Detail Comparison - Layer {mid_layer}\nBefore and After Ultra-Smooth Processing', 
                 fontsize=16, fontweight='bold')
    
    # 对比三个头的CLS V向量
    for head_idx in range(3):
        # 查找文件
        cls_file = f"cls_v_magnitude_head{head_idx}.png"
        cls_path = os.path.join(layer_path, cls_file)
        
        if os.path.exists(cls_path):
            # 加载图像
            img = Image.open(cls_path)
            img_array = np.array(img)
            
            # 原始图像（上排）
            axes[0, head_idx].imshow(img)
            axes[0, head_idx].set_title(f'Head {head_idx} - Ultra-Smooth\n(PLASMA + Multi-Filter)', fontsize=12, fontweight='bold')
            axes[0, head_idx].axis('off')
            
            # 放大中心区域显示细节（下排）
            h, w = img_array.shape[:2]
            center_h, center_w = h//2, w//2
            crop_size = min(h, w) // 4
            
            cropped = img_array[center_h-crop_size:center_h+crop_size, 
                              center_w-crop_size:center_w+crop_size]
            
            axes[1, head_idx].imshow(cropped)
            axes[1, head_idx].set_title(f'Head {head_idx} - Center Detail\n(4x Zoom)', fontsize=12)
            axes[1, head_idx].axis('off')
        else:
            axes[0, head_idx].text(0.5, 0.5, f'Head {head_idx}\nNot Available', ha='center', va='center')
            axes[0, head_idx].set_title(f'Head {head_idx}', fontsize=12)
            axes[0, head_idx].axis('off')
            
            axes[1, head_idx].text(0.5, 0.5, f'Head {head_idx}\nDetail N/A', ha='center', va='center')
            axes[1, head_idx].set_title(f'Head {head_idx} Detail', fontsize=12)
            axes[1, head_idx].axis('off')
    
    # 添加说明
    fig.text(0.02, 0.75, 'Full Image', rotation=90, va='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.25, 'Center Detail\n(4x Zoom)', rotation=90, va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存详细对比图
    detail_path = f"analysis_results/V_VECTOR_DETAIL_LAYER_{mid_layer}.png"
    plt.savefig(detail_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: 单层详细对比图已保存: {detail_path}")
    
    return detail_path

if __name__ == "__main__":
    print("="*80)
    print("Generating Ultra-Smooth V-Vector Visualizations")
    print("="*80)
    print("Improvements:")
    print("- Multi-step filtering: Gaussian(σ=1.0) + Median(3x3) + Gaussian(σ=0.3)")
    print("- PLASMA colormap (smoother than VIRIDIS)")
    print("- Bilinear interpolation throughout")
    print("="*80)
    
    # 重新生成超平滑V向量
    success = generate_ultra_smooth_v_vectors()
    
    if success:
        # 创建平滑度对比图
        comparison_path = create_smoothness_comparison()
        
        # 创建单层详细对比
        detail_path = create_single_layer_detail_comparison()
        
        print("\n" + "="*80)
        print("SUCCESS: Ultra-smooth V-vector generation completed!")
        print("="*80)
        print("Generated files:")
        print(f"- Smoothness comparison: {comparison_path}")
        if detail_path:
            print(f"- Detail comparison: {detail_path}")
        print(f"- Ultra-smooth analysis: analysis_results/value_analysis_ultra_smooth/")
        print("\nThe grid artifacts should now be completely eliminated!")
        print("V-vectors now use PLASMA colormap with multi-step smoothing.")
    else:
        print("\nERROR: Failed to generate ultra-smooth V-vectors")
