"""重新生成蓝白配色的可视化效果"""
import os
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def regenerate_with_blue_white():
    """使用蓝白配色重新生成测试图片的效果"""
    print("="*80)
    print("重新生成蓝白配色的可视化效果")
    print("="*80)
    
    # 重新运行NNS扫描（使用新的蓝白配色）
    print("Step 1: 重新运行NNS扫描（蓝白配色）...")
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    image_path = r"D:\project\vit_114\20251114142550.png"
    
    result = subprocess.run([
        python_path, "quick_generate_for_new_image.py",
        "--image_path", image_path,
        "--output_name", "test_20251114_blue_white"
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    if result.returncode == 0:
        print("SUCCESS: 蓝白配色NNS扫描完成")
        print(result.stdout)
    else:
        print("ERROR: NNS扫描失败")
        print(result.stderr)
        return False
    
    return True

def create_colormap_comparison():
    """创建不同颜色映射的对比图"""
    print("\nStep 2: 创建颜色映射对比图...")
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Color Mapping Comparison: Different Colormap Effects', 
                 fontsize=18, fontweight='bold')
    
    # 原图
    original_path = r"D:\project\vit_114\20251114142550.png"
    if os.path.exists(original_path):
        img = Image.open(original_path)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Test Image\n(20251114142550.png)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 旧版本（黄色配色）
    old_grid_path = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_auto_test_20251114\0_test_20251114\test_20251114_OPTIMIZED_12layers_grid.png"
    if os.path.exists(old_grid_path):
        img = Image.open(old_grid_path)
        axes[0, 1].imshow(img)
        axes[0, 1].set_title('Previous Version\n(JET/PLASMA Colormap)\nYellow-Red Colors', 
                           fontsize=12, fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'Previous Version\nNot Available', ha='center', va='center')
        axes[0, 1].set_title('Previous Version', fontsize=12)
    axes[0, 1].axis('off')
    
    # 新版本（蓝白配色）
    new_grid_path = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_auto_test_20251114_blue_white\0_test_20251114_blue_white\test_20251114_blue_white_OPTIMIZED_12layers_grid.png"
    if os.path.exists(new_grid_path):
        img = Image.open(new_grid_path)
        axes[0, 2].imshow(img)
        axes[0, 2].set_title('New Version\n(WINTER Colormap)\nBlue-White Colors', 
                           fontsize=12, fontweight='bold')
    else:
        axes[0, 2].text(0.5, 0.5, 'New Version\nNot Available', ha='center', va='center')
        axes[0, 2].set_title('New Version', fontsize=12)
    axes[0, 2].axis('off')
    
    # 颜色映射说明
    axes[1, 0].text(0.1, 0.9, 'Available Colormaps', fontsize=14, fontweight='bold')
    axes[1, 0].text(0.1, 0.8, '• JET: Red-Yellow-Green-Blue', fontsize=11)
    axes[1, 0].text(0.1, 0.7, '• PLASMA: Purple-Pink-Yellow', fontsize=11)
    axes[1, 0].text(0.1, 0.6, '• WINTER: Blue-Cyan-White', fontsize=11, color='blue', fontweight='bold')
    axes[1, 0].text(0.1, 0.5, '• COOL: Cyan-Magenta', fontsize=11)
    axes[1, 0].text(0.1, 0.4, '• HOT: Black-Red-Yellow-White', fontsize=11)
    axes[1, 0].text(0.1, 0.3, '• BONE: Black-White (Grayscale)', fontsize=11)
    axes[1, 0].text(0.1, 0.2, '• OCEAN: Black-Blue-Cyan-White', fontsize=11)
    axes[1, 0].text(0.1, 0.1, '• SPRING: Magenta-Yellow', fontsize=11)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # 优势对比
    axes[1, 1].text(0.1, 0.9, 'Yellow-Red Colormap', fontsize=14, fontweight='bold', color='orange')
    axes[1, 1].text(0.1, 0.8, '+ High contrast', fontsize=11)
    axes[1, 1].text(0.1, 0.75, '+ Warm colors', fontsize=11)
    axes[1, 1].text(0.1, 0.7, '+ Traditional scientific visualization', fontsize=11)
    axes[1, 1].text(0.1, 0.6, '- Can be harsh on eyes', fontsize=11, color='red')
    axes[1, 1].text(0.1, 0.55, '- May cause visual fatigue', fontsize=11, color='red')
    
    axes[1, 1].text(0.1, 0.4, 'Blue-White Colormap', fontsize=14, fontweight='bold', color='blue')
    axes[1, 1].text(0.1, 0.3, '+ Cool, calming colors', fontsize=11, color='blue')
    axes[1, 1].text(0.1, 0.25, '+ Easy on the eyes', fontsize=11, color='blue')
    axes[1, 1].text(0.1, 0.2, '+ Professional appearance', fontsize=11, color='blue')
    axes[1, 1].text(0.1, 0.15, '+ Better for presentations', fontsize=11, color='blue')
    axes[1, 1].text(0.1, 0.05, '- Slightly lower contrast', fontsize=11)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    # 技术细节
    axes[1, 2].text(0.1, 0.9, 'Technical Details', fontsize=14, fontweight='bold')
    axes[1, 2].text(0.1, 0.8, 'WINTER Colormap:', fontsize=12, fontweight='bold', color='blue')
    axes[1, 2].text(0.1, 0.75, '• Low values: Dark blue', fontsize=11)
    axes[1, 2].text(0.1, 0.7, '• Medium values: Cyan/Light blue', fontsize=11)
    axes[1, 2].text(0.1, 0.65, '• High values: White', fontsize=11)
    axes[1, 2].text(0.1, 0.6, '• Smooth gradient', fontsize=11)
    axes[1, 2].text(0.1, 0.55, '• No grid artifacts', fontsize=11)
    
    axes[1, 2].text(0.1, 0.4, 'Implementation:', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.35, '• cv2.COLORMAP_WINTER', fontsize=10, family='monospace')
    axes[1, 2].text(0.1, 0.3, '• Applied to both attention', fontsize=11)
    axes[1, 2].text(0.1, 0.25, '  and V-vector maps', fontsize=11)
    axes[1, 2].text(0.1, 0.2, '• Consistent color scheme', fontsize=11)
    axes[1, 2].text(0.1, 0.15, '• High-quality interpolation', fontsize=11)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = "analysis_results/COLORMAP_COMPARISON.png"
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: 颜色映射对比图已保存: {comparison_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(comparison_path)
            print("SUCCESS: 对比图已自动打开")
        except:
            pass
    
    return comparison_path

def create_colormap_samples():
    """创建不同颜色映射的样本展示"""
    print("\nStep 3: 创建颜色映射样本...")
    
    # 创建测试数据
    x = np.linspace(0, 1, 256).reshape(1, -1)
    
    # 不同的colormap
    colormaps = [
        ('JET (Previous)', 'jet'),
        ('WINTER (New)', 'winter'),
        ('COOL (Alternative)', 'cool'),
        ('OCEAN (Alternative)', 'ocean')
    ]
    
    fig, axes = plt.subplots(len(colormaps), 1, figsize=(12, 8))
    fig.suptitle('Colormap Samples for ViT NNS Visualization', fontsize=16, fontweight='bold')
    
    for i, (name, cmap) in enumerate(colormaps):
        axes[i].imshow(x, aspect='auto', cmap=cmap)
        axes[i].set_title(f'{name} Colormap', fontsize=12, fontweight='bold')
        axes[i].set_xlim(0, 255)
        axes[i].set_xticks([0, 64, 128, 192, 255])
        axes[i].set_xticklabels(['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
        axes[i].set_yticks([])
        
        if i == 1:  # WINTER (新选择)
            axes[i].set_ylabel('← Recommended', fontsize=12, fontweight='bold', color='blue')
    
    plt.tight_layout()
    
    # 保存样本图
    samples_path = "analysis_results/COLORMAP_SAMPLES.png"
    plt.savefig(samples_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: 颜色映射样本已保存: {samples_path}")
    
    return samples_path

def main():
    """主函数"""
    print("="*80)
    print("颜色映射修改: 从黄色配色改为蓝白配色")
    print("="*80)
    print("修改内容:")
    print("• 注意力图: JET colormap → WINTER colormap")
    print("• V向量图: PLASMA colormap → WINTER colormap")
    print("• 效果: 黄红色调 → 蓝白色调")
    print("="*80)
    
    # Step 1: 重新生成
    success = regenerate_with_blue_white()
    
    # Step 2: 创建对比图
    comparison_path = create_colormap_comparison()
    
    # Step 3: 创建样本图
    samples_path = create_colormap_samples()
    
    print("\n" + "="*80)
    print("SUCCESS: 蓝白配色版本生成完成!")
    print("="*80)
    print("生成的文件:")
    print(f"• 颜色映射对比: {comparison_path}")
    print(f"• 颜色映射样本: {samples_path}")
    if success:
        print("• 新的蓝白配色网格图: scan_results/.../test_20251114_blue_white_OPTIMIZED_12layers_grid.png")
    print("\n现在您的可视化使用蓝白配色方案，更加清爽和专业!")
    print("蓝色表示低激活，白色表示高激活，过渡自然平滑。")

if __name__ == "__main__":
    main()
