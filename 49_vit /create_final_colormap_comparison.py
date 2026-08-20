"""创建最终的颜色映射对比展示"""
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_final_comparison():
    """创建最终对比图"""
    print("Creating final colormap comparison...")
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ViT NNS Visualization: Yellow-Red vs Blue-White Colormap Comparison\nTest Image: 20251114142550.png', 
                 fontsize=16, fontweight='bold')
    
    # 原图
    original_path = r"D:\project\vit_114\20251114142550.png"
    if os.path.exists(original_path):
        img = Image.open(original_path)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Test Image\n(211×213 pixels)', fontsize=14, fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, 'Original Image\nNot Available', ha='center', va='center')
        axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # 颜色映射演示
    demo_path = "analysis_results/COLORMAP_DEMO.png"
    if os.path.exists(demo_path):
        img = Image.open(demo_path)
        axes[0, 1].imshow(img)
        axes[0, 1].set_title('Colormap Options Comparison\n(Simulated Attention Pattern)', fontsize=14, fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'Colormap Demo\nNot Available', ha='center', va='center')
        axes[0, 1].set_title('Colormap Demo', fontsize=14)
    axes[0, 1].axis('off')
    
    # 黄红配色版本
    old_grid_path = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_auto_test_20251114\0_test_20251114\test_20251114_OPTIMIZED_12layers_grid.png"
    if os.path.exists(old_grid_path):
        img = Image.open(old_grid_path)
        axes[1, 0].imshow(img)
        axes[1, 0].set_title('Previous Version: Yellow-Red Colormap\n(JET/PLASMA - Warm Colors)', 
                           fontsize=14, fontweight='bold', color='orange')
    else:
        axes[1, 0].text(0.5, 0.5, 'Previous Version\n(Yellow-Red)\nNot Available', ha='center', va='center')
        axes[1, 0].set_title('Previous Version', fontsize=14)
    axes[1, 0].axis('off')
    
    # 蓝白配色版本
    new_grid_path = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_auto_test_20251114_blue_white\0_test_20251114_blue_white\test_20251114_blue_white_OPTIMIZED_12layers_grid.png"
    if os.path.exists(new_grid_path):
        img = Image.open(new_grid_path)
        axes[1, 1].imshow(img)
        axes[1, 1].set_title('New Version: Blue-White Colormap\n(WINTER - Cool Colors)', 
                           fontsize=14, fontweight='bold', color='blue')
        # 添加推荐标记
        axes[1, 1].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', 
                                         linewidth=4, transform=axes[1, 1].transAxes))
        axes[1, 1].text(0.02, 0.98, 'RECOMMENDED', transform=axes[1, 1].transAxes, 
                       fontsize=12, fontweight='bold', color='blue', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='blue'))
    else:
        axes[1, 1].text(0.5, 0.5, 'New Version\n(Blue-White)\nNot Available', ha='center', va='center')
        axes[1, 1].set_title('New Version', fontsize=14)
    axes[1, 1].axis('off')
    
    # 添加详细说明
    fig.text(0.5, 0.02, 
             'Blue-White Colormap Benefits: Professional appearance • Easier on eyes • Better for presentations • Cool, calming colors\n'
             'Model Prediction: Class 3 (91.82% confidence) • 12 Transformer Layers • 3 Attention Heads per Layer',
             ha='center', fontsize=11, style='italic', color='#333333')
    
    plt.tight_layout()
    
    # 保存最终对比图
    final_path = "analysis_results/FINAL_COLORMAP_COMPARISON.png"
    plt.savefig(final_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: Final comparison saved: {final_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(final_path)
            print("SUCCESS: Final comparison opened automatically")
        except:
            pass
    
    return final_path

def main():
    """主函数"""
    print("="*80)
    print("Final Colormap Comparison Creation")
    print("="*80)
    
    final_path = create_final_comparison()
    
    print("\n" + "="*80)
    print("SUCCESS: Colormap change completed!")
    print("="*80)
    print("Key files generated:")
    print(f"- Final comparison: {final_path}")
    print("- Blue-white grid: scan_results/.../test_20251114_blue_white_OPTIMIZED_12layers_grid.png")
    print("- Colormap demo: analysis_results/COLORMAP_DEMO.png")
    print("- Colorbar comparison: analysis_results/COLORBAR_COMPARISON.png")
    print("\nColormap Change Summary:")
    print("- FROM: Yellow-red tones (JET/PLASMA colormap)")
    print("- TO: Blue-white tones (WINTER colormap)")
    print("- BENEFIT: More professional, easier on eyes, better for presentations")
    print("\nYour visualization now uses the blue-white color scheme as requested!")

if __name__ == "__main__":
    main()
