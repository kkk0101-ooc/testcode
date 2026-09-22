"""创建颜色映射演示"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def create_colormap_demo():
    """创建颜色映射演示图"""
    print("Creating colormap demonstration...")
    
    # 创建测试数据 - 模拟注意力图
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    X, Y = np.meshgrid(x, y)
    
    # 创建一个类似注意力图的模式
    attention_pattern = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.1) + \
                       0.5 * np.exp(-((X-0.3)**2 + (Y-0.3)**2) / 0.05) + \
                       0.3 * np.exp(-((X-0.7)**2 + (Y-0.7)**2) / 0.03)
    
    # 归一化到0-255
    attention_uint8 = ((attention_pattern - attention_pattern.min()) / 
                      (attention_pattern.max() - attention_pattern.min()) * 255).astype(np.uint8)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Colormap Comparison for ViT NNS Visualization\nFrom Yellow-Red to Blue-White', 
                 fontsize=18, fontweight='bold')
    
    # 不同的colormap
    colormaps = [
        ('JET (Previous)', cv2.COLORMAP_JET, 'jet'),
        ('PLASMA (Previous)', cv2.COLORMAP_PLASMA, 'plasma'),  
        ('WINTER (New)', cv2.COLORMAP_WINTER, 'winter'),
        ('COOL (Alternative)', cv2.COLORMAP_COOL, 'cool')
    ]
    
    for i, (name, cv2_cmap, plt_cmap) in enumerate(colormaps):
        # OpenCV版本 (实际使用的)
        cv2_result = cv2.applyColorMap(attention_uint8, cv2_cmap)
        cv2_result_rgb = cv2.cvtColor(cv2_result, cv2.COLOR_BGR2RGB)
        
        axes[0, i].imshow(cv2_result_rgb)
        axes[0, i].set_title(f'{name}\n(OpenCV Implementation)', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Matplotlib版本 (参考)
        axes[1, i].imshow(attention_pattern, cmap=plt_cmap)
        axes[1, i].set_title(f'{name}\n(Matplotlib Reference)', fontsize=12)
        axes[1, i].axis('off')
        
        if i == 2:  # WINTER (推荐)
            for ax in [axes[0, i], axes[1, i]]:
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', 
                                         linewidth=3, transform=ax.transAxes))
    
    # 添加说明
    fig.text(0.02, 0.75, 'Actual\nImplementation\n(OpenCV)', rotation=90, va='center', 
             fontsize=14, fontweight='bold')
    fig.text(0.02, 0.25, 'Reference\n(Matplotlib)', rotation=90, va='center', 
             fontsize=14, fontweight='bold')
    
    # 添加技术说明
    fig.text(0.5, 0.02, 
             'Blue box indicates RECOMMENDED colormap: WINTER provides blue-to-white gradient, '
             'professional appearance, and better visual comfort compared to yellow-red schemes.',
             ha='center', fontsize=12, style='italic', color='blue')
    
    plt.tight_layout()
    
    # 保存演示图
    demo_path = "analysis_results/COLORMAP_DEMO.png"
    plt.savefig(demo_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: Colormap demo saved: {demo_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(demo_path)
            print("SUCCESS: Demo opened automatically")
        except:
            pass
    
    return demo_path

def create_colorbar_comparison():
    """创建颜色条对比"""
    print("Creating colorbar comparison...")
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    fig.suptitle('Colormap Gradients Comparison', fontsize=16, fontweight='bold')
    
    # 创建渐变数据
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    
    colormaps = [
        ('JET (Yellow-Red, Previous)', 'jet'),
        ('PLASMA (Purple-Yellow, Previous)', 'plasma'),
        ('WINTER (Blue-White, NEW)', 'winter'),
        ('COOL (Cyan-Magenta, Alternative)', 'cool')
    ]
    
    for i, (name, cmap) in enumerate(colormaps):
        im = axes[i].imshow(gradient, aspect='auto', cmap=cmap)
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xlim(0, 255)
        axes[i].set_xticks([0, 64, 128, 192, 255])
        axes[i].set_xticklabels(['Low\nActivation', 'Low-Med', 'Medium', 'Med-High', 'High\nActivation'])
        axes[i].set_yticks([])
        
        if i == 2:  # WINTER (新推荐)
            axes[i].set_ylabel('RECOMMENDED', fontsize=12, fontweight='bold', color='blue')
            # 添加边框
            for spine in axes[i].spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(3)
    
    plt.tight_layout()
    
    # 保存颜色条对比
    colorbar_path = "analysis_results/COLORBAR_COMPARISON.png"
    plt.savefig(colorbar_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: Colorbar comparison saved: {colorbar_path}")
    
    return colorbar_path

def main():
    """主函数"""
    print("="*80)
    print("Colormap Change: Yellow-Red to Blue-White")
    print("="*80)
    print("Changes made:")
    print("- Attention maps: JET colormap -> WINTER colormap")
    print("- V-vector maps: PLASMA colormap -> WINTER colormap")
    print("- Effect: Yellow-red tones -> Blue-white tones")
    print("="*80)
    
    # 创建演示图
    demo_path = create_colormap_demo()
    
    # 创建颜色条对比
    colorbar_path = create_colorbar_comparison()
    
    print("\n" + "="*80)
    print("SUCCESS: Colormap demonstration completed!")
    print("="*80)
    print("Generated files:")
    print(f"- Colormap demo: {demo_path}")
    print(f"- Colorbar comparison: {colorbar_path}")
    print("\nTo apply the blue-white colormap to your images:")
    print("1. The colormap has been changed in the code")
    print("2. Re-run the NNS scan to generate new images")
    print("3. New images will use blue-white color scheme")
    print("\nBenefits of blue-white colormap:")
    print("- More professional appearance")
    print("- Easier on the eyes")
    print("- Better for presentations")
    print("- Cool, calming colors")

if __name__ == "__main__":
    main()
