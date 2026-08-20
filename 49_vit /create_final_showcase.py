"""创建最终展示效果图 - 对比两张测试图的完整分析结果"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import matplotlib.patches as patches

def create_final_showcase():
    """创建最终的展示效果图"""
    
    print("创建最终展示效果图...")
    
    # 图像路径
    fish_image = r"d:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG"
    new_image = r"d:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\scanimg\n02085620\n02085620_1.JPEG"
    
    # 优化网格图路径
    fish_grid = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final_fixed\0_n01443537_24_fixed\fish_OPTIMIZED_12layers_grid.png"
    new_grid = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final_new\0_n02085620_1_new\new_image_OPTIMIZED_12layers_grid.png"
    
    # 分析结果图路径
    attention_layer_changes = r".\analysis_results\attention_analysis\layer_progression\layer_changes.png"
    attention_entropy = r".\analysis_results\attention_analysis\quantitative\entropy_analysis.png"
    value_layer_changes = r".\analysis_results\value_analysis\layer_progression\layer_changes.png"
    value_entropy = r".\analysis_results\value_analysis\quantitative\entropy_analysis.png"
    
    # 创建大型展示图 (24x16 inches, 高分辨率)
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.2)
    
    # 标题
    fig.suptitle('ViT NNS Complete Analysis Showcase\nVision Transformer Neural Network Scanning Results', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # === 第一行：原始图像和基本信息 ===
    
    # 鱼图像
    ax1 = fig.add_subplot(gs[0, 0])
    if os.path.exists(fish_image):
        img = Image.open(fish_image).convert('RGB')
        ax1.imshow(img)
        ax1.set_title('Fish Image\n(n01443537_24.JPEG)\nClass 0, 100% Confidence', 
                      fontsize=12, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # 新图像
    ax2 = fig.add_subplot(gs[0, 1])
    if os.path.exists(new_image):
        img = Image.open(new_image).convert('RGB')
        ax2.imshow(img)
        ax2.set_title('New Image\n(n02085620_1.JPEG)\nClass 6, 76.78% Confidence', 
                      fontsize=12, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # 模型信息
    ax3 = fig.add_subplot(gs[0, 2:4])
    ax3.text(0.1, 0.8, 'Model Architecture', fontsize=16, fontweight='bold')
    ax3.text(0.1, 0.65, '• Vision Transformer (ViT)', fontsize=12)
    ax3.text(0.1, 0.55, '• Depth: 12 layers', fontsize=12)
    ax3.text(0.1, 0.45, '• Attention Heads: 3 per layer', fontsize=12)
    ax3.text(0.1, 0.35, '• Patch Size: 4×4', fontsize=12)
    ax3.text(0.1, 0.25, '• Training: 200 epochs', fontsize=12)
    ax3.text(0.1, 0.15, '• Final Accuracy: 43.8%', fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # 优化信息
    ax4 = fig.add_subplot(gs[0, 4:6])
    ax4.text(0.1, 0.8, 'NNS Optimization', fontsize=16, fontweight='bold')
    ax4.text(0.1, 0.65, '• Contrast Enhancement: 10x boost', fontsize=12)
    ax4.text(0.1, 0.55, '• Resolution: 32×32 → 512×512', fontsize=12)
    ax4.text(0.1, 0.45, '• Method: Percentile (5%-95%)', fontsize=12)
    ax4.text(0.1, 0.35, '• Best Head Selection', fontsize=12)
    ax4.text(0.1, 0.25, '• Cubic Interpolation', fontsize=12)
    ax4.text(0.1, 0.15, '• Clear Object Contours', fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # === 第二行：优化的12层网格图 ===
    
    # 鱼图像网格
    ax5 = fig.add_subplot(gs[1, 0:3])
    if os.path.exists(fish_grid):
        img = Image.open(fish_grid)
        ax5.imshow(img)
        ax5.set_title('Fish Image - 12-Layer Attention Evolution\n(Optimized with 10x Contrast Enhancement)', 
                      fontsize=14, fontweight='bold', pad=15)
    else:
        ax5.text(0.5, 0.5, 'Fish Grid\nNot Available', ha='center', va='center', fontsize=12)
        ax5.set_title('Fish Image - 12-Layer Grid', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # 新图像网格
    ax6 = fig.add_subplot(gs[1, 3:6])
    if os.path.exists(new_grid):
        img = Image.open(new_grid)
        ax6.imshow(img)
        ax6.set_title('New Image - 12-Layer Attention Evolution\n(Optimized with 10x Contrast Enhancement)', 
                      fontsize=14, fontweight='bold', pad=15)
    else:
        ax6.text(0.5, 0.5, 'New Image Grid\nNot Available', ha='center', va='center', fontsize=12)
        ax6.set_title('New Image - 12-Layer Grid', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # === 第三行：注意力分析结果 ===
    
    # 注意力层间变化
    ax7 = fig.add_subplot(gs[2, 0:2])
    if os.path.exists(attention_layer_changes):
        img = Image.open(attention_layer_changes)
        ax7.imshow(img)
        ax7.set_title('Attention Layer-to-Layer Changes\n(Quantitative Analysis)', 
                      fontsize=12, fontweight='bold', pad=10)
    else:
        ax7.text(0.5, 0.5, 'Attention\nLayer Changes\nNot Available', ha='center', va='center', fontsize=10)
        ax7.set_title('Attention Layer Changes', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # 注意力熵分析
    ax8 = fig.add_subplot(gs[2, 2:4])
    if os.path.exists(attention_entropy):
        img = Image.open(attention_entropy)
        ax8.imshow(img)
        ax8.set_title('Attention Information Distribution\n(Shannon Entropy Analysis)', 
                      fontsize=12, fontweight='bold', pad=10)
    else:
        ax8.text(0.5, 0.5, 'Attention\nEntropy Analysis\nNot Available', ha='center', va='center', fontsize=10)
        ax8.set_title('Attention Entropy', fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    # 注意力发现总结
    ax9 = fig.add_subplot(gs[2, 4:6])
    ax9.text(0.05, 0.9, 'Key Attention Findings', fontsize=14, fontweight='bold')
    ax9.text(0.05, 0.75, '• Shallow Layers (02-10):', fontsize=11, fontweight='bold')
    ax9.text(0.1, 0.68, 'Local textures and edges', fontsize=10)
    ax9.text(0.05, 0.58, '• Middle Layers (14-30):', fontsize=11, fontweight='bold')
    ax9.text(0.1, 0.51, 'Object parts and contours', fontsize=10)
    ax9.text(0.05, 0.41, '• Deep Layers (34-46):', fontsize=11, fontweight='bold')
    ax9.text(0.1, 0.34, 'Global semantic understanding', fontsize=10)
    ax9.text(0.05, 0.24, '• Multi-Head Specialization:', fontsize=11, fontweight='bold')
    ax9.text(0.1, 0.17, 'Different heads focus on different', fontsize=10)
    ax9.text(0.1, 0.12, 'aspects (edges, textures, shapes)', fontsize=10)
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    # === 第四行：V向量分析结果 ===
    
    # V向量层间变化
    ax10 = fig.add_subplot(gs[3, 0:2])
    if os.path.exists(value_layer_changes):
        img = Image.open(value_layer_changes)
        ax10.imshow(img)
        ax10.set_title('Value Vector Layer-to-Layer Changes\n(CLS vs Mean Analysis)', 
                       fontsize=12, fontweight='bold', pad=10)
    else:
        ax10.text(0.5, 0.5, 'Value Vector\nLayer Changes\nNot Available', ha='center', va='center', fontsize=10)
        ax10.set_title('Value Layer Changes', fontsize=12, fontweight='bold')
    ax10.axis('off')
    
    # V向量熵分析
    ax11 = fig.add_subplot(gs[3, 2:4])
    if os.path.exists(value_entropy):
        img = Image.open(value_entropy)
        ax11.imshow(img)
        ax11.set_title('Value Vector Information Distribution\n(Shannon Entropy Analysis)', 
                       fontsize=12, fontweight='bold', pad=10)
    else:
        ax11.text(0.5, 0.5, 'Value Vector\nEntropy Analysis\nNot Available', ha='center', va='center', fontsize=10)
        ax11.set_title('Value Entropy', fontsize=12, fontweight='bold')
    ax11.axis('off')
    
    # V向量发现总结
    ax12 = fig.add_subplot(gs[3, 4:6])
    ax12.text(0.05, 0.9, 'Key Value Vector Findings', fontsize=14, fontweight='bold')
    ax12.text(0.05, 0.75, '• CLS vs Mean Differences:', fontsize=11, fontweight='bold')
    ax12.text(0.1, 0.68, 'Significant variations between', fontsize=10)
    ax12.text(0.1, 0.63, 'CLS and Mean token vectors', fontsize=10)
    ax12.text(0.05, 0.53, '• Feature Localization:', fontsize=11, fontweight='bold')
    ax12.text(0.1, 0.46, 'Clear identification of key', fontsize=10)
    ax12.text(0.1, 0.41, 'activation regions', fontsize=10)
    ax12.text(0.05, 0.31, '• Anomaly Detection:', fontsize=11, fontweight='bold')
    ax12.text(0.1, 0.24, 'Discovered interesting', fontsize=10)
    ax12.text(0.1, 0.19, 'activation patterns', fontsize=10)
    ax12.text(0.05, 0.09, '• Evolution Patterns:', fontsize=11, fontweight='bold')
    ax12.text(0.1, 0.02, 'Progressive specialization', fontsize=10)
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    
    # 添加装饰性边框
    for ax in [ax1, ax2, ax5, ax6, ax7, ax8, ax10, ax11]:
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='#3498db', 
                               facecolor='none', transform=ax.transAxes)
        ax.add_patch(rect)
    
    # 添加底部信息
    fig.text(0.5, 0.02, 
             'ViT NNS Analysis System | 12-Layer Vision Transformer | Neural Network Scanning with Enhanced Visualization | '
             'Contrast: 10x Enhancement | Resolution: 512×512 | Method: Percentile + Best Head Selection',
             ha='center', fontsize=10, style='italic', color='#666666')
    
    # 保存高质量图像
    output_path = "analysis_results/FINAL_SHOWCASE_COMPARISON.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"SUCCESS: Final showcase saved: {output_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(output_path)
            print("SUCCESS: Showcase opened automatically")
        except:
            pass
    
    return output_path

def create_summary_statistics():
    """创建统计总结图"""
    
    print("创建统计总结...")
    
    # 创建统计图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ViT NNS Analysis Statistics Summary', fontsize=20, fontweight='bold')
    
    # 对比度提升统计
    ax1 = axes[0, 0]
    methods = ['Original', 'Percentile', 'Adaptive', 'Power', 'Sigmoid']
    contrasts = [0.027, 0.284, 0.251, 0.200, 0.062]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
    
    bars = ax1.bar(methods, contrasts, color=colors, alpha=0.8)
    ax1.set_title('Contrast Enhancement Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Contrast Value')
    ax1.set_ylim(0, 0.3)
    
    # 添加数值标签
    for bar, contrast in zip(bars, contrasts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{contrast:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 分辨率提升统计
    ax2 = axes[0, 1]
    resolutions = ['Original\n32×32', 'Enhanced\n512×512']
    pixels = [32*32, 512*512]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax2.bar(resolutions, pixels, color=colors, alpha=0.8)
    ax2.set_title('Resolution Enhancement', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Pixels')
    ax2.set_yscale('log')
    
    # 添加数值标签
    for bar, pixel in zip(bars, pixels):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{pixel:,}', ha='center', va='bottom', fontweight='bold')
    
    # 层级分析统计
    ax3 = axes[1, 0]
    layers = ['Shallow\n(02-10)', 'Middle\n(14-30)', 'Deep\n(34-46)']
    layer_counts = [3, 5, 4]  # 每个层级的层数
    colors = ['#3498db', '#f39c12', '#e74c3c']
    
    bars = ax3.bar(layers, layer_counts, color=colors, alpha=0.8)
    ax3.set_title('Layer Distribution Analysis', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Layers')
    
    # 添加功能标签
    functions = ['Local Features\n(Textures, Edges)', 'Object Parts\n(Contours, Shapes)', 'Global Semantics\n(Understanding)']
    for bar, count, func in zip(bars, layer_counts, functions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count} layers', ha='center', va='bottom', fontweight='bold')
        ax3.text(bar.get_x() + bar.get_width()/2., -0.3,
                func, ha='center', va='top', fontsize=9, style='italic')
    
    # 模型性能统计
    ax4 = axes[1, 1]
    metrics = ['Accuracy', 'Layers', 'Heads', 'Epochs']
    values = [43.8, 12, 3, 200]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6']
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
    ax4.set_title('Model Configuration', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Value')
    
    # 添加数值标签
    units = ['%', '', '', '']
    for bar, value, unit in zip(bars, values, units):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{value}{unit}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存统计图
    stats_path = "analysis_results/STATISTICS_SUMMARY.png"
    plt.savefig(stats_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: Statistics summary saved: {stats_path}")
    
    return stats_path

if __name__ == "__main__":
    print("="*80)
    print("Creating Final Showcase Images")
    print("="*80)
    
    # 创建主要展示图
    showcase_path = create_final_showcase()
    
    # 创建统计总结图
    stats_path = create_summary_statistics()
    
    print("\n" + "="*80)
    print("SUCCESS: Final showcase images created!")
    print("="*80)
    print(f"Main showcase: {showcase_path}")
    print(f"Statistics summary: {stats_path}")
    print("\nShowcase content:")
    print("- Two test images comparison (Fish vs New Image)")
    print("- 12-layer attention grids (optimized version)")
    print("- Attention analysis results (layer changes, entropy)")
    print("- V-vector analysis results (CLS vs Mean, localization)")
    print("- Key findings summary")
    print("- Technical specifications and optimization effects")
    print("\nComplete ViT NNS analysis showcase is ready!")
