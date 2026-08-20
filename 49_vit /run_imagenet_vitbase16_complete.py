"""完整的ImageNet + ViT-base-16 NNS分析流水线"""
import os
import sys
import subprocess
import time

def check_environment():
    """检查环境和依赖"""
    print("="*80)
    print("检查环境和依赖...")
    print("="*80)
    
    # 检查timm
    try:
        import timm
        print(f"✓ timm版本: {timm.__version__}")
    except ImportError:
        print("✗ timm未安装，正在安装...")
        subprocess.run([
            r"D:\Program Files\miniconda3\envs\vit_114\python.exe", 
            "-m", "pip", "install", "timm"
        ])
        print("✓ timm安装完成")
    
    # 检查ImageNet数据集
    imagenet_paths = [
        r"D:\project\vit_114\vitnns\49_vit\datas\imagenet-scanner",
        r"D:\project\vit_114\NNS_model_and_dataset\dataset\imagenet\val",
        r"D:\project\vit_114\NNS_model_and_dataset\dataset\ImageNet\val",
        r"D:\data\imagenet\val"
    ]
    
    imagenet_path = None
    for path in imagenet_paths:
        if os.path.exists(path):
            imagenet_path = path
            print(f"✓ 找到ImageNet数据集: {path}")
            break
    
    if not imagenet_path:
        print("✗ 未找到ImageNet数据集")
        print("请确保ImageNet验证集位于以下路径之一:")
        for path in imagenet_paths:
            print(f"  - {path}")
        return None
    
    return imagenet_path

def run_imagenet_vitbase16_nns(imagenet_path, count=5):
    """运行ImageNet + ViT-base-16 NNS扫描"""
    print("="*80)
    print(f"运行ImageNet + ViT-base-16 NNS扫描 (处理{count}张图像)")
    print("="*80)
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 运行NNS扫描
    cmd = [
        python_path, "train_scanner_cls.py",
        "--device", "0",
        "--count", str(count),
        "--dataset", "ImageNet",
        "--classes", "1000",
        "--imagenet_path", imagenet_path,
        "--process_wrong_classification", "True",
        "--nns_resize", "64"  # 使用64x64降采样节省内存
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print("\n开始处理...")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✓ NNS扫描完成 (耗时: {end_time-start_time:.1f}秒)")
        return True
    else:
        print(f"✗ NNS扫描失败:")
        print(result.stderr)
        return False

def create_analysis_grids():
    """创建分析网格图"""
    print("="*80)
    print("创建分析网格图...")
    print("="*80)
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 创建10层网格图
    if os.path.exists("create_imagenet_10grid.py"):
        print("Step 1: 创建10层网格图...")
        result = subprocess.run([python_path, "create_imagenet_10grid.py"], 
                              capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            print("✓ 10层网格图创建完成")
        else:
            print(f"✗ 10层网格图创建失败: {result.stderr}")
    
    # 创建层级对比图
    if os.path.exists("combine_layer_attention.py"):
        print("Step 2: 创建层级对比图...")
        result = subprocess.run([
            python_path, "combine_layer_attention.py",
            "--base_dir", "scan_results/scan_vit/114VisionTransformer_ImageNet",
            "--grid_size", "3", "4"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            print("✓ 层级对比图创建完成")
        else:
            print(f"✗ 层级对比图创建失败: {result.stderr}")

def run_imagenet_analysis():
    """运行ImageNet图像的注意力和V向量分析"""
    print("="*80)
    print("运行ImageNet图像分析...")
    print("="*80)
    
    # 查找ImageNet扫描结果目录
    base_pattern = "scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained"
    
    if not os.path.exists(base_pattern):
        print(f"✗ 未找到ImageNet扫描结果: {base_pattern}")
        return False
    
    # 查找具体的图像目录
    scan_dirs = [d for d in os.listdir(base_pattern) 
                if os.path.isdir(os.path.join(base_pattern, d)) and not d.endswith('.png')]
    
    if not scan_dirs:
        print("✗ 未找到具体的图像扫描目录")
        return False
    
    # 选择第一个目录进行分析
    target_dir = os.path.join(base_pattern, scan_dirs[0])
    print(f"分析目录: {target_dir}")
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 运行注意力分析
    if os.path.exists("do_images/visualize_attention/analyze_vit_attention.py"):
        print("Step 1: 运行注意力分析...")
        result = subprocess.run([
            python_path, "do_images/visualize_attention/analyze_vit_attention.py",
            "--base_dir", target_dir,
            "--output_dir", "analysis_results/imagenet_attention_analysis"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("✓ ImageNet注意力分析完成")
        else:
            print(f"✗ 注意力分析失败: {result.stderr}")
    
    # 运行V向量分析
    if os.path.exists("do_images/visualize_v/analyze_vit_values.py"):
        print("Step 2: 运行V向量分析...")
        result = subprocess.run([
            python_path, "do_images/visualize_v/analyze_vit_values.py",
            "--base_dir", target_dir,
            "--output_dir", "analysis_results/imagenet_value_analysis"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("✓ ImageNet V向量分析完成")
        else:
            print(f"✗ V向量分析失败: {result.stderr}")
    
    return True

def create_final_imagenet_showcase():
    """创建ImageNet最终展示图"""
    print("="*80)
    print("创建ImageNet最终展示图...")
    print("="*80)
    
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    # 创建展示图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ImageNet + ViT-base-16 Complete Analysis Results', fontsize=20, fontweight='bold')
    
    # 10层网格图
    grid_path = "scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained/imagenet_vit_base_10layers_grid.png"
    if os.path.exists(grid_path):
        img = Image.open(grid_path)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('10-Layer Attention Grid\n(ViT-base-16 on ImageNet)', fontsize=14, fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, '10-Layer Grid\nNot Available', ha='center', va='center')
        axes[0, 0].set_title('10-Layer Attention Grid', fontsize=14)
    axes[0, 0].axis('off')
    
    # 注意力分析结果
    attention_paths = [
        ("analysis_results/imagenet_attention_analysis/layer_progression/head0_progression.png", "Attention Head 0 Progression"),
        ("analysis_results/imagenet_attention_analysis/quantitative/entropy_analysis.png", "Attention Entropy Analysis"),
        ("analysis_results/imagenet_value_analysis/layer_progression/head0_progression.png", "Value Vector Head 0 Progression"),
        ("analysis_results/imagenet_value_analysis/layer_progression/layer_changes.png", "Value Vector Layer Changes")
    ]
    
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    
    for i, ((path, title), (row, col)) in enumerate(zip(attention_paths, positions)):
        if os.path.exists(path):
            img = Image.open(path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        else:
            axes[row, col].text(0.5, 0.5, f'{title}\nNot Available', ha='center', va='center')
            axes[row, col].set_title(title, fontsize=12)
        axes[row, col].axis('off')
    
    # 技术规格
    axes[1, 2].text(0.1, 0.9, 'ImageNet + ViT-base-16 Specs', fontsize=14, fontweight='bold')
    axes[1, 2].text(0.1, 0.8, '• Model: vit_base_patch16_224', fontsize=11)
    axes[1, 2].text(0.1, 0.7, '• Depth: 12 Transformer layers', fontsize=11)
    axes[1, 2].text(0.1, 0.6, '• Attention Heads: 12 per layer', fontsize=11)
    axes[1, 2].text(0.1, 0.5, '• Input Size: 224×224', fontsize=11)
    axes[1, 2].text(0.1, 0.4, '• Patch Size: 16×16', fontsize=11)
    axes[1, 2].text(0.1, 0.3, '• Classes: 1000 (ImageNet)', fontsize=11)
    axes[1, 2].text(0.1, 0.2, '• Pretrained: ImageNet-1k', fontsize=11)
    axes[1, 2].text(0.1, 0.1, '• NNS Processing: 64×64 resize', fontsize=11)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存展示图
    showcase_path = "analysis_results/IMAGENET_VITBASE16_SHOWCASE.png"
    os.makedirs("analysis_results", exist_ok=True)
    plt.savefig(showcase_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"✓ ImageNet展示图已保存: {showcase_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(showcase_path)
            print("✓ 展示图已自动打开")
        except:
            pass
    
    return showcase_path

def main():
    """主函数"""
    print("="*80)
    print("ImageNet + ViT-base-16 Complete Analysis Pipeline")
    print("="*80)
    print("这将执行:")
    print("1. 环境检查")
    print("2. NNS扫描 (ViT-base-16 + ImageNet)")
    print("3. 创建分析网格图")
    print("4. 运行注意力和V向量分析")
    print("5. 创建最终展示图")
    print("="*80)
    
    # Step 1: 检查环境
    imagenet_path = check_environment()
    if not imagenet_path:
        print("环境检查失败，退出")
        return
    
    # Step 2: 运行NNS扫描
    if not run_imagenet_vitbase16_nns(imagenet_path, count=3):
        print("NNS扫描失败，退出")
        return
    
    # Step 3: 创建网格图
    create_analysis_grids()
    
    # Step 4: 运行分析
    run_imagenet_analysis()
    
    # Step 5: 创建最终展示
    showcase_path = create_final_imagenet_showcase()
    
    print("\n" + "="*80)
    print("SUCCESS: ImageNet + ViT-base-16 Complete Analysis Finished!")
    print("="*80)
    print("生成的文件:")
    print("• 10层网格图: scan_results/.../imagenet_vit_base_10layers_grid.png")
    print("• 注意力分析: analysis_results/imagenet_attention_analysis/")
    print("• V向量分析: analysis_results/imagenet_value_analysis/")
    print(f"• 最终展示图: {showcase_path}")
    print("\n现在您拥有完整的ImageNet + ViT-base-16分析结果!")

if __name__ == "__main__":
    main()
