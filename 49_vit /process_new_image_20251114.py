"""处理新图片 20251114142550.png 生成NNS可视化"""
import os
import sys
import subprocess
import shutil
from PIL import Image

def prepare_image():
    """准备图片"""
    print("="*80)
    print("准备新图片...")
    print("="*80)
    
    source_image = r"D:\project\vit_114\20251114142550.png"
    
    # 检查源图片是否存在
    if not os.path.exists(source_image):
        print(f"ERROR: 源图片不存在: {source_image}")
        return None
    
    # 显示图片信息
    try:
        img = Image.open(source_image)
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        print(f"Image format: {img.format}")
    except Exception as e:
        print(f"ERROR: 无法读取图片: {e}")
        return None
    
    # 创建目标目录
    target_dir = r"d:\project\vit_114\vitnns\49_vit\test_images"
    os.makedirs(target_dir, exist_ok=True)
    
    # 复制图片到测试目录
    target_image = os.path.join(target_dir, "new_test_image.png")
    shutil.copy2(source_image, target_image)
    print(f"SUCCESS: Image copied to: {target_image}")
    
    return target_image

def run_nns_scan(image_path):
    """运行NNS扫描"""
    print("="*80)
    print("运行NNS扫描...")
    print("="*80)
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 使用现有的depth=12模型进行扫描
    model_path = r"d:\project\vit_114\NNS_model_and_dataset\models\vit_3channel_mutlihead\tiny200ILSVRC_VisionTransformer_32imgsize_10classes_45threshold_3heads_12depth_4patch_size_VisionTransformer_200_final.pth"
    
    # 创建临时扫描脚本
    scan_script = """
import sys
sys.path.append('.')
from NNS.Scannerpic1107_cls_nonet import NNS_NONET
import torch
from torchvision import transforms
from PIL import Image
import os

def scan_single_image():
    # 加载模型
    model_path = r"{model_path}"
    image_path = r"{image_path}"
    
    # 加载图像
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    print(f"图像尺寸: {{image_tensor.shape}}")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # 创建NNS扫描器（使用正确的参数）
    scanner = NNS_NONET(
        network=model,
        imageTensor=image_tensor,
        img_name="new_test_image",
        id=0,
        device=device,
        model_name="VisionTransformer_200_final",
        model_type="tiny200ILSVRC",
        dataset="114VisionTransformer_tiny200ILSVRC",
        patch_size=4,
        num_heads=3,
        depth=12,
        params=None
    )
    
    # 确保目录存在
    os.makedirs(scanner.rootdir, exist_ok=True)
    os.makedirs(scanner.scan_attn_map, exist_ok=True)
    os.makedirs(scanner.scan_v, exist_ok=True)
    
    # 加载图像
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    print(f"图像尺寸: {{image_tensor.shape}}")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    print("开始NNS扫描...")
    
    # 运行扫描
    with torch.no_grad():
        scanner.scan_cls_nonet(model, image_tensor, 0, "new_test_image")
    
    print("NNS扫描完成!")
    return scanner.rootdir

if __name__ == "__main__":
    result_dir = scan_single_image()
    print(f"结果保存在: {{result_dir}}")
""".format(model_path=model_path, image_path=image_path)
    
    # 保存临时脚本
    temp_script = "temp_scan_new_image.py"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(scan_script)
    
    # 运行扫描
    print("执行NNS扫描...")
    result = subprocess.run([python_path, temp_script], 
                          capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    # 清理临时脚本
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    if result.returncode == 0:
        print("SUCCESS: NNS scan completed")
        print(result.stdout)
        return True
    else:
        print("ERROR: NNS scan failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

def create_optimized_grid():
    """创建优化的12层网格图"""
    print("="*80)
    print("创建优化网格图...")
    print("="*80)
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 创建网格图生成脚本
    grid_script = """
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def create_12layer_grid():
    base_dir = './scan_results/scan_vit/114VisionTransformer_tiny200ILSVRC/VisionTransformer_200_final_new_test/0_new_test_image/scan_attn_map'
    
    if not os.path.exists(base_dir):
        print(f"ERROR: 扫描结果目录不存在: {{base_dir}}")
        return None
    
    # 查找所有层目录
    layer_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    layer_dirs.sort(key=int)
    
    print(f"找到层目录: {{layer_dirs}}")
    
    if len(layer_dirs) < 12:
        print(f"WARNING: 只找到{{len(layer_dirs)}}层，期望12层")
    
    # 创建3x4网格 (原图 + 11层注意力图)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('New Test Image - 12-Layer ViT Attention Evolution\\n(Optimized Visualization)', 
                 fontsize=16, fontweight='bold')
    
    # 原图 (左上角)
    original_path = r"{image_path}"
    if os.path.exists(original_path):
        img = Image.open(original_path)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, 'Original\\nImage', ha='center', va='center')
        axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # 注意力图 (其余11个位置)
    positions = [(0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3), (2,0), (2,1), (2,2), (2,3)]
    
    for i, (row, col) in enumerate(positions):
        if i < len(layer_dirs):
            layer = layer_dirs[i]
            
            # 查找最佳注意力图
            layer_path = os.path.join(base_dir, layer, 'input_attn')
            
            # 尝试不同的head
            best_img = None
            best_contrast = 0
            
            for head_idx in range(3):  # 3个注意力头
                head_dir = os.path.join(layer_path, f'head{{head_idx}}')
                if os.path.exists(head_dir):
                    avg_cls_path = os.path.join(head_dir, f'avg_cls_attn_head{{head_idx}}.png')
                    if os.path.exists(avg_cls_path):
                        # 计算对比度
                        img = cv2.imread(avg_cls_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            contrast = np.std(img)
                            if contrast > best_contrast:
                                best_contrast = contrast
                                best_img = avg_cls_path
            
            if best_img:
                img = Image.open(best_img)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f'Layer {{layer}}\\n(Contrast: {{best_contrast:.1f}})', 
                                       fontsize=10, fontweight='bold')
            else:
                axes[row, col].text(0.5, 0.5, f'Layer {{layer}}\\nNot Available', ha='center', va='center')
                axes[row, col].set_title(f'Layer {{layer}}', fontsize=10)
            
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存网格图
    output_path = './scan_results/scan_vit/114VisionTransformer_tiny200ILSVRC/VisionTransformer_200_final_new_test/new_test_image_12layers_grid.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"✓ 12层网格图已保存: {{output_path}}")
    return output_path

if __name__ == "__main__":
    result = create_12layer_grid()
    if result:
        print(f"SUCCESS: {{result}}")
    else:
        print("FAILED")
""".format(image_path=image_path)
    
    # 保存网格脚本
    temp_grid_script = "temp_create_grid.py"
    with open(temp_grid_script, 'w', encoding='utf-8') as f:
        f.write(grid_script)
    
    # 运行网格生成
    print("生成12层网格图...")
    result = subprocess.run([python_path, temp_grid_script], 
                          capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    # 清理临时脚本
    if os.path.exists(temp_grid_script):
        os.remove(temp_grid_script)
    
    if result.returncode == 0:
        print("SUCCESS: Grid generation completed")
        print(result.stdout)
        return True
    else:
        print("ERROR: Grid generation failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

def run_analysis():
    """运行注意力和V向量分析"""
    print("="*80)
    print("运行分析...")
    print("="*80)
    
    base_dir = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final_new_test\0_new_test_image"
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 运行注意力分析
    if os.path.exists("do_images/visualize_attention/analyze_vit_attention.py"):
        print("Step 1: 运行注意力分析...")
        result = subprocess.run([
            python_path, "do_images/visualize_attention/analyze_vit_attention.py",
            "--base_dir", base_dir,
            "--output_dir", "analysis_results/new_test_attention_analysis"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("SUCCESS: Attention analysis completed")
        else:
            print(f"ERROR: Attention analysis failed: {result.stderr}")
    
    # 运行V向量分析（使用超平滑版本）
    if os.path.exists("do_images/visualize_v/analyze_vit_values.py"):
        print("Step 2: 运行V向量分析...")
        result = subprocess.run([
            python_path, "do_images/visualize_v/analyze_vit_values.py",
            "--base_dir", base_dir,
            "--output_dir", "analysis_results/new_test_value_analysis"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print("SUCCESS: V-vector analysis completed")
        else:
            print(f"ERROR: V-vector analysis failed: {result.stderr}")

def create_final_showcase():
    """创建最终展示图"""
    print("="*80)
    print("创建最终展示图...")
    print("="*80)
    
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # 创建展示图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('New Test Image (20251114142550.png) - Complete ViT NNS Analysis', 
                 fontsize=18, fontweight='bold')
    
    # 原图
    original_path = r"D:\project\vit_114\20251114142550.png"
    if os.path.exists(original_path):
        img = Image.open(original_path)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Test Image\n(20251114142550.png)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 12层网格图
    grid_path = r".\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\VisionTransformer_200_final_new_test\new_test_image_12layers_grid.png"
    if os.path.exists(grid_path):
        img = Image.open(grid_path)
        axes[0, 1].imshow(img)
        axes[0, 1].set_title('12-Layer Attention Grid\n(Optimized Visualization)', fontsize=12, fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, '12-Layer Grid\nNot Available', ha='center', va='center')
        axes[0, 1].set_title('12-Layer Attention Grid', fontsize=12)
    axes[0, 1].axis('off')
    
    # 分析结果
    analysis_paths = [
        ("analysis_results/new_test_attention_analysis/layer_progression/head0_progression.png", "Attention Head 0 Progression"),
        ("analysis_results/new_test_value_analysis/layer_progression/head0_progression.png", "Value Vector Head 0 Progression"),
        ("analysis_results/new_test_attention_analysis/quantitative/entropy_analysis.png", "Attention Entropy Analysis"),
        ("analysis_results/new_test_value_analysis/layer_progression/layer_changes.png", "Value Vector Layer Changes")
    ]
    
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
    
    for (path, title), (row, col) in zip(analysis_paths, positions):
        if os.path.exists(path):
            img = Image.open(path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=11, fontweight='bold')
        else:
            axes[row, col].text(0.5, 0.5, f'{title}\\nNot Available', ha='center', va='center')
            axes[row, col].set_title(title, fontsize=11)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存展示图
    showcase_path = "analysis_results/NEW_TEST_IMAGE_SHOWCASE.png"
    os.makedirs("analysis_results", exist_ok=True)
    plt.savefig(showcase_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"SUCCESS: Final showcase saved: {showcase_path}")
    
    # 自动打开
    import platform
    if platform.system() == 'Windows':
        try:
            os.startfile(showcase_path)
            print("SUCCESS: Showcase opened automatically")
        except:
            pass
    
    return showcase_path

def main():
    """主函数"""
    print("="*80)
    print("处理新测试图片: 20251114142550.png")
    print("="*80)
    
    # Step 1: 准备图片
    image_path = prepare_image()
    if not image_path:
        return
    
    # Step 2: 运行NNS扫描
    if not run_nns_scan(image_path):
        print("NNS扫描失败，退出")
        return
    
    # Step 3: 创建网格图
    if not create_optimized_grid():
        print("网格图创建失败")
    
    # Step 4: 运行分析
    run_analysis()
    
    # Step 5: 创建最终展示
    showcase_path = create_final_showcase()
    
    print("\n" + "="*80)
    print("SUCCESS: 新测试图片分析完成!")
    print("="*80)
    print("生成的文件:")
    print("• 12层网格图: scan_results/.../new_test_image_12layers_grid.png")
    print("• 注意力分析: analysis_results/new_test_attention_analysis/")
    print("• V向量分析: analysis_results/new_test_value_analysis/")
    print(f"• 最终展示图: {showcase_path}")
    print("\n新测试图片的完整ViT NNS分析结果已准备就绪!")

if __name__ == "__main__":
    main()
