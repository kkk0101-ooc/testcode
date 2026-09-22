"""组织ImageNet验证集图片到类别目录"""
import os
from pathlib import Path
import shutil

def organize_imagenet_val():
    """将ImageNet验证集图片按类别组织"""
    val_dir = Path(r'D:\project\vit_114\NNS_model_and_dataset\dataset\imagenet_full\val')
    
    # 获取所有JPEG图片
    images = sorted(val_dir.glob('*.JPEG'))
    print(f'找到 {len(images)} 张图片')
    
    if len(images) > 0:
        print('按类别组织图片...')
        
        # ImageNet验证集有50000张图片，1000个类别，每类50张
        images_per_class = 50
        num_classes = 1000
        
        for class_idx in range(num_classes):
            # 创建类别目录
            class_name = f'n{class_idx:08d}'
            class_dir = val_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # 移动图片到类别目录
            start_idx = class_idx * images_per_class
            end_idx = min(start_idx + images_per_class, len(images))
            
            for img in images[start_idx:end_idx]:
                target = class_dir / img.name
                if not target.exists():
                    shutil.move(str(img), str(target))
            
            if (class_idx + 1) % 100 == 0:
                print(f'  已处理 {class_idx + 1}/{num_classes} 个类别')
        
        print('组织完成!')
    else:
        print('未找到JPEG图片，可能已经组织好了')
    
    # 统计结果
    class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob('*.JPEG'))) for d in class_dirs)
    print(f'最终统计: {len(class_dirs)} 个类别, {total_images} 张图片')

if __name__ == '__main__':
    organize_imagenet_val()

# 原有代码
import glob

def check_imagenet_availability():
    """检查ImageNet数据集的可用性"""
    print("="*80)
    print("检查ImageNet数据集情况")
    print("="*80)
    
    # 可能的ImageNet路径
    imagenet_paths = [
        r"D:\project\vit_114\NNS_model_and_dataset\dataset\imagenet\val",
        r"D:\project\vit_114\NNS_model_and_dataset\dataset\ImageNet\val",
        r"D:\data\imagenet\val",
        r"D:\ImageNet\val",
        r"C:\data\imagenet\val",
        r"C:\ImageNet\val"
    ]
    
    print("正在检查可能的ImageNet路径:")
    found_paths = []
    
    for path in imagenet_paths:
        print(f"  检查: {path}")
        if os.path.exists(path):
            # 检查是否包含类别目录
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            class_dirs = [d for d in subdirs if d.startswith('n')]
            
            if class_dirs:
                print(f"    ✓ 找到! 包含 {len(class_dirs)} 个类别目录")
                
                # 统计图片数量
                total_images = 0
                sample_classes = class_dirs[:5]  # 检查前5个类别
                for class_dir in sample_classes:
                    class_path = os.path.join(path, class_dir)
                    images = glob.glob(os.path.join(class_path, "*.JPEG")) + \
                            glob.glob(os.path.join(class_path, "*.jpg")) + \
                            glob.glob(os.path.join(class_path, "*.png"))
                    total_images += len(images)
                    print(f"      类别 {class_dir}: {len(images)} 张图片")
                
                avg_per_class = total_images / len(sample_classes) if sample_classes else 0
                estimated_total = avg_per_class * len(class_dirs)
                print(f"    估计总图片数: {estimated_total:.0f} 张")
                
                found_paths.append({
                    'path': path,
                    'classes': len(class_dirs),
                    'estimated_images': estimated_total
                })
            else:
                print(f"    - 存在但无类别目录")
        else:
            print(f"    - 不存在")
    
    return found_paths

def check_current_setup():
    """检查当前的设置情况"""
    print("\n" + "="*80)
    print("检查当前设置情况")
    print("="*80)
    
    # 检查tiny-imagenet设置
    tiny_imagenet_path = r"D:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5"
    print(f"Tiny-ImageNet路径: {tiny_imagenet_path}")
    
    if os.path.exists(tiny_imagenet_path):
        # 统计类别和图片
        train_path = os.path.join(tiny_imagenet_path, "train")
        val_path = os.path.join(tiny_imagenet_path, "val")
        test_path = os.path.join(tiny_imagenet_path, "test111")
        
        if os.path.exists(train_path):
            train_classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
            print(f"  训练集: {len(train_classes)} 个类别")
        
        if os.path.exists(val_path):
            val_classes = [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]
            print(f"  验证集: {len(val_classes)} 个类别")
        
        if os.path.exists(test_path):
            test_classes = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
            print(f"  测试集: {len(test_classes)} 个类别")
    else:
        print("  ✗ Tiny-ImageNet不存在")
    
    # 检查现有的ImageNet扫描数据
    imagenet_scanner_path = r"d:\project\vit_114\vitnns\49_vit\datas\imagenet-scanner"
    print(f"\nImageNet扫描数据: {imagenet_scanner_path}")
    
    if os.path.exists(imagenet_scanner_path):
        scanner_classes = [d for d in os.listdir(imagenet_scanner_path) if os.path.isdir(os.path.join(imagenet_scanner_path, d))]
        print(f"  可用类别: {len(scanner_classes)} 个")
        
        total_images = 0
        for class_dir in scanner_classes:
            class_path = os.path.join(imagenet_scanner_path, class_dir)
            images = glob.glob(os.path.join(class_path, "*.JPEG")) + \
                    glob.glob(os.path.join(class_path, "*.jpg"))
            total_images += len(images)
            print(f"    {class_dir}: {len(images)} 张图片")
        
        print(f"  总图片数: {total_images} 张")
    else:
        print("  ✗ ImageNet扫描数据不存在")

def analyze_migration_requirements():
    """分析迁移需求"""
    print("\n" + "="*80)
    print("分析从Tiny-ImageNet到ImageNet的迁移需求")
    print("="*80)
    
    print("当前Tiny-ImageNet设置:")
    print("  • 图像尺寸: 64×64 → 32×32 (模型输入)")
    print("  • 类别数: 200 → 10 (实际使用)")
    print("  • 模型: 自训练ViT (12层, 3头, patch_size=4)")
    print("  • 嵌入维度: 192")
    
    print("\nImageNet迁移需求:")
    print("  • 图像尺寸: 224×224 → 需要调整预处理")
    print("  • 类别数: 1000 → 需要新模型或预训练模型")
    print("  • 模型选择:")
    print("    - 选项1: 使用timm预训练ViT-base-16 (推荐)")
    print("    - 选项2: 重新训练现有架构")
    
    print("\n关键差异:")
    print("  1. 输入尺寸: 64×64 vs 224×224")
    print("  2. Patch大小: 4×4 vs 16×16")
    print("  3. 类别数量: 10 vs 1000")
    print("  4. 模型复杂度: 简化版 vs 标准版")

def create_migration_plan():
    """创建迁移计划"""
    print("\n" + "="*80)
    print("ImageNet迁移计划")
    print("="*80)
    
    print("方案A: 使用预训练ViT-base-16 (推荐)")
    print("  ✓ 优势: 预训练权重, 标准架构, 高性能")
    print("  ✓ 实现: 已有代码支持 (run_imagenet_vitbase16_complete.py)")
    print("  ✓ 数据: 可使用现有imagenet-scanner数据")
    print("  • 模型: timm.vit_base_patch16_224")
    print("  • 输入: 224×224")
    print("  • Patch: 16×16")
    print("  • 层数: 12")
    print("  • 头数: 12")
    
    print("\n方案B: 扩展现有模型")
    print("  • 优势: 保持现有架构")
    print("  • 挑战: 需要重新训练, 数据量需求大")
    print("  • 修改: 输入层, 位置编码, 分类头")
    
    print("\n推荐执行步骤:")
    print("  1. 使用方案A (预训练ViT-base-16)")
    print("  2. 利用现有imagenet-scanner数据 (10类, 100张图)")
    print("  3. 运行NNS可视化")
    print("  4. 生成ImageNet版本的分析结果")
    print("  5. 对比Tiny-ImageNet vs ImageNet的差异")

def main():
    """主函数"""
    print("ImageNet数据集检查和迁移分析")
    
    # 检查ImageNet数据集
    found_paths = check_imagenet_availability()
    
    # 检查当前设置
    check_current_setup()
    
    # 分析迁移需求
    analyze_migration_requirements()
    
    # 创建迁移计划
    create_migration_plan()
    
    print("\n" + "="*80)
    print("总结和建议")
    print("="*80)
    
    if found_paths:
        print("✓ 发现可用的ImageNet数据集:")
        for path_info in found_paths:
            print(f"  • {path_info['path']}: {path_info['classes']} 类, 约 {path_info['estimated_images']:.0f} 张图")
        print("\n建议: 使用完整ImageNet数据集 + 预训练ViT-base-16")
    else:
        print("• 未发现完整ImageNet数据集")
        print("• 但有imagenet-scanner数据 (10类, 约100张图)")
        print("\n建议: 使用现有imagenet-scanner数据 + 预训练ViT-base-16")
    
    print("\n下一步:")
    print("1. 运行: python run_imagenet_vitbase16_complete.py")
    print("2. 这将使用预训练ViT-base-16模型")
    print("3. 自动查找可用的ImageNet数据")
    print("4. 生成ImageNet版本的NNS可视化")
    print("5. 对比分析Tiny-ImageNet vs ImageNet的结果")

if __name__ == "__main__":
    main()
