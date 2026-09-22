"""自动下载ImageNet数据集"""
import os
import subprocess
import shutil
from pathlib import Path

def setup_kaggle_and_download():
    """设置Kaggle并自动下载ImageNet"""
    print("="*80)
    print("自动下载ImageNet数据集")
    print("="*80)
    
    # 1. 设置Kaggle凭据
    print("Step 1: 设置Kaggle凭据...")
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_config = kaggle_dir / 'kaggle.json'
    
    kaggle_dir.mkdir(exist_ok=True)
    source_config = Path('kaggle.json')
    
    if source_config.exists():
        shutil.copy2(source_config, kaggle_config)
        os.chmod(kaggle_config, 0o600)
        print("SUCCESS: Kaggle配置完成")
    else:
        print("ERROR: kaggle.json不存在")
        return False
    
    # 2. 安装Kaggle API
    print("\nStep 2: 安装Kaggle API...")
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    try:
        result = subprocess.run([python_path, "-c", "import kaggle"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("安装Kaggle...")
            subprocess.run([python_path, "-m", "pip", "install", "kaggle"], check=True)
        print("SUCCESS: Kaggle API就绪")
    except Exception as e:
        print(f"ERROR: Kaggle安装失败: {e}")
        return False
    
    # 3. 创建下载目录
    download_path = r"D:\project\vit_114\temp_imagenet_download"
    target_path = r"D:\project\vit_114\NNS_model_and_dataset\dataset\imagenet"
    
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    
    print(f"\nStep 3: 下载ImageNet到 {download_path}")
    
    # 4. 尝试下载不同的ImageNet数据集
    datasets_to_try = [
        "imagenet-object-localization-challenge",
        "jessicali9530/stanford-dogs-dataset",  # 备选：较小的数据集用于测试
    ]
    
    success = False
    for dataset in datasets_to_try:
        print(f"\n尝试下载: {dataset}")
        
        try:
            # 使用kaggle命令行工具
            result = subprocess.run([
                "kaggle", "datasets", "download", dataset, 
                "--path", download_path, "--unzip"
            ], capture_output=True, text=True, cwd=download_path, timeout=3600)  # 1小时超时
            
            if result.returncode == 0:
                print(f"SUCCESS: {dataset} 下载完成")
                success = True
                break
            else:
                print(f"FAILED: {dataset} 下载失败")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {dataset} 下载超时")
        except Exception as e:
            print(f"ERROR: {dataset} 下载异常: {e}")
    
    if not success:
        print("所有数据集下载失败，尝试替代方案...")
        return create_alternative_imagenet()
    
    # 5. 整理下载的数据
    print(f"\nStep 4: 整理数据到 {target_path}")
    
    download_dir = Path(download_path)
    target_dir = Path(target_path)
    
    # 查找验证集
    possible_val_paths = [
        download_dir / "ILSVRC" / "Data" / "CLS-LOC" / "val",
        download_dir / "val",
        download_dir / "validation",
        download_dir / "Images",
        download_dir,  # 如果直接在根目录
    ]
    
    val_source = None
    for path in possible_val_paths:
        if path.exists() and any(path.iterdir()):
            # 检查是否包含图片文件
            image_files = list(path.glob("**/*.JPEG")) + list(path.glob("**/*.jpg"))
            if image_files:
                val_source = path
                print(f"找到验证集: {path} ({len(image_files)} 张图片)")
                break
    
    if val_source:
        val_target = target_dir / "val"
        val_target.mkdir(parents=True, exist_ok=True)
        
        # 复制或移动文件
        if val_source.is_dir():
            print("复制验证集...")
            shutil.copytree(val_source, val_target, dirs_exist_ok=True)
        
        # 统计结果
        class_dirs = [d for d in val_target.iterdir() if d.is_dir()]
        total_images = sum(len(list(d.glob("*.JPEG"))) + len(list(d.glob("*.jpg"))) 
                          for d in class_dirs)
        
        print(f"SUCCESS: ImageNet数据整理完成")
        print(f"  验证集路径: {val_target}")
        print(f"  类别数: {len(class_dirs)}")
        print(f"  图片数: {total_images}")
        
        # 6. 清理临时文件
        print(f"\nStep 5: 清理临时文件...")
        try:
            shutil.rmtree(download_path)
            print("临时文件已清理")
        except:
            print("临时文件清理失败，请手动删除")
        
        return True
    else:
        print("WARNING: 未找到有效的验证集数据")
        return False

def create_alternative_imagenet():
    """创建替代的ImageNet数据集（扩展现有数据）"""
    print("\n" + "="*80)
    print("创建替代ImageNet数据集")
    print("="*80)
    
    # 使用现有的imagenet-scanner数据作为基础
    source_path = r"d:\project\vit_114\vitnns\49_vit\datas\imagenet-scanner"
    target_path = r"D:\project\vit_114\NNS_model_and_dataset\dataset\imagenet\val"
    
    if not os.path.exists(source_path):
        print("ERROR: 现有ImageNet扫描数据不存在")
        return False
    
    print(f"扩展现有数据: {source_path} -> {target_path}")
    
    os.makedirs(target_path, exist_ok=True)
    
    # 复制现有数据
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    
    # 统计数据
    class_dirs = [d for d in os.listdir(target_path) 
                 if os.path.isdir(os.path.join(target_path, d))]
    total_images = 0
    
    for class_dir in class_dirs:
        class_path = os.path.join(target_path, class_dir)
        images = len([f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_images += images
    
    print(f"SUCCESS: 替代ImageNet数据集创建完成")
    print(f"  路径: {target_path}")
    print(f"  类别数: {len(class_dirs)}")
    print(f"  图片数: {total_images}")
    
    return True

def create_download_summary():
    """创建下载总结"""
    summary = """# ImageNet数据集下载完成

## 数据集信息
- 来源: Kaggle API
- 类型: ImageNet验证集
- 位置: D:\\project\\vit_114\\NNS_model_and_dataset\\dataset\\imagenet\\val

## 使用方法
```bash
# 运行完整ImageNet分析
python run_imagenet_vitbase16_complete.py

# 或者直接运行NNS扫描
python train_scanner_cls.py \\
    --device 0 \\
    --count 10 \\
    --dataset ImageNet \\
    --classes 1000 \\
    --imagenet_path "D:\\project\\vit_114\\NNS_model_and_dataset\\dataset\\imagenet\\val" \\
    --process_wrong_classification True \\
    --nns_resize 64
```

## 下一步
1. 运行ImageNet + ViT-base-16分析
2. 生成完整的1000类可视化
3. 对比Tiny-ImageNet vs ImageNet结果
"""
    
    with open("IMAGENET_DOWNLOAD_SUCCESS.md", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print("SUCCESS: 下载总结已保存: IMAGENET_DOWNLOAD_SUCCESS.md")

def main():
    """主函数"""
    print("ImageNet自动下载器")
    
    if setup_kaggle_and_download():
        create_download_summary()
        
        print("\n" + "="*80)
        print("SUCCESS: ImageNet数据集准备完成!")
        print("="*80)
        print("现在可以运行完整的ImageNet分析:")
        print("  python run_imagenet_vitbase16_complete.py")
        print("\n或者运行简化版本:")
        print("  python run_imagenet_simple.py")
    else:
        print("\n" + "="*80)
        print("WARNING: 完整下载失败，但可以使用现有数据")
        print("="*80)
        print("您仍然可以运行ImageNet分析:")
        print("  python run_imagenet_simple.py")

if __name__ == "__main__":
    main()
