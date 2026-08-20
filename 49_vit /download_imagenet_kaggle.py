"""使用Kaggle API下载完整ImageNet数据集"""
import os
import subprocess
import shutil
import zipfile
from pathlib import Path

def setup_kaggle_credentials():
    """设置Kaggle凭据"""
    print("="*80)
    print("设置Kaggle凭据...")
    print("="*80)
    
    # Kaggle配置文件路径
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_config = kaggle_dir / 'kaggle.json'
    
    # 创建.kaggle目录
    kaggle_dir.mkdir(exist_ok=True)
    
    # 复制kaggle.json到正确位置
    source_config = Path('kaggle.json')
    if source_config.exists():
        shutil.copy2(source_config, kaggle_config)
        # 设置权限（仅用户可读）
        os.chmod(kaggle_config, 0o600)
        print(f"SUCCESS: Kaggle配置已复制到 {kaggle_config}")
    else:
        print("ERROR: kaggle.json文件不存在")
        return False
    
    return True

def install_kaggle():
    """安装Kaggle API"""
    print("\n检查并安装Kaggle API...")
    
    try:
        import kaggle
        print("SUCCESS: Kaggle API已安装")
        return True
    except ImportError:
        print("安装Kaggle API...")
        python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
        result = subprocess.run([python_path, "-m", "pip", "install", "kaggle"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("SUCCESS: Kaggle API安装完成")
            return True
        else:
            print(f"ERROR: Kaggle API安装失败: {result.stderr}")
            return False

def search_imagenet_datasets():
    """搜索可用的ImageNet数据集"""
    print("\n" + "="*80)
    print("搜索可用的ImageNet数据集...")
    print("="*80)
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 搜索ImageNet相关数据集
    search_terms = [
        "imagenet",
        "imagenet-object-localization-challenge",
        "imagenet-1k",
        "ilsvrc2012"
    ]
    
    available_datasets = []
    
    for term in search_terms:
        print(f"\n搜索: {term}")
        result = subprocess.run([
            python_path, "-c", 
            f"import kaggle; kaggle.api.dataset_list(search='{term}', page_size=5)"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"搜索结果: {result.stdout}")
        else:
            print(f"搜索失败: {result.stderr}")
    
    # 推荐的ImageNet数据集
    recommended_datasets = [
        "imagenet-object-localization-challenge",
        "jessicali9530/stanford-dogs-dataset",  # 较小的测试数据集
        "pytorch/imagenet-1k",  # 如果可用
    ]
    
    print(f"\n推荐的数据集:")
    for i, dataset in enumerate(recommended_datasets, 1):
        print(f"{i}. {dataset}")
    
    return recommended_datasets

def download_imagenet_dataset(dataset_name, download_path):
    """下载指定的ImageNet数据集"""
    print(f"\n" + "="*80)
    print(f"下载数据集: {dataset_name}")
    print(f"下载路径: {download_path}")
    print("="*80)
    
    # 创建下载目录
    os.makedirs(download_path, exist_ok=True)
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 下载数据集
    print("开始下载... (这可能需要很长时间)")
    result = subprocess.run([
        python_path, "-c",
        f"""
import kaggle
import os
os.chdir(r'{download_path}')
kaggle.api.dataset_download_files('{dataset_name}', unzip=True)
print('下载完成!')
"""
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    if result.returncode == 0:
        print("SUCCESS: 数据集下载完成!")
        print(result.stdout)
        return True
    else:
        print("ERROR: 数据集下载失败")
        print(result.stderr)
        return False

def organize_imagenet_data(download_path, target_path):
    """整理ImageNet数据到标准格式"""
    print(f"\n" + "="*80)
    print("整理ImageNet数据到标准格式...")
    print("="*80)
    
    download_dir = Path(download_path)
    target_dir = Path(target_path)
    
    # 创建目标目录结构
    val_dir = target_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找下载的文件
    print(f"查找下载文件在: {download_dir}")
    
    # 常见的ImageNet文件结构
    possible_paths = [
        download_dir / 'ILSVRC' / 'Data' / 'CLS-LOC' / 'val',
        download_dir / 'val',
        download_dir / 'validation',
        download_dir / 'imagenet' / 'val',
    ]
    
    source_val_dir = None
    for path in possible_paths:
        if path.exists():
            source_val_dir = path
            print(f"找到验证集: {path}")
            break
    
    if source_val_dir:
        # 复制验证集数据
        print("复制验证集数据...")
        if source_val_dir != val_dir:
            shutil.copytree(source_val_dir, val_dir, dirs_exist_ok=True)
        
        # 统计数据
        class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
        total_images = sum(len(list(d.glob('*.JPEG'))) + len(list(d.glob('*.jpg'))) 
                          for d in class_dirs)
        
        print(f"SUCCESS: 数据整理完成")
        print(f"  类别数: {len(class_dirs)}")
        print(f"  图片数: {total_images}")
        
        return True
    else:
        print("WARNING: 未找到标准的验证集目录")
        print("请手动检查下载的文件结构")
        return False

def create_imagenet_config():
    """创建ImageNet配置文件"""
    print("\n创建ImageNet配置...")
    
    config_content = """# ImageNet数据集配置

## 数据集路径
IMAGENET_PATH = r"D:\\project\\vit_114\\NNS_model_and_dataset\\dataset\\imagenet\\val"

## 数据集统计
- 类别数: 1000
- 验证集图片数: ~50,000
- 每类图片数: ~50

## 使用方法
```python
# 在train_scanner_cls.py中使用
python train_scanner_cls.py \\
    --device 0 \\
    --count 10 \\
    --dataset ImageNet \\
    --classes 1000 \\
    --imagenet_path "D:\\project\\vit_114\\NNS_model_and_dataset\\dataset\\imagenet\\val" \\
    --process_wrong_classification True \\
    --nns_resize 64
```

## 注意事项
1. 确保有足够的磁盘空间 (>10GB)
2. 首次处理可能较慢
3. 建议使用GPU加速
4. 可以通过--count参数控制处理的图片数量
"""
    
    with open("IMAGENET_CONFIG.md", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("SUCCESS: 配置文件已创建: IMAGENET_CONFIG.md")

def main():
    """主函数"""
    print("="*80)
    print("ImageNet数据集下载器 (Kaggle API)")
    print("="*80)
    
    # Step 1: 设置Kaggle凭据
    if not setup_kaggle_credentials():
        return
    
    # Step 2: 安装Kaggle API
    if not install_kaggle():
        return
    
    # Step 3: 搜索可用数据集
    datasets = search_imagenet_datasets()
    
    # Step 4: 选择数据集
    print("\n" + "="*80)
    print("选择要下载的数据集:")
    print("="*80)
    print("1. imagenet-object-localization-challenge (官方, ~6.3GB)")
    print("2. jessicali9530/stanford-dogs-dataset (较小, 测试用)")
    print("3. 自定义数据集名称")
    print("4. 跳过下载，使用现有数据")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == "1":
        dataset_name = "imagenet-object-localization-challenge"
    elif choice == "2":
        dataset_name = "jessicali9530/stanford-dogs-dataset"
    elif choice == "3":
        dataset_name = input("请输入数据集名称: ").strip()
    elif choice == "4":
        print("跳过下载，使用现有数据")
        create_imagenet_config()
        return
    else:
        print("无效选择，使用默认数据集")
        dataset_name = "imagenet-object-localization-challenge"
    
    # Step 5: 设置下载路径
    download_path = r"D:\project\vit_114\temp_download"
    target_path = r"D:\project\vit_114\NNS_model_and_dataset\dataset\imagenet"
    
    print(f"\n下载设置:")
    print(f"  数据集: {dataset_name}")
    print(f"  临时下载路径: {download_path}")
    print(f"  最终路径: {target_path}")
    
    confirm = input("\n确认下载? (y/n): ").strip().lower()
    if confirm != 'y':
        print("下载已取消")
        return
    
    # Step 6: 下载数据集
    if download_imagenet_dataset(dataset_name, download_path):
        # Step 7: 整理数据
        organize_imagenet_data(download_path, target_path)
        
        # Step 8: 创建配置
        create_imagenet_config()
        
        # Step 9: 清理临时文件
        cleanup = input("\n删除临时下载文件? (y/n): ").strip().lower()
        if cleanup == 'y':
            shutil.rmtree(download_path, ignore_errors=True)
            print("临时文件已清理")
        
        print("\n" + "="*80)
        print("SUCCESS: ImageNet数据集下载和配置完成!")
        print("="*80)
        print(f"数据集位置: {target_path}")
        print("现在可以运行完整的ImageNet + ViT-base-16分析了!")
        print("\n下一步:")
        print("python run_imagenet_vitbase16_complete.py")
    else:
        print("下载失败，请检查网络连接和Kaggle配置")

if __name__ == "__main__":
    main()
