"""使用Kaggle API下载完整ImageNet数据集"""
import os
import subprocess
import shutil
import time
from pathlib import Path

def setup_kaggle_environment():
    """设置Kaggle环境"""
    print("="*80)
    print("设置Kaggle环境...")
    print("="*80)
    
    # 1. 复制kaggle.json到正确位置
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_config = kaggle_dir / 'kaggle.json'
    
    kaggle_dir.mkdir(exist_ok=True)
    
    source_config = Path('kaggle.json')
    if source_config.exists():
        shutil.copy2(source_config, kaggle_config)
        os.chmod(kaggle_config, 0o600)
        print(f"SUCCESS: Kaggle配置已设置到 {kaggle_config}")
    else:
        print("ERROR: kaggle.json文件不存在")
        return False
    
    # 2. 安装/更新Kaggle API
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    print("检查Kaggle API...")
    try:
        result = subprocess.run([python_path, "-c", "import kaggle; print('Kaggle version:', kaggle.__version__)"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("SUCCESS: Kaggle API已安装")
            print(result.stdout.strip())
        else:
            print("安装Kaggle API...")
            subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "kaggle"], 
                         check=True, timeout=300)
            print("SUCCESS: Kaggle API安装完成")
    except Exception as e:
        print(f"ERROR: Kaggle API设置失败: {e}")
        return False
    
    # 3. 测试Kaggle连接
    print("测试Kaggle API连接...")
    try:
        result = subprocess.run([python_path, "-c", "import kaggle; kaggle.api.authenticate(); print('Authentication successful')"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("SUCCESS: Kaggle API认证成功")
        else:
            print(f"WARNING: 认证测试失败: {result.stderr}")
    except Exception as e:
        print(f"WARNING: 认证测试异常: {e}")
    
    return True

def search_imagenet_datasets():
    """搜索ImageNet数据集"""
    print("\n" + "="*80)
    print("搜索ImageNet数据集...")
    print("="*80)
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 推荐的ImageNet数据集
    datasets = [
        "imagenet-object-localization-challenge",  # 官方ILSVRC数据集
        "jessicali9530/stanford-dogs-dataset",     # 备选：斯坦福狗数据集
        "pytorch/vision",                          # PyTorch官方数据集
    ]
    
    print("推荐的ImageNet数据集:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
        
        # 检查数据集信息
        try:
            result = subprocess.run([
                python_path, "-c", 
                f"import kaggle; info = kaggle.api.dataset_view('{dataset}'); print(f'Size: {{info.totalBytes // (1024**3)}}GB, Files: {{info.totalFiles}}')"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"   {result.stdout.strip()}")
            else:
                print(f"   信息获取失败: {result.stderr}")
        except Exception as e:
            print(f"   检查异常: {e}")
    
    return datasets

def download_imagenet_dataset(dataset_name="imagenet-object-localization-challenge"):
    """下载ImageNet数据集"""
    print(f"\n" + "="*80)
    print(f"下载ImageNet数据集: {dataset_name}")
    print("="*80)
    
    # 设置路径
    download_dir = Path(r"D:\project\vit_114\temp_imagenet_download")
    target_dir = Path(r"D:\project\vit_114\NNS_model_and_dataset\dataset\imagenet_full")
    
    # 创建目录
    download_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"下载目录: {download_dir}")
    print(f"目标目录: {target_dir}")
    
    # 检查磁盘空间
    import shutil as disk_usage
    free_space = disk_usage.disk_usage(download_dir.parent).free
    free_gb = free_space / (1024**3)
    
    print(f"可用磁盘空间: {free_gb:.1f} GB")
    
    if free_gb < 15:
        print("WARNING: 磁盘空间可能不足，ImageNet数据集约需要6-12GB")
        response = input("是否继续下载? (y/n): ").strip().lower()
        if response != 'y':
            print("下载已取消")
            return False
    
    # 开始下载
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    print("开始下载ImageNet数据集...")
    print("注意: 这可能需要30分钟到2小时，取决于网络速度")
    
    start_time = time.time()
    
    try:
        # 使用Python API下载
        download_script = f'''
import kaggle
import os
import time

print("开始下载 {dataset_name}...")
os.chdir(r"{download_dir}")

start = time.time()
kaggle.api.dataset_download_files("{dataset_name}", unzip=True)
end = time.time()

print(f"下载完成! 耗时: {{(end-start)/60:.1f}} 分钟")

# 列出下载的文件
import os
for root, dirs, files in os.walk("."):
    print(f"目录: {{root}}")
    for file in files[:5]:  # 只显示前5个文件
        print(f"  {{file}}")
    if len(files) > 5:
        print(f"  ... 还有 {{len(files)-5}} 个文件")
'''
        
        with open("temp_download_script.py", "w", encoding="utf-8") as f:
            f.write(download_script)
        
        # 执行下载
        result = subprocess.run([python_path, "temp_download_script.py"], 
                              cwd=str(download_dir.parent), 
                              timeout=7200)  # 2小时超时
        
        # 清理临时脚本
        if os.path.exists("temp_download_script.py"):
            os.remove("temp_download_script.py")
        
        end_time = time.time()
        download_time = (end_time - start_time) / 60
        
        if result.returncode == 0:
            print(f"SUCCESS: 数据集下载完成! (耗时: {download_time:.1f} 分钟)")
            return organize_downloaded_data(download_dir, target_dir)
        else:
            print("ERROR: 下载失败")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: 下载超时 (2小时)")
        return False
    except Exception as e:
        print(f"ERROR: 下载异常: {e}")
        return False

def organize_downloaded_data(download_dir, target_dir):
    """整理下载的数据"""
    print(f"\n整理数据: {download_dir} -> {target_dir}")
    
    # 查找验证集数据
    possible_val_paths = [
        download_dir / "ILSVRC" / "Data" / "CLS-LOC" / "val",
        download_dir / "val",
        download_dir / "validation", 
        download_dir / "imagenet" / "val",
        download_dir,  # 有时直接在根目录
    ]
    
    val_source = None
    for path in possible_val_paths:
        if path.exists():
            # 检查是否包含图片
            image_files = list(path.glob("**/*.JPEG")) + list(path.glob("**/*.jpg"))
            if len(image_files) > 1000:  # 至少1000张图片才认为是有效的
                val_source = path
                print(f"找到验证集: {path} ({len(image_files)} 张图片)")
                break
    
    if not val_source:
        print("WARNING: 未找到有效的验证集数据")
        # 列出下载目录内容
        print("下载目录内容:")
        for item in download_dir.iterdir():
            if item.is_dir():
                print(f"  目录: {item.name}")
            else:
                print(f"  文件: {item.name}")
        return False
    
    # 复制到目标目录
    val_target = target_dir / "val"
    
    print("复制验证集数据...")
    if val_source != val_target:
        shutil.copytree(val_source, val_target, dirs_exist_ok=True)
    
    # 统计结果
    class_dirs = [d for d in val_target.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob("*.JPEG"))) + len(list(d.glob("*.jpg"))) 
                      for d in class_dirs)
    
    print(f"SUCCESS: 数据整理完成")
    print(f"  验证集路径: {val_target}")
    print(f"  类别数: {len(class_dirs)}")
    print(f"  图片数: {total_images}")
    
    # 创建配置文件
    create_imagenet_config(val_target, len(class_dirs), total_images)
    
    return True

def create_imagenet_config(data_path, num_classes, num_images):
    """创建ImageNet配置文件"""
    config_content = f"""# 完整ImageNet数据集配置

## 数据集信息
- 路径: {data_path}
- 类别数: {num_classes}
- 图片数: {num_images}
- 下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 使用方法

### 运行完整ImageNet分析
```bash
python run_imagenet_vitbase16_complete.py
```

### 自定义分析
```bash
python train_scanner_cls.py \\
    --device 0 \\
    --count 50 \\
    --dataset ImageNet \\
    --classes 1000 \\
    --imagenet_path "{data_path}" \\
    --process_wrong_classification True \\
    --nns_resize 64
```

### 批量处理
```bash
# 处理100张图片
python train_scanner_cls.py --count 100 --imagenet_path "{data_path}"

# 处理特定类别
python train_scanner_cls.py --count 10 --imagenet_path "{data_path}" --target_classes n01440764,n01443537
```

## 性能建议
- 使用GPU: --device 0
- 内存优化: --nns_resize 64
- 批量处理: --count 适当设置
- 并行处理: 可以同时运行多个实例处理不同类别

## 生成的文件
- 网格图: scan_results/.../imagenet_full_*layers_grid.png
- 详细分析: analysis_results/imagenet_full_*
- 统计报告: analysis_results/IMAGENET_FULL_STATISTICS.png
"""
    
    with open("IMAGENET_FULL_CONFIG.md", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("SUCCESS: 配置文件已创建: IMAGENET_FULL_CONFIG.md")

def cleanup_temp_files(download_dir):
    """清理临时文件"""
    print(f"\n清理临时文件: {download_dir}")
    
    try:
        if download_dir.exists():
            shutil.rmtree(download_dir)
            print("SUCCESS: 临时文件已清理")
    except Exception as e:
        print(f"WARNING: 临时文件清理失败: {e}")

def main():
    """主函数"""
    print("="*80)
    print("完整ImageNet数据集下载器")
    print("="*80)
    print("这将下载完整的ImageNet验证集 (~6-12GB)")
    print("包含1000个类别，约50,000张图片")
    print("="*80)
    
    # Step 1: 设置环境
    if not setup_kaggle_environment():
        print("环境设置失败，退出")
        return
    
    # Step 2: 搜索数据集
    datasets = search_imagenet_datasets()
    
    # Step 3: 确认下载
    print(f"\n推荐下载: {datasets[0]}")
    print("这是官方的ImageNet Object Localization Challenge数据集")
    
    confirm = input("\n确认下载完整ImageNet数据集? (y/n): ").strip().lower()
    if confirm != 'y':
        print("下载已取消")
        return
    
    # Step 4: 下载数据集
    print("\n开始下载...")
    success = download_imagenet_dataset(datasets[0])
    
    if success:
        # Step 5: 清理临时文件
        download_dir = Path(r"D:\project\vit_114\temp_imagenet_download")
        cleanup_temp_files(download_dir)
        
        print("\n" + "="*80)
        print("SUCCESS: 完整ImageNet数据集下载完成!")
        print("="*80)
        print("现在您拥有:")
        print("- 完整ImageNet验证集 (1000类, ~50k图片)")
        print("- 标准目录结构")
        print("- 配置文件和使用说明")
        print("\n可以开始大规模ImageNet分析了!")
        print("运行: python run_imagenet_vitbase16_complete.py")
    else:
        print("\n" + "="*80)
        print("下载失败或不完整")
        print("="*80)
        print("您仍然可以使用现有的小规模ImageNet数据")

if __name__ == "__main__":
    main()
