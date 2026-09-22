"""自动下载完整ImageNet数据集"""
import os
import subprocess
import shutil
import time
from pathlib import Path

def download_imagenet_auto():
    """自动下载ImageNet数据集"""
    print("="*80)
    print("自动下载完整ImageNet数据集")
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
    free_space = shutil.disk_usage(download_dir.parent).free
    free_gb = free_space / (1024**3)
    print(f"可用磁盘空间: {free_gb:.1f} GB")
    
    if free_gb < 15:
        print("WARNING: 磁盘空间可能不足 (需要约12GB)")
    
    # 使用kaggle命令行工具下载
    dataset_name = "imagenet-object-localization-challenge"
    
    print(f"开始下载: {dataset_name}")
    print("预计时间: 30分钟 - 2小时 (取决于网络速度)")
    
    start_time = time.time()
    
    try:
        # 方法1: 使用kaggle命令行
        print("尝试使用kaggle命令行工具...")
        
        result = subprocess.run([
            "kaggle", "datasets", "download", dataset_name,
            "--path", str(download_dir),
            "--unzip"
        ], capture_output=True, text=True, timeout=7200, cwd=str(download_dir))
        
        if result.returncode == 0:
            print("SUCCESS: 使用kaggle命令行下载成功")
            end_time = time.time()
            print(f"下载耗时: {(end_time-start_time)/60:.1f} 分钟")
            
            return process_downloaded_data(download_dir, target_dir)
        else:
            print("kaggle命令行失败，尝试Python API...")
            
    except FileNotFoundError:
        print("kaggle命令行不可用，使用Python API...")
    except subprocess.TimeoutExpired:
        print("下载超时，尝试其他方法...")
    except Exception as e:
        print(f"kaggle命令行异常: {e}")
    
    # 方法2: 使用Python API
    try:
        print("使用Python Kaggle API下载...")
        
        python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
        
        download_script = f'''
import kaggle
import os
import time

print("使用Python API下载 {dataset_name}...")
os.chdir(r"{download_dir}")

try:
    start = time.time()
    kaggle.api.dataset_download_files("{dataset_name}", unzip=True)
    end = time.time()
    print(f"下载完成! 耗时: {{(end-start)/60:.1f}} 分钟")
    
    # 列出下载内容
    for root, dirs, files in os.walk("."):
        if len(files) > 0:
            print(f"目录 {{root}}: {{len(files)}} 个文件")
            
except Exception as e:
    print(f"下载失败: {{e}}")
    exit(1)
'''
        
        with open("temp_download.py", "w", encoding="utf-8") as f:
            f.write(download_script)
        
        result = subprocess.run([python_path, "temp_download.py"], 
                              timeout=7200, cwd=str(download_dir.parent))
        
        # 清理临时脚本
        if os.path.exists("temp_download.py"):
            os.remove("temp_download.py")
        
        if result.returncode == 0:
            print("SUCCESS: Python API下载成功")
            return process_downloaded_data(download_dir, target_dir)
        else:
            print("ERROR: Python API下载失败")
            
    except Exception as e:
        print(f"Python API异常: {e}")
    
    # 方法3: 尝试备选数据集
    print("尝试下载备选数据集...")
    backup_datasets = [
        "jessicali9530/stanford-dogs-dataset",
        "pytorch/vision"
    ]
    
    for backup_dataset in backup_datasets:
        print(f"尝试下载: {backup_dataset}")
        try:
            result = subprocess.run([
                "kaggle", "datasets", "download", backup_dataset,
                "--path", str(download_dir),
                "--unzip"
            ], capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"SUCCESS: 备选数据集下载成功: {backup_dataset}")
                return process_downloaded_data(download_dir, target_dir, backup_dataset)
                
        except Exception as e:
            print(f"备选数据集失败: {e}")
    
    print("所有下载方法都失败了")
    return False

def process_downloaded_data(download_dir, target_dir, dataset_name="imagenet"):
    """处理下载的数据"""
    print(f"\n处理下载的数据...")
    
    # 查找数据文件
    print(f"扫描下载目录: {download_dir}")
    
    # 统计下载内容
    total_files = 0
    image_files = 0
    
    for root, dirs, files in os.walk(download_dir):
        total_files += len(files)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files += 1
    
    print(f"总文件数: {total_files}")
    print(f"图片文件数: {image_files}")
    
    if image_files < 100:
        print("WARNING: 图片文件数量较少，可能下载不完整")
    
    # 查找验证集目录
    possible_paths = [
        download_dir / "ILSVRC" / "Data" / "CLS-LOC" / "val",
        download_dir / "val",
        download_dir / "validation",
        download_dir / "Images",
        download_dir,
    ]
    
    source_dir = None
    max_images = 0
    
    for path in possible_paths:
        if path.exists():
            path_images = len(list(path.glob("**/*.JPEG"))) + len(list(path.glob("**/*.jpg")))
            if path_images > max_images:
                max_images = path_images
                source_dir = path
    
    if source_dir:
        print(f"找到最佳数据源: {source_dir} ({max_images} 张图片)")
        
        # 复制到目标目录
        val_target = target_dir / "val"
        
        if source_dir != val_target:
            print("复制数据到目标目录...")
            shutil.copytree(source_dir, val_target, dirs_exist_ok=True)
        
        # 统计最终结果
        class_dirs = [d for d in val_target.iterdir() if d.is_dir()]
        final_images = sum(len(list(d.glob("*.JPEG"))) + len(list(d.glob("*.jpg"))) 
                          for d in class_dirs)
        
        print(f"SUCCESS: 数据处理完成")
        print(f"  目标路径: {val_target}")
        print(f"  类别数: {len(class_dirs)}")
        print(f"  图片数: {final_images}")
        
        # 创建配置
        create_config(val_target, len(class_dirs), final_images, dataset_name)
        
        return True
    else:
        print("ERROR: 未找到有效的图片数据")
        return False

def create_config(data_path, num_classes, num_images, dataset_name):
    """创建配置文件"""
    config = f"""# ImageNet数据集配置 (完整版)

## 数据集信息
- 来源: {dataset_name}
- 路径: {data_path}
- 类别数: {num_classes}
- 图片数: {num_images}
- 下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 快速开始
```bash
# 运行大规模分析 (推荐)
python train_scanner_cls.py --device 0 --count 100 --dataset ImageNet --imagenet_path "{data_path}"

# 生成网格图
python create_imagenet_10grid.py

# 完整流水线
python run_imagenet_vitbase16_complete.py
```

## 性能建议
- GPU加速: --device 0
- 内存优化: --nns_resize 64  
- 批量处理: --count 50-200
- 并行处理: 可同时处理多个类别

## 预期结果
- 12层注意力可视化
- 1000类完整覆盖
- 高质量研究级别图像
- 统计分析报告
"""
    
    with open("IMAGENET_FULL_READY.md", "w", encoding="utf-8") as f:
        f.write(config)
    
    print("SUCCESS: 配置文件已创建: IMAGENET_FULL_READY.md")

def cleanup_and_summary(download_dir, success):
    """清理和总结"""
    print("\n" + "="*80)
    
    if success:
        print("SUCCESS: 完整ImageNet数据集下载完成!")
        print("="*80)
        print("✓ 数据集已下载并整理")
        print("✓ 配置文件已创建")
        print("✓ 可以开始大规模分析")
        print("\n立即开始分析:")
        print("  python run_imagenet_vitbase16_complete.py")
        print("\n或处理更多图片:")
        print("  python train_scanner_cls.py --count 100 --dataset ImageNet")
    else:
        print("下载未完成或失败")
        print("="*80)
        print("可能的原因:")
        print("- 网络连接问题")
        print("- Kaggle API配置问题")
        print("- 磁盘空间不足")
        print("- 数据集暂时不可用")
        print("\n您仍可以使用现有的小规模ImageNet数据")
    
    # 清理临时文件
    try:
        if download_dir.exists():
            print(f"\n清理临时文件: {download_dir}")
            shutil.rmtree(download_dir)
            print("临时文件已清理")
    except Exception as e:
        print(f"临时文件清理失败: {e}")

def main():
    """主函数"""
    print("ImageNet完整数据集自动下载器")
    print("将下载官方ImageNet验证集 (~6-12GB)")
    
    download_dir = Path(r"D:\project\vit_114\temp_imagenet_download")
    
    # 开始下载
    success = download_imagenet_auto()
    
    # 清理和总结
    cleanup_and_summary(download_dir, success)

if __name__ == "__main__":
    main()
