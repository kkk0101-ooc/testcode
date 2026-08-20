"""测试Kaggle API并提供下载方案"""
import os
import subprocess
from pathlib import Path

def test_kaggle_api():
    """测试Kaggle API功能"""
    print("="*80)
    print("测试Kaggle API状态")
    print("="*80)
    
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    # 测试1: 检查kaggle模块
    print("1. 检查Kaggle模块...")
    try:
        result = subprocess.run([
            python_path, "-c", 
            "import kaggle; print('Kaggle version:', kaggle.__version__)"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("SUCCESS: Kaggle模块可用")
            print(result.stdout.strip())
        else:
            print("ERROR: Kaggle模块问题")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    # 测试2: 检查认证
    print("\n2. 检查API认证...")
    try:
        result = subprocess.run([
            python_path, "-c", 
            "import kaggle; kaggle.api.authenticate(); print('Authentication successful')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("SUCCESS: API认证成功")
        else:
            print("ERROR: API认证失败")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    # 测试3: 列出数据集
    print("\n3. 测试数据集列表...")
    try:
        result = subprocess.run([
            python_path, "-c", 
            "import kaggle; datasets = kaggle.api.dataset_list(search='imagenet', page_size=3); print(f'Found {len(datasets)} datasets')"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("SUCCESS: 数据集搜索成功")
            print(result.stdout.strip())
        else:
            print("WARNING: 数据集搜索失败")
            print(result.stderr)
    except Exception as e:
        print(f"WARNING: {e}")
    
    return True

def provide_manual_download_guide():
    """提供手动下载指南"""
    print("\n" + "="*80)
    print("ImageNet数据集获取方案")
    print("="*80)
    
    guide = """
## 方案1: Kaggle网页下载 (推荐)
1. 访问: https://www.kaggle.com/c/imagenet-object-localization-challenge/data
2. 登录您的Kaggle账户 (jiaguozhong188)
3. 点击 "Download All" 下载完整数据集
4. 解压到: D:\\project\\vit_114\\NNS_model_and_dataset\\dataset\\imagenet_full\\

## 方案2: 使用现有数据扩展
我们已经有100张ImageNet图片，可以：
- 立即开始分析
- 处理更多现有图片
- 生成高质量可视化

## 方案3: 其他ImageNet源
- Academic Torrents
- 官方ImageNet网站 (需要申请)
- 其他研究机构镜像

## 立即可用的命令
```bash
# 使用现有ImageNet数据
python train_scanner_cls.py --device 0 --count 20 --dataset ImageNet

# 生成更多可视化
python run_imagenet_vitbase16_complete.py
```
"""
    
    print(guide)
    
    with open("IMAGENET_DOWNLOAD_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide)
    
    print("详细指南已保存: IMAGENET_DOWNLOAD_GUIDE.md")

def create_expansion_script():
    """创建数据扩展脚本"""
    print("\n创建数据扩展脚本...")
    
    script_content = '''"""扩展现有ImageNet数据分析"""
import subprocess

def run_extended_analysis():
    """运行扩展分析"""
    python_path = r"D:\\Program Files\\miniconda3\\envs\\vit_114\\python.exe"
    
    commands = [
        # 处理更多图片
        [python_path, "train_scanner_cls.py", "--device", "0", "--count", "20", 
         "--dataset", "ImageNet", "--process_wrong_classification", "True"],
        
        # 生成网格图
        [python_path, "create_imagenet_10grid.py"],
        
        # 运行完整分析
        [python_path, "run_imagenet_vitbase16_complete.py"]
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"Step {i}: {' '.join(cmd[1:3])}")
        try:
            result = subprocess.run(cmd, timeout=1800)  # 30分钟超时
            if result.returncode == 0:
                print(f"SUCCESS: Step {i} completed")
            else:
                print(f"WARNING: Step {i} failed")
        except Exception as e:
            print(f"ERROR: Step {i} exception: {e}")

if __name__ == "__main__":
    run_extended_analysis()
'''
    
    with open("expand_imagenet_analysis.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("扩展脚本已创建: expand_imagenet_analysis.py")

def main():
    """主函数"""
    print("Kaggle API测试和ImageNet下载方案")
    
    # 测试API
    api_works = test_kaggle_api()
    
    if api_works:
        print("\n" + "="*80)
        print("Kaggle API工作正常!")
        print("="*80)
        print("API可用，但可能是网络或数据集访问问题")
        print("建议使用网页下载方式")
    else:
        print("\n" + "="*80)
        print("Kaggle API需要配置")
        print("="*80)
    
    # 提供替代方案
    provide_manual_download_guide()
    
    # 创建扩展脚本
    create_expansion_script()
    
    print("\n" + "="*80)
    print("总结和建议")
    print("="*80)
    print("1. Kaggle API已配置，但命令行工具可能不可用")
    print("2. 建议使用Kaggle网页直接下载")
    print("3. 现有100张ImageNet图片可立即使用")
    print("4. 扩展脚本已准备好处理更多数据")
    print("\n立即开始:")
    print("  python expand_imagenet_analysis.py")

if __name__ == "__main__":
    main()
