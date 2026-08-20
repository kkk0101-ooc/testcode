"""扩展现有ImageNet数据分析"""
import subprocess

def run_extended_analysis():
    """运行扩展分析"""
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
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
