"""简化的ImageNet + ViT-base-16运行脚本"""
import os
import subprocess

def main():
    """运行ImageNet + ViT-base-16分析"""
    print("="*80)
    print("Running ImageNet + ViT-base-16 Analysis")
    print("="*80)
    
    # 检查ImageNet数据
    imagenet_path = r"d:\project\vit_114\vitnns\49_vit\datas\imagenet-scanner"
    
    if not os.path.exists(imagenet_path):
        print(f"ERROR: ImageNet data not found: {imagenet_path}")
        return
    
    print(f"Using ImageNet data: {imagenet_path}")
    
    # 运行NNS扫描
    python_path = r"D:\Program Files\miniconda3\envs\vit_114\python.exe"
    
    print("Step 1: Running NNS scan with ViT-base-16...")
    
    cmd = [
        python_path, "train_scanner_cls.py",
        "--device", "0",
        "--count", "3",
        "--dataset", "ImageNet", 
        "--classes", "1000",
        "--imagenet_path", imagenet_path,
        "--process_wrong_classification", "True",
        "--nns_resize", "64"
    ]
    
    print("Command:", " ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    if result.returncode == 0:
        print("SUCCESS: NNS scan completed")
        print(result.stdout)
        
        # 创建网格图
        print("\nStep 2: Creating grid visualization...")
        grid_result = subprocess.run([
            python_path, "create_imagenet_10grid.py"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if grid_result.returncode == 0:
            print("SUCCESS: Grid visualization created")
            print(grid_result.stdout)
        else:
            print("WARNING: Grid creation failed")
            print(grid_result.stderr)
        
        print("\n" + "="*80)
        print("SUCCESS: ImageNet analysis completed!")
        print("="*80)
        print("Check results in:")
        print("- scan_results/scan_vit/114VisionTransformer_ImageNet/")
        print("- Look for *_10layers_grid.png files")
        
    else:
        print("ERROR: NNS scan failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

if __name__ == "__main__":
    main()
