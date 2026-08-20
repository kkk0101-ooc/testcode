"""
对两张指定的ImageNet图片进行NNS扫描
"""

import torch
from pathlib import Path
import sys

# 假设NNS扫描的主脚本路径
sys.path.insert(0, str(Path(__file__).parent))

def scan_specific_images():
    """扫描两张指定图片"""
    
    print("="*80)
    print("NNS-ViT 扫描指定图片")
    print("="*80)
    print()
    
    # 图片路径
    images = [
        "datas/ILSVRC2012_val/ILSVRC2012_val_00000441.JPEG",
        "datas/ILSVRC2012_val/ILSVRC2012_val_00006100.JPEG"
    ]
    
    # 验证图片存在
    print("验证图片...")
    for img_path in images:
        p = Path(img_path)
        if p.exists():
            print(f"  OK {p.name}")
        else:
            print(f"  X 不存在: {img_path}")
            return
    
    print("\n准备扫描...")
    print("注意: 这将运行完整的ViT-Base模型推理")
    print("预计每张图片耗时: 30-60秒")
    print()
    
    # 检查是否有现成的扫描脚本
    scan_script = Path("114VisionTransformer_ImageNet.py")
    if not scan_script.exists():
        print(f"错误: 找不到扫描脚本 {scan_script}")
        print("需要手动运行NNS扫描")
        print()
        print("建议步骤:")
        print("1. 找到NNS扫描的主脚本")
        print("2. 配置扫描参数指向这两张图片")
        print("3. 运行扫描")
        print()
        print("或者，如果已有类似的扫描结果，可以复制一个修改")
        return
    
    print(f"使用扫描脚本: {scan_script}")
    print("建议直接运行该脚本并指定这两张图片")

if __name__ == "__main__":
    scan_specific_images()
