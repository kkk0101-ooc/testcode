"""
检查ImageNet验证集状态
"""

from pathlib import Path
import tarfile
import os

def check_imagenet_status():
    print("="*80)
    print("ImageNet验证集状态检查")
    print("="*80)
    
    # 检查tar文件
    tar_path = Path("datas/ILSVRC2012_img_val.tar")
    if tar_path.exists():
        size_gb = tar_path.stat().st_size / (1024**3)
        print(f"\nOK Tar文件存在:")
        print(f"   路径: {tar_path}")
        print(f"   大小: {size_gb:.2f} GB")
        
        # 检查tar文件内容
        print(f"\n正在检查tar文件内容...")
        try:
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                print(f"   包含文件数: {len(members)}")
                
                # 显示前10个文件
                print(f"\n   前10个文件:")
                for i, member in enumerate(members[:10]):
                    print(f"   {i+1}. {member.name}")
                
                if len(members) > 10:
                    print(f"   ... 还有 {len(members)-10} 个文件")
        except Exception as e:
            print(f"   错误: {e}")
    else:
        print(f"\nX Tar文件不存在: {tar_path}")
    
    # 检查可能的解压目录
    print(f"\n" + "="*80)
    print("检查可能的解压目录")
    print("="*80)
    
    possible_dirs = [
        Path("datas/ILSVRC2012_val"),
        Path("datas/val"),
        Path("datas/imagenet_val"),
        Path("datas/imagenet-val"),
        Path("ILSVRC2012_val"),
        Path("imagenet_val"),
    ]
    
    found_any = False
    for dir_path in possible_dirs:
        if dir_path.exists():
            found_any = True
            count = len(list(dir_path.glob("*.JPEG"))) + len(list(dir_path.glob("*.jpg")))
            print(f"OK 找到: {dir_path}")
            print(f"   图片数量: {count}")
        else:
            print(f"X 不存在: {dir_path}")
    
    if not found_any:
        print(f"\n! 未找到解压后的验证集")
        print(f"   Tar文件需要解压!")
    
    # 检查scan_results
    print(f"\n" + "="*80)
    print("检查已处理的图片")
    print("="*80)
    
    scan_dir = Path("scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained")
    if scan_dir.exists():
        all_dirs = [d for d in scan_dir.iterdir() if d.is_dir() and 'ILSVRC' in d.name]
        print(f"OK 已处理的ImageNet图片: {len(all_dirs)}")
        
        if len(all_dirs) < 50000:
            print(f"   ! 只处理了 {len(all_dirs)}/50000 张图片")
            print(f"   可能原因:")
            print(f"   1. 只处理了部分图片")
            print(f"   2. 处理过程被中断")
            print(f"   3. 原始验证集没有完全解压")
    else:
        print(f"X 未找到scan_results目录")
    
    # 总结
    print(f"\n" + "="*80)
    print("总结")
    print("="*80)
    
    if tar_path.exists() and not found_any:
        print(f"\n>> 状态: Tar文件存在但未解压")
        print(f"\n建议:")
        print(f"1. 解压tar文件到 datas/ILSVRC2012_val/")
        print(f"2. 命令: tar -xf datas/ILSVRC2012_img_val.tar -C datas/")
        print(f"3. 解压后应该有约50000张JPEG图片")
    elif found_any:
        print(f"\n>> 状态: 验证集已解压")
        print(f"   但只有 {len(all_dirs) if scan_dir.exists() else 0} 张被处理")
    else:
        print(f"\n>> 状态: 需要下载ImageNet验证集")

if __name__ == "__main__":
    check_imagenet_status()
