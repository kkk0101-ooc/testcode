"""
智能解压和处理ImageNet - 控制磁盘空间
"""

import tarfile
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import os
import sys

def get_disk_space(path="."):
    """获取磁盘可用空间（GB）"""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    used_gb = stat.used / (1024**3)
    return free_gb, total_gb, used_gb

def check_disk_space(min_free_gb=20):
    """检查磁盘空间是否足够"""
    free_gb, total_gb, used_gb = get_disk_space()
    print(f"\n磁盘空间:")
    print(f"  总容量: {total_gb:.2f} GB")
    print(f"  已使用: {used_gb:.2f} GB ({used_gb/total_gb*100:.1f}%)")
    print(f"  可用: {free_gb:.2f} GB ({free_gb/total_gb*100:.1f}%)")
    
    if free_gb < min_free_gb:
        print(f"\n警告: 可用空间不足 {min_free_gb} GB!")
        return False
    return True

def extract_partial_imagenet(num_images=500):
    """部分解压ImageNet，避免占用太多空间"""
    print("="*80)
    print("智能解压ImageNet（部分）")
    print("="*80)
    
    tar_path = Path("datas/ILSVRC2012_img_val.tar")
    extract_dir = Path("datas/ILSVRC2012_val")
    
    if not tar_path.exists():
        print(f"错误: tar文件不存在 - {tar_path}")
        return False, 0
    
    # 检查磁盘空间
    if not check_disk_space(min_free_gb=30):
        print("\n磁盘空间不足，建议清理后再运行")
        return False, 0
    
    # 创建解压目录
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查已解压的数量
    existing_files = list(extract_dir.glob("*.JPEG")) + list(extract_dir.glob("*.jpg"))
    if len(existing_files) >= num_images:
        print(f"\n已有 {len(existing_files)} 张图片，无需重复解压")
        return True, len(existing_files)
    
    print(f"\n目标: 解压 {num_images} 张图片")
    print(f"已有: {len(existing_files)} 张")
    print(f"需要: {num_images - len(existing_files)} 张\n")
    
    start_time = time.time()
    extracted_count = 0
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            print(f"Tar文件包含: {len(members)} 张图片")
            
            # 只解压需要的数量
            to_extract = min(num_images, len(members))
            
            for member in tqdm(members[:to_extract], desc="解压中", unit="张"):
                # 检查文件是否已存在
                target_path = extract_dir / member.name
                if not target_path.exists():
                    tar.extract(member, extract_dir)
                    extracted_count += 1
                
                # 每100张检查一次磁盘空间
                if extracted_count % 100 == 0:
                    free_gb, _, _ = get_disk_space()
                    if free_gb < 15:
                        print(f"\n\n警告: 磁盘空间不足15GB，停止解压!")
                        break
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n解压完成!")
        print(f"  新解压: {extracted_count} 张")
        print(f"  总计: {len(existing_files) + extracted_count} 张")
        print(f"  耗时: {minutes}分{seconds}秒")
        
        return True, len(existing_files) + extracted_count
        
    except Exception as e:
        print(f"\n错误: {e}")
        return False, len(existing_files)

def main():
    print("\n" + "="*80)
    print("智能ImageNet处理方案")
    print("="*80)
    
    # 初始磁盘空间检查
    print("\n1. 检查磁盘空间...")
    free_gb, total_gb, used_gb = get_disk_space()
    
    if free_gb < 20:
        print(f"\n磁盘空间不足! 只剩 {free_gb:.2f} GB")
        print("建议:")
        print("  1. 清理临时文件")
        print("  2. 删除不需要的大文件")
        print("  3. 只处理少量图片（如100张）")
        return
    
    # 决定处理数量
    if free_gb > 100:
        suggested_num = 1000
    elif free_gb > 50:
        suggested_num = 500
    else:
        suggested_num = 200
    
    print(f"\n根据可用空间 ({free_gb:.2f} GB)，建议处理 {suggested_num} 张图片")
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        try:
            num_images = int(sys.argv[1])
            print(f"\n使用命令行参数: {num_images} 张图片")
        except:
            print("\n命令行参数无效，使用建议值")
            num_images = suggested_num
    else:
        # 非交互模式，直接使用建议值
        num_images = suggested_num
        print(f"\n自动使用建议值: {num_images} 张图片")
    
    # 安全检查
    if num_images > 5000:
        print(f"\n警告: {num_images} 张图片可能占用大量空间")
        print("限制为最多5000张")
        num_images = 5000
    
    print(f"\n开始处理 {num_images} 张图片...")
    
    # 解压
    print("\n" + "="*80)
    print("步骤1: 解压图片")
    print("="*80)
    
    success, actual_count = extract_partial_imagenet(num_images)
    
    if not success:
        print("\n解压失败，停止处理")
        return
    
    # 最终磁盘空间检查
    print("\n" + "="*80)
    print("最终状态")
    print("="*80)
    
    check_disk_space(min_free_gb=10)
    
    print(f"\n已解压图片: {actual_count} 张")
    print(f"位置: datas/ILSVRC2012_val/")
    
    print("\n下一步:")
    print("现在可以对这些图片运行NNS扫描或生成更多可视化")
    print("注意: 不要一次性处理所有图片，分批处理以控制磁盘空间")

if __name__ == "__main__":
    main()
