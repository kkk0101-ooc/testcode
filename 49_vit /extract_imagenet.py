"""
解压ImageNet验证集tar文件
"""

import tarfile
from pathlib import Path
from tqdm import tqdm
import time

def extract_imagenet():
    print("="*80)
    print("解压ImageNet验证集")
    print("="*80)
    
    tar_path = Path("datas/ILSVRC2012_img_val.tar")
    extract_dir = Path("datas/ILSVRC2012_val")
    
    if not tar_path.exists():
        print(f"错误: tar文件不存在 - {tar_path}")
        return False
    
    # 创建解压目录
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n目标目录: {extract_dir}")
    
    # 检查文件大小
    size_gb = tar_path.stat().st_size / (1024**3)
    print(f"Tar文件大小: {size_gb:.2f} GB")
    
    # 开始解压
    print(f"\n开始解压...")
    print(f"⚠️ 这可能需要10-30分钟，请耐心等待...\n")
    
    start_time = time.time()
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            print(f"总文件数: {len(members)}")
            
            # 使用tqdm显示进度
            for member in tqdm(members, desc="解压中", unit="文件"):
                tar.extract(member, extract_dir)
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n✅ 解压完成!")
        print(f"   耗时: {minutes}分{seconds}秒")
        print(f"   位置: {extract_dir}")
        
        # 验证解压结果
        jpeg_files = list(extract_dir.glob("*.JPEG"))
        jpg_files = list(extract_dir.glob("*.jpg"))
        total_files = len(jpeg_files) + len(jpg_files)
        
        print(f"   图片数量: {total_files}")
        
        if total_files == 50000:
            print(f"   ✅ 完整! 所有50000张图片都已解压")
        elif total_files > 0:
            print(f"   ⚠️ 只解压了 {total_files}/50000 张")
        else:
            print(f"   ❌ 未找到图片文件")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 解压失败: {e}")
        return False

def main():
    # 确认操作
    print("\n⚠️ 注意:")
    print("1. 这将解压约50000张ImageNet图片到 datas/ILSVRC2012_val/")
    print("2. 需要约7GB磁盘空间")
    print("3. 解压时间约10-30分钟")
    print("4. 不会删除原tar文件")
    
    response = input("\n是否继续? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        success = extract_imagenet()
        
        if success:
            print("\n" + "="*80)
            print("下一步: 运行NNS扫描")
            print("="*80)
            print("\n现在可以对完整的50000张ImageNet图片运行NNS扫描:")
            print("python run_imagenet_vitbase16_complete.py")
    else:
        print("\n操作已取消")

if __name__ == "__main__":
    main()
