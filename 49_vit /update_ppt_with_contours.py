"""
更新PPT图片为带轮廓的优化版本
"""

from pathlib import Path
import shutil

def update_ppt_images():
    """更新PPT图片文件夹"""
    
    print("="*60)
    print("更新PPT图片为优化版本（带轮廓）")
    print("="*60)
    print()
    
    # 源文件和目标文件
    updates = [
        {
            'source': 'ppt_results/imagenet_visualizations/combined/imagenet_00000441_combined_contours.png',
            'target': 'PPT_Images/1_狗_清晰案例_imagenet_00000441_combined.png',
            'name': '狗图片 (00000441)'
        },
        {
            'source': 'ppt_results/imagenet_visualizations/combined/imagenet_00006100_combined_contours.png',
            'target': 'PPT_Images/2_螃蟹_对比案例_imagenet_00006100_combined.png',
            'name': '螃蟹图片 (00006100)'
        }
    ]
    
    success_count = 0
    
    for update in updates:
        source = Path(update['source'])
        target = Path(update['target'])
        
        print(f"更新: {update['name']}")
        
        if source.exists():
            # 备份旧文件
            if target.exists():
                backup = target.with_suffix('.png.bak')
                shutil.copy2(target, backup)
                print(f"  备份旧文件: {backup.name}")
            
            # 复制新文件
            shutil.copy2(source, target)
            
            # 检查文件大小
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  OK 已更新 ({size_mb:.2f} MB)")
            success_count += 1
        else:
            print(f"  X 源文件不存在: {source}")
        
        print()
    
    print("="*60)
    print(f"完成! 成功更新 {success_count} 张图片")
    print("="*60)
    print()
    print("优化内容:")
    print("  ✓ 在V向量热力图上添加白色轮廓线")
    print("  ✓ 在CLS Attention上添加更粗的白色轮廓线")
    print("  ✓ 在高attention区域添加白色散点标记")
    print("  ✓ 突出显示目标的身体边界")
    print()
    print("可以直接使用PPT_Images/文件夹中的图片！")

if __name__ == "__main__":
    update_ppt_images()
