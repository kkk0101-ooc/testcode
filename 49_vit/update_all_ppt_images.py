"""
更新PPT_Images文件夹，使用新生成的图片
"""

from pathlib import Path
import shutil

def update_ppt_images():
    """更新PPT图片"""
    
    print("="*60)
    print("更新PPT图片为新图片（自然colormap效果）")
    print("="*60)
    print()
    
    # 更新映射
    updates = [
        {
            'source': 'ppt_results/imagenet_visualizations/combined/imagenet_00000441_combined.png',
            'target': 'PPT_Images/1_狗_清晰案例_imagenet_00000441_combined.png',
            'name': '1. 狗 - 完整可视化'
        },
        {
            'source': 'ppt_results/imagenet_visualizations/combined/imagenet_00006100_combined.png',
            'target': 'PPT_Images/2_螃蟹_对比案例_imagenet_00006100_combined.png',
            'name': '2. 螃蟹 - 完整可视化'
        },
        {
            'source': 'ppt_results/imagenet_visualizations/v_vectors/v_vectors_00000441.png',
            'target': 'PPT_Images/4_V向量热力图_v_vectors_00000441.png',
            'name': '4. 狗 - V向量网格'
        },
        {
            'source': 'ppt_results/imagenet_visualizations/cls_layers/cls_layers_00000441.png',
            'target': 'PPT_Images/5_CLS演化_cls_layers_00000441.png',
            'name': '5. 狗 - CLS演化'
        }
    ]
    
    success_count = 0
    
    for update in updates:
        source = Path(update['source'])
        target = Path(update['target'])
        
        print(f"{update['name']}")
        
        if source.exists():
            # 备份旧文件
            if target.exists():
                backup = target.with_suffix('.png.old')
                shutil.copy2(target, backup)
                print(f"  备份: {backup.name}")
            
            # 复制新文件
            shutil.copy2(source, target)
            
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  OK ({size_mb:.2f} MB)")
            success_count += 1
        else:
            print(f"  错误: 源文件不存在")
        
        print()
    
    print("="*60)
    print(f"完成! 成功更新 {success_count}/{len(updates)} 张图片")
    print("="*60)
    print()
    print("优化说明:")
    print("  - 使用自然的colormap效果")
    print("  - hot: 高值自然显示为白色/黄色（V向量）")
    print("  - viridis: 高值自然显示为黄色/亮色（Attention）")
    print("  - 无人工添加的轮廓和标记")
    print()
    print("文件位置: PPT_Images/")

if __name__ == "__main__":
    update_ppt_images()
