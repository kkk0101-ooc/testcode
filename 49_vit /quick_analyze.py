"""
快速分析脚本 - 可以在这里直接配置参数
"""

from generate_single_image_analysis import generate_analysis
from pathlib import Path

# ============================================================
# 配置区域 - 在这里修改你的参数
# ============================================================

# 方式1: 分析单张图片
IMAGE_PATH = "datas/ILSVRC2012_val/ILSVRC2012_val_00028758.JPEG"
OUTPUT_DIR = "do_imagenet_vit_28758"
IMAGE_NAME = "ILSVRC2012_val_00028758"  # 可选，用于显示

# 方式2: 分析多张图片（批量）
BATCH_MODE = True  # 改为True启用批量模式
BATCH_IMAGES = [
    ("datas/ILSVRC2012_val/ILSVRC2012_val_00000441.JPEG", "do_imagenet_vit_dog"),
    ("datas/ILSVRC2012_val/ILSVRC2012_val_00006100.JPEG", "do_imagenet_vit_crab"),
    # 添加更多图片...
]

# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    if not BATCH_MODE:
        # 单张图片模式
        print(f"分析单张图片: {IMAGE_PATH}")
        success = generate_analysis(IMAGE_PATH, OUTPUT_DIR, IMAGE_NAME)
        
        if success:
            print("\n[OK] 分析完成！")
            print(f"\n打开报告：")
            print(f"  start {OUTPUT_DIR}\\visualize_attention\\analysis_report.html")
        else:
            print("\n[FAIL] 分析失败！")
    
    else:
        # 批量模式
        print(f"批量分析 {len(BATCH_IMAGES)} 张图片")
        
        success_count = 0
        for img_path, output_dir in BATCH_IMAGES:
            print(f"\n处理: {img_path}")
            
            img_name = Path(img_path).stem
            success = generate_analysis(img_path, output_dir, img_name)
            
            if success:
                success_count += 1
                print(f"  [OK] 完成")
            else:
                print(f"  [FAIL] 失败")
        
        print(f"\n完成 {success_count}/{len(BATCH_IMAGES)} 张图片")
