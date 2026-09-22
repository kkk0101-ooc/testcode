"""
PPT图片准备工具
复制所有需要的图片到PPT_Images文件夹
"""

from pathlib import Path
import shutil

def prepare_ppt_images():
    print("="*60)
    print("PPT图片准备工具")
    print("="*60)
    print()
    
    # 创建目标文件夹
    target_folder = Path("PPT_Images")
    if target_folder.exists():
        print("清理旧文件夹...")
        shutil.rmtree(target_folder)
    
    target_folder.mkdir(exist_ok=True)
    print(f"OK 创建文件夹: {target_folder}")
    print()
    
    # 必用图片列表
    essential_images = [
        ("1_狗_清晰案例", "ppt_results/imagenet_visualizations/combined/imagenet_00000441_combined.png"),
        ("2_螃蟹_对比案例", "ppt_results/imagenet_visualizations/combined/imagenet_00006100_combined.png"),
        ("3_多目标场景", "ppt_results/imagenet_visualizations/combined/imagenet_00000110_combined.png"),
        ("4_V向量热力图", "ppt_results/imagenet_visualizations/v_vectors/imagenet_00000001_v_vectors.png"),
        ("5_CLS演化", "ppt_results/imagenet_visualizations/cls_layers/imagenet_00000001_cls_layers.png"),
        ("6_实验1_模块演化", "ppt_results/figures/exp1_module_0_ILSVRC2012_val_00000001_wrong_pred65.png"),
        ("7_实验2_V向量网格", "ppt_results/figures_final/exp2_v_vectors_002.png"),
        ("8_实验3_信息熵", "ppt_results/figures_v2/exp3_entropy_v2.png"),
        ("9_实验4_层间变化", "ppt_results/figures/exp4_changes_fixed.png"),
        ("10_实验5_距离衰减", "ppt_results/figures_v2/exp5_decay_v2.png"),
        ("11_实验6_集中度", "ppt_results/figures_v2/exp6_conc_head0_v2.png"),
    ]
    
    # 可选补充图片
    optional_images = [
        ("12_简单场景", "ppt_results/imagenet_visualizations/combined/imagenet_00000119_combined.png"),
        ("13_困难样本", "ppt_results/imagenet_visualizations/combined/imagenet_00000103_combined.png"),
        ("14_大型目标", "ppt_results/imagenet_visualizations/combined/imagenet_00000128_combined.png"),
        ("15_精细纹理", "ppt_results/imagenet_visualizations/combined/imagenet_00000145_combined.png"),
    ]
    
    success_count = 0
    fail_count = 0
    
    print("复制必用图片...")
    for name, path in essential_images:
        src = Path(path)
        if src.exists():
            dst = target_folder / f"{name}_{src.name}"
            shutil.copy2(src, dst)
            print(f"  OK {name}")
            success_count += 1
        else:
            print(f"  X {name} - 文件不存在")
            fail_count += 1
    
    print()
    print("复制可选图片...")
    for name, path in optional_images:
        src = Path(path)
        if src.exists():
            dst = target_folder / f"{name}_{src.name}"
            shutil.copy2(src, dst)
            print(f"  OK {name}")
            success_count += 1
        else:
            print(f"  ! {name} - 文件不存在（可选）")
    
    print()
    print("="*60)
    print("完成!")
    print(f"成功: {success_count} 张")
    if fail_count > 0:
        print(f"失败: {fail_count} 张")
    print("="*60)
    print()
    print(f"图片已保存到: {target_folder.absolute()}")
    print("可以直接在PPT中使用这些图片")
    
    # 生成图片清单
    list_file = target_folder / "图片清单.txt"
    with open(list_file, 'w', encoding='utf-8') as f:
        f.write("PPT图片清单\n")
        f.write("="*60 + "\n\n")
        f.write("必用图片 (11张):\n\n")
        for name, _ in essential_images:
            f.write(f"  {name}\n")
        f.write("\n可选补充图片 (4张):\n\n")
        for name, _ in optional_images:
            f.write(f"  {name}\n")
        f.write(f"\n" + "="*60 + "\n")
        f.write(f"总计: {success_count} 张图片\n")
    
    print(f"\n图片清单已生成: {list_file}")
    
    # 统计文件大小
    total_size = sum(f.stat().st_size for f in target_folder.glob("*.png"))
    size_mb = total_size / (1024 * 1024)
    print(f"总大小: {size_mb:.2f} MB")

if __name__ == "__main__":
    prepare_ppt_images()
