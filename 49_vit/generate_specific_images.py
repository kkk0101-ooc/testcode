"""
为指定的ImageNet图片生成可视化
专门处理 00000441 和 00006100
"""

import sys
import os
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from generate_imagenet_visualizations import ImageNetVisualizer

def generate_for_specific_images():
    """为两张指定图片生成可视化"""
    
    print("="*80)
    print("为指定图片生成可视化")
    print("="*80)
    print()
    
    # 要处理的图片ID
    target_ids = ["00000441", "00006100"]
    
    # 扫描结果基础目录
    scan_base = Path("scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained")
    
    # 查找这些图片的目录
    found_dirs = []
    
    print("查找图片目录...")
    if scan_base.exists():
        all_dirs = [d for d in scan_base.iterdir() if d.is_dir()]
        
        for target_id in target_ids:
            # 查找包含这个ID的目录
            matching = [d for d in all_dirs if f"val_{target_id}" in d.name]
            
            if matching:
                found_dirs.append((target_id, matching[0]))
                print(f"  找到: {target_id} -> {matching[0].name}")
            else:
                print(f"  未找到: {target_id} (需要先运行扫描)")
    else:
        print(f"  错误: scan_results目录不存在")
        return
    
    if len(found_dirs) == 0:
        print("\n没有找到任何图片目录")
        print("需要先对这两张图片运行NNS扫描")
        return
    
    print(f"\n找到 {len(found_dirs)} 张图片，开始生成可视化...")
    print()
    
    # 创建可视化器
    visualizer = ImageNetVisualizer()
    
    results = {
        'images': []
    }
    
    for idx, (img_id, img_dir) in enumerate(found_dirs):
        print(f"\n处理图片 {img_id}...")
        
        img_result = {
            'id': img_id,
            'dir': img_dir.name
        }
        
        try:
            # 生成V向量可视化
            print("  - 生成V向量可视化...")
            v_success = visualizer.visualize_v_vectors(img_dir, idx)
            if v_success:
                img_result['v_vectors'] = f"imagenet_{img_id}_v_vectors.png"
                print("    OK")
        except Exception as e:
            print(f"    错误: {e}")
        
        try:
            # 生成各层CLS可视化
            print("  - 生成CLS演化可视化...")
            cls_success = visualizer.visualize_cls_layers(img_dir, idx)
            if cls_success:
                img_result['cls_layers'] = f"imagenet_{img_id}_cls_layers.png"
                print("    OK")
        except Exception as e:
            print(f"    错误: {e}")
        
        try:
            # 生成组合可视化
            print("  - 生成组合可视化...")
            combined_success = visualizer.visualize_combined(img_dir, idx)
            if combined_success:
                img_result['combined'] = f"imagenet_{img_id}_combined.png"
                print("    OK")
        except Exception as e:
            print(f"    错误: {e}")
        
        results['images'].append(img_result)
    
    print("\n" + "="*80)
    print("完成!")
    print("="*80)
    print()
    
    # 显示生成的文件
    print("生成的文件:")
    for img in results['images']:
        print(f"\n图片 {img['id']}:")
        if 'combined' in img:
            print(f"  - 组合图: ppt_results/imagenet_visualizations/combined/{img['combined']}")
        if 'v_vectors' in img:
            print(f"  - V向量: ppt_results/imagenet_visualizations/v_vectors/{img['v_vectors']}")
        if 'cls_layers' in img:
            print(f"  - CLS演化: ppt_results/imagenet_visualizations/cls_layers/{img['cls_layers']}")

if __name__ == "__main__":
    generate_for_specific_images()
