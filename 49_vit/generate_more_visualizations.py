"""
为新解压的ImageNet图片生成更多可视化
从已有的scan结果中选择，避免重新扫描
"""

from pathlib import Path
import json
import shutil
from generate_imagenet_visualizations import ImageNetVisualizer

def find_available_images(max_images=50):
    """找到已经扫描过的ImageNet图片"""
    scan_dir = Path("scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained")
    
    if not scan_dir.exists():
        print(f"错误: scan_results目录不存在")
        return []
    
    # 找到所有已扫描的ImageNet目录
    all_dirs = sorted([d for d in scan_dir.iterdir() 
                      if d.is_dir() and 'ILSVRC2012_val' in d.name])
    
    print(f"找到 {len(all_dirs)} 张已扫描的ImageNet图片")
    
    # 验证每个目录是否有完整的数据
    valid_dirs = []
    for img_dir in all_dirs[:max_images]:
        # 检查是否有必要的文件
        tem0_dir = img_dir / "tem0"
        scan_v_dir = img_dir / "scan_v"
        
        if tem0_dir.exists() and scan_v_dir.exists():
            # 检查是否有足够的层级数据
            layer_dirs = [d for d in scan_v_dir.iterdir() if d.is_dir()]
            if len(layer_dirs) >= 10:  # 至少有10层的数据
                valid_dirs.append(img_dir)
    
    print(f"其中 {len(valid_dirs)} 张有完整数据")
    return valid_dirs

def check_disk_space():
    """检查磁盘空间"""
    import shutil
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    return free_gb

def main():
    print("="*80)
    print("生成更多ImageNet可视化")
    print("="*80)
    
    # 检查磁盘空间
    free_gb = check_disk_space()
    print(f"\n可用磁盘空间: {free_gb:.2f} GB")
    
    if free_gb < 10:
        print("磁盘空间不足10GB，停止生成")
        return
    
    # 找到可用的图片
    print("\n查找已扫描的图片...")
    valid_dirs = find_available_images(max_images=50)
    
    if len(valid_dirs) == 0:
        print("没有找到可用的图片")
        return
    
    print(f"\n将为 {len(valid_dirs)} 张图片生成可视化")
    
    # 使用已有的可视化器
    visualizer = ImageNetVisualizer()
    
    # 手动生成可视化（使用已有的目录）
    results = {
        'v_vectors': 0,
        'cls_layers': 0,
        'combined': 0,
        'images': []
    }
    
    from tqdm import tqdm
    
    for idx, img_dir in enumerate(tqdm(valid_dirs, desc="生成可视化")):
        # 提取图片信息
        img_id, pred_id = visualizer.extract_image_info(img_dir.name)
        
        img_result = {
            'id': img_id,
            'pred': pred_id,
            'dir': img_dir.name
        }
        
        try:
            # 生成V向量可视化
            v_success = visualizer.visualize_v_vectors(img_dir, idx)
            if v_success:
                results['v_vectors'] += 1
                img_result['v_vectors'] = f"imagenet_{img_id}_v_vectors.png"
        except Exception as e:
            print(f"\nV向量生成失败 {img_id}: {e}")
        
        try:
            # 生成各层CLS可视化
            cls_success = visualizer.visualize_cls_layers(img_dir, idx)
            if cls_success:
                results['cls_layers'] += 1
                img_result['cls_layers'] = f"imagenet_{img_id}_cls_layers.png"
        except Exception as e:
            print(f"\nCLS生成失败 {img_id}: {e}")
        
        try:
            # 生成组合可视化
            combined_success = visualizer.visualize_combined(img_dir, idx)
            if combined_success:
                results['combined'] += 1
                img_result['combined'] = f"imagenet_{img_id}_combined.png"
        except Exception as e:
            print(f"\n组合图生成失败 {img_id}: {e}")
        
        results['images'].append(img_result)
        
        # 每10张检查一次磁盘空间
        if (idx + 1) % 10 == 0:
            free_gb = check_disk_space()
            if free_gb < 5:
                print(f"\n\n磁盘空间不足5GB，停止生成")
                break
    
    # 保存索引
    index_file = visualizer.output_dir / "visualization_index_extended.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n生成完成:")
    print(f"  V向量: {results['v_vectors']} 张")
    print(f"  各层CLS: {results['cls_layers']} 张")
    print(f"  组合图: {results['combined']} 张")
    print(f"  总计: {results['v_vectors'] + results['cls_layers'] + results['combined']} 张")
    
    # 更新README
    visualizer.create_index_markdown(results)
    
    print(f"\n索引已更新: {visualizer.output_dir / 'README.md'}")

if __name__ == "__main__":
    main()
