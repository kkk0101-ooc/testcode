"""
生成更多ImageNet图片的可视化
包括：
1. 原图 + V向量网格（12个head）
2. 原图 + 各层CLS V-magnitude（12层）
3. 完整的可视化报告
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
from PIL import Image
import json

class ImageNetVisualizer:
    def __init__(self):
        self.scan_dir = Path("scan_results/scan_vit/114VisionTransformer_ImageNet/vit_base_patch16_224_pretrained")
        self.output_dir = Path("ppt_results/imagenet_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.v_vectors_dir = self.output_dir / "v_vectors"
        self.cls_layers_dir = self.output_dir / "cls_layers"
        self.combined_dir = self.output_dir / "combined"
        
        for d in [self.v_vectors_dir, self.cls_layers_dir, self.combined_dir]:
            d.mkdir(exist_ok=True)
    
    def get_imagenet_dirs(self, num_images=30):
        """获取ImageNet图片目录（只要ILSVRC开头的）"""
        all_dirs = sorted([d for d in self.scan_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')])
        
        # 过滤出ImageNet验证集图片（ILSVRC2012_val开头）
        imagenet_dirs = [d for d in all_dirs if 'ILSVRC2012_val' in d.name]
        
        print(f"找到 {len(imagenet_dirs)} 张ImageNet图片")
        return imagenet_dirs[:num_images]
    
    def extract_image_info(self, dir_name):
        """从目录名提取图片信息"""
        # 格式: 0_ILSVRC2012_val_00000001_wrong_pred65
        parts = dir_name.split('_')
        
        # 找到val_xxxxx部分
        img_id = "unknown"
        for i, part in enumerate(parts):
            if part == 'val' and i+1 < len(parts):
                img_id = parts[i+1]  # 00000001
                break
        
        # 提取pred ID
        pred_id = parts[-1].replace('pred', '') if 'pred' in parts[-1] else "unknown"
        
        return img_id, pred_id
    
    # ========================================================================
    # 可视化1: 原图 + 12个head的V向量 (Layer 11)
    # ========================================================================
    def visualize_v_vectors(self, img_dir, output_idx):
        """生成V向量可视化（12个head + 原图）"""
        # 找原图
        tem0_dir = img_dir / "tem0"
        if not tem0_dir.exists():
            return False
        
        img_files = (list(tem0_dir.glob("*.JPEG")) + list(tem0_dir.glob("*.jpg")) + 
                    list(tem0_dir.glob("*.png")))
        if not img_files:
            return False
        
        original_img = Image.open(img_files[0]).convert('RGB')
        original_img_resized = original_img.resize((224, 224))
        
        # 找V向量（Layer 11 = folder 35）
        scan_v_35 = img_dir / "scan_v" / "35"
        if not scan_v_35.exists():
            return False
        
        # 提取图片信息
        img_id, pred_id = self.extract_image_info(img_dir.name)
        
        # 创建3×5网格
        fig, axes = plt.subplots(3, 5, figsize=(22, 13))
        fig.suptitle(f'ImageNet Image: {img_id} (Pred: {pred_id}) - V-Vector Magnitudes at Layer 11', 
                   fontsize=18, fontweight='bold', y=0.995)
        
        # 第一个位置: 原图
        axes[0, 0].imshow(original_img_resized)
        axes[0, 0].set_title('Original Image', fontsize=13, fontweight='bold', 
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        axes[0, 0].axis('off')
        
        # 剩余14个位置: 12个head的V-magnitude
        plot_idx = 1
        for head_idx in range(12):
            v_file = scan_v_35 / f"cls_v_magnitude_head{head_idx}.png"
            
            if v_file.exists() and plot_idx < 15:
                row = plot_idx // 5
                col = plot_idx % 5
                
                v_img = cv2.imread(str(v_file))
                if v_img is not None:
                    v_gray = cv2.cvtColor(v_img, cv2.COLOR_BGR2GRAY)
                    im = axes[row, col].imshow(v_gray, cmap='hot', interpolation='bilinear')
                    axes[row, col].set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
                axes[row, col].axis('off')
                plot_idx += 1
        
        # 隐藏多余的subplot
        for idx in range(plot_idx, 15):
            row = idx // 5
            col = idx % 5
            axes[row, col].axis('off')
        
        output_file = self.v_vectors_dir / f"imagenet_{img_id}_v_vectors.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        return True
    
    # ========================================================================
    # 可视化2: 原图 + 各层CLS V-magnitude（Head 0）
    # ========================================================================
    def visualize_cls_layers(self, img_dir, output_idx):
        """生成各层CLS可视化（12层 + 原图，Head 0）"""
        # 找原图
        tem0_dir = img_dir / "tem0"
        if not tem0_dir.exists():
            return False
        
        img_files = (list(tem0_dir.glob("*.JPEG")) + list(tem0_dir.glob("*.jpg")) + 
                    list(tem0_dir.glob("*.png")))
        if not img_files:
            return False
        
        original_img = Image.open(img_files[0]).convert('RGB')
        original_img_resized = original_img.resize((224, 224))
        
        scan_v_dir = img_dir / "scan_v"
        if not scan_v_dir.exists():
            return False
        
        # 提取图片信息
        img_id, pred_id = self.extract_image_info(img_dir.name)
        
        layer_folders = ['02', '05', '08', '11', '14', '17', '20', '23', '26', '29', '32', '35']
        
        # 创建3×5网格
        fig, axes = plt.subplots(3, 5, figsize=(22, 13))
        fig.suptitle(f'ImageNet Image: {img_id} (Pred: {pred_id}) - CLS V-Magnitude Across Layers (Head 0)', 
                   fontsize=18, fontweight='bold', y=0.995)
        
        # 第一个位置: 原图
        axes[0, 0].imshow(original_img_resized)
        axes[0, 0].set_title('Original Image', fontsize=13, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        axes[0, 0].axis('off')
        
        # 剩余位置: 12层的CLS V-magnitude (Head 0)
        plot_idx = 1
        for layer_idx, layer_folder in enumerate(layer_folders):
            if plot_idx >= 15:
                break
            
            layer_dir = scan_v_dir / layer_folder
            v_file = layer_dir / "cls_v_magnitude_head0.png"
            
            row = plot_idx // 5
            col = plot_idx % 5
            
            if v_file.exists():
                v_img = cv2.imread(str(v_file))
                if v_img is not None:
                    v_gray = cv2.cvtColor(v_img, cv2.COLOR_BGR2GRAY)
                    im = axes[row, col].imshow(v_gray, cmap='viridis', interpolation='bilinear')
                    axes[row, col].set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            else:
                axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16)
            
            axes[row, col].axis('off')
            plot_idx += 1
        
        # 隐藏多余的subplot
        for idx in range(plot_idx, 15):
            row = idx // 5
            col = idx % 5
            axes[row, col].axis('off')
        
        output_file = self.cls_layers_dir / f"imagenet_{img_id}_cls_layers.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        return True
    
    # ========================================================================
    # 可视化3: 组合图（原图 + V向量 + 各层CLS）
    # ========================================================================
    def visualize_combined(self, img_dir, output_idx):
        """生成组合可视化（上：V向量，下：各层CLS）"""
        # 找原图
        tem0_dir = img_dir / "tem0"
        if not tem0_dir.exists():
            return False
        
        img_files = (list(tem0_dir.glob("*.JPEG")) + list(tem0_dir.glob("*.jpg")) + 
                    list(tem0_dir.glob("*.png")))
        if not img_files:
            return False
        
        original_img = Image.open(img_files[0]).convert('RGB')
        
        # 提取图片信息
        img_id, pred_id = self.extract_image_info(img_dir.name)
        
        # 创建大图：3行布局
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 3, 3], hspace=0.15)
        
        # 第一行：原图（大）
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(original_img)
        ax_orig.set_title(f'ImageNet Image: {img_id} (Pred: {pred_id})', 
                         fontsize=20, fontweight='bold', pad=20)
        ax_orig.axis('off')
        
        # 第二行：V向量（12个head）
        gs_v = gs[1, 0].subgridspec(2, 6, hspace=0.3, wspace=0.2)
        scan_v_35 = img_dir / "scan_v" / "35"
        
        if scan_v_35.exists():
            for head_idx in range(12):
                row = head_idx // 6
                col = head_idx % 6
                ax = fig.add_subplot(gs_v[row, col])
                
                v_file = scan_v_35 / f"cls_v_magnitude_head{head_idx}.png"
                if v_file.exists():
                    v_img = cv2.imread(str(v_file))
                    if v_img is not None:
                        v_gray = cv2.cvtColor(v_img, cv2.COLOR_BGR2GRAY)
                        im = ax.imshow(v_gray, cmap='hot', interpolation='bilinear')
                        ax.set_title(f'V-Vec H{head_idx}', fontsize=10, fontweight='bold')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.axis('off')
        
        # 第三行：各层CLS（12层，Head 0）
        gs_cls = gs[2, 0].subgridspec(2, 6, hspace=0.3, wspace=0.2)
        scan_v_dir = img_dir / "scan_v"
        layer_folders = ['02', '05', '08', '11', '14', '17', '20', '23', '26', '29', '32', '35']
        
        if scan_v_dir.exists():
            for layer_idx, layer_folder in enumerate(layer_folders):
                row = layer_idx // 6
                col = layer_idx % 6
                ax = fig.add_subplot(gs_cls[row, col])
                
                layer_dir = scan_v_dir / layer_folder
                v_file = layer_dir / "cls_v_magnitude_head0.png"
                
                if v_file.exists():
                    v_img = cv2.imread(str(v_file))
                    if v_img is not None:
                        v_gray = cv2.cvtColor(v_img, cv2.COLOR_BGR2GRAY)
                        im = ax.imshow(v_gray, cmap='viridis', interpolation='bilinear')
                        ax.set_title(f'CLS L{layer_idx}', fontsize=10, fontweight='bold')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.axis('off')
        
        output_file = self.combined_dir / f"imagenet_{img_id}_combined.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        return True
    
    # ========================================================================
    # 批量生成
    # ========================================================================
    def generate_all(self, num_images=20):
        """批量生成所有可视化"""
        print("\n" + "="*80)
        print(f"生成ImageNet可视化 (前{num_images}张)")
        print("="*80)
        
        imagenet_dirs = self.get_imagenet_dirs(num_images)
        
        results = {
            'v_vectors': 0,
            'cls_layers': 0,
            'combined': 0,
            'images': []
        }
        
        for idx, img_dir in enumerate(tqdm(imagenet_dirs, desc="生成可视化")):
            img_id, pred_id = self.extract_image_info(img_dir.name)
            
            img_result = {
                'id': img_id,
                'pred': pred_id,
                'dir': img_dir.name
            }
            
            # 生成V向量可视化
            if self.visualize_v_vectors(img_dir, idx):
                results['v_vectors'] += 1
                img_result['v_vectors'] = f"imagenet_{img_id}_v_vectors.png"
            
            # 生成各层CLS可视化
            if self.visualize_cls_layers(img_dir, idx):
                results['cls_layers'] += 1
                img_result['cls_layers'] = f"imagenet_{img_id}_cls_layers.png"
            
            # 生成组合图
            if self.visualize_combined(img_dir, idx):
                results['combined'] += 1
                img_result['combined'] = f"imagenet_{img_id}_combined.png"
            
            results['images'].append(img_result)
        
        # 保存结果索引
        index_file = self.output_dir / "visualization_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nOK 生成完成:")
        print(f"   V向量可视化: {results['v_vectors']} 张")
        print(f"   各层CLS可视化: {results['cls_layers']} 张")
        print(f"   组合可视化: {results['combined']} 张")
        print(f"   总计: {results['v_vectors'] + results['cls_layers'] + results['combined']} 张图片")
        
        return results
    
    # ========================================================================
    # 生成Markdown索引
    # ========================================================================
    def create_index_markdown(self, results):
        """生成可视化索引Markdown"""
        print("\n生成索引文档...")
        
        md_content = f"""# ImageNet可视化图库

## 📊 统计

- **V向量可视化**: {results['v_vectors']} 张
- **各层CLS可视化**: {results['cls_layers']} 张
- **组合可视化**: {results['combined']} 张
- **总计**: {results['v_vectors'] + results['cls_layers'] + results['combined']} 张图片

---

## 🖼️ 可视化类型说明

### 1. V向量可视化 (v_vectors/)
- **原图** + **12个attention head的V-magnitude** (Layer 11)
- 热力图显示，colormap='hot'
- 3行×5列布局

### 2. 各层CLS可视化 (cls_layers/)
- **原图** + **12层的CLS V-magnitude** (Head 0)
- 热力图显示，colormap='viridis'
- 3行×5列布局

### 3. 组合可视化 (combined/)
- **上**: 原图（大）
- **中**: 12个head的V向量 (2×6布局)
- **下**: 12层的CLS (2×6布局)
- 完整的信息展示

---

## 📁 图片列表

| ID | 预测ID | V向量 | 各层CLS | 组合图 |
|----|--------|-------|---------|--------|
"""
        
        for img in results['images']:
            v_check = "✅" if 'v_vectors' in img else "❌"
            cls_check = "✅" if 'cls_layers' in img else "❌"
            comb_check = "✅" if 'combined' in img else "❌"
            
            md_content += f"| {img['id']} | {img['pred']} | {v_check} | {cls_check} | {comb_check} |\n"
        
        md_content += f"""
---

## 📂 文件结构

```
ppt_results/imagenet_visualizations/
├── v_vectors/                      # V向量可视化（{results['v_vectors']}张）
│   ├── imagenet_val_00000001_v_vectors.png
│   ├── imagenet_val_00000002_v_vectors.png
│   └── ...
├── cls_layers/                     # 各层CLS可视化（{results['cls_layers']}张）
│   ├── imagenet_val_00000001_cls_layers.png
│   ├── imagenet_val_00000002_cls_layers.png
│   └── ...
├── combined/                       # 组合可视化（{results['combined']}张）
│   ├── imagenet_val_00000001_combined.png
│   ├── imagenet_val_00000002_combined.png
│   └── ...
├── visualization_index.json        # 图片索引（JSON格式）
└── README.md                       # 本文档
```

---

## 🎯 使用建议

### 论文使用
- **图表质量**: 所有图片均为200 DPI高分辨率
- **推荐使用**: `combined/` 目录中的组合图（信息最全）
- **单独分析**: 使用 `v_vectors/` 或 `cls_layers/` 进行专项分析

### 快速查看
1. 打开对应目录
2. 按文件名排序（已按ImageNet ID排序）
3. 查看 `visualization_index.json` 获取完整索引

---

**生成时间**: 2025-11-21
**数据集**: ImageNet-1K Validation Set
**模型**: ViT-Base-16
"""
        
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"OK 索引文档保存: {readme_file}")

def main():
    visualizer = ImageNetVisualizer()
    
    # 生成20张ImageNet图片的可视化
    results = visualizer.generate_all(num_images=20)
    
    # 生成索引文档
    visualizer.create_index_markdown(results)
    
    print("\n" + "="*80)
    print("全部完成！")
    print("="*80)
    print(f"\n输出目录: ppt_results/imagenet_visualizations/")
    print(f"  - v_vectors/: V向量可视化")
    print(f"  - cls_layers/: 各层CLS可视化")
    print(f"  - combined/: 组合可视化")
    print(f"  - README.md: 图片索引")

if __name__ == "__main__":
    main()
