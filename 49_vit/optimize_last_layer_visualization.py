"""
优化最后一层（Layer 46/50）的可视化效果
使用多种增强技术来改善深层注意力的可视化质量
"""
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import argparse
from pathlib import Path

class LastLayerOptimizer:
    def __init__(self, base_dir, output_dir, original_img_path):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.original_img_path = original_img_path
        
        # 加载原图
        self.original_img = self.load_original_image()
    
    def load_original_image(self):
        """加载原始图像"""
        if not os.path.exists(self.original_img_path):
            print(f"⚠️ 原图不存在: {self.original_img_path}")
            return None
        
        img = cv2.imread(self.original_img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    def load_attention_map(self, layer_idx, head_idx=0, attn_type="avg_cls_attn"):
        """加载注意力图"""
        attn_path = self.base_dir / str(layer_idx) / "input_attn" / f"head{head_idx}" / f"{attn_type}_head{head_idx}.png"
        
        if not attn_path.exists():
            print(f"⚠️ 注意力图不存在: {attn_path}")
            return None
        
        # 加载图像
        img = cv2.imread(str(attn_path))
        if img is None:
            return None
        
        # 转换为灰度并归一化到 [0, 1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        
        return normalized
    
    def load_class_feature_map(self, predicted_class=0):
        """加载类别特征图"""
        # 假设类别特征图在 scan_RGB/50/ 目录下
        rgb_dir = self.base_dir.parent / "0_n01443537_24scan_RGB" / "50"
        class_feature_path = rgb_dir / f"{predicted_class}.png"
        
        if not class_feature_path.exists():
            print(f"⚠️ 类别特征图不存在: {class_feature_path}")
            return None
        
        # 加载图像
        img = cv2.imread(str(class_feature_path))
        if img is None:
            return None
        
        # 转换为灰度并归一化到 [0, 1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        
        return normalized
    
    def enhance_contrast_clahe(self, img):
        """使用CLAHE（对比度限制自适应直方图均衡化）增强对比度"""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_uint8)
        
        return enhanced.astype(np.float32) / 255.0
    
    def enhance_contrast_percentile(self, img, low=1, high=99):
        """使用百分位数裁剪增强对比度"""
        p_low = np.percentile(img, low)
        p_high = np.percentile(img, high)
        
        img_clipped = np.clip(img, p_low, p_high)
        img_normalized = (img_clipped - p_low) / (p_high - p_low + 1e-8)
        
        return img_normalized
    
    def enhance_contrast_adaptive(self, img):
        """自适应对比度增强（根据图像统计特性选择方法）"""
        # 计算变异系数
        cv = np.std(img) / (np.mean(img) + 1e-8)
        
        if cv < 0.1:
            # 变异系数很小，使用更激进的百分位裁剪
            return self.enhance_contrast_percentile(img, low=0.5, high=99.5)
        elif cv < 0.3:
            # 变异系数适中，使用CLAHE
            return self.enhance_contrast_clahe(img)
        else:
            # 变异系数较大，使用温和的百分位裁剪
            return self.enhance_contrast_percentile(img, low=2, high=98)
    
    def apply_guided_filter(self, img, radius=8, eps=0.01):
        """应用引导滤波（保边去噪）"""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 引导滤波
        filtered = cv2.ximgproc.guidedFilter(img_uint8, img_uint8, radius, eps)
        
        return filtered.astype(np.float32) / 255.0
    
    def fuse_with_class_feature(self, attn_map, class_feature_map, alpha_attn=0.3, alpha_class=0.7):
        """融合注意力图和类别特征图"""
        if class_feature_map is None:
            return attn_map
        
        # 确保尺寸一致
        if attn_map.shape != class_feature_map.shape:
            class_feature_map = cv2.resize(class_feature_map, 
                                          (attn_map.shape[1], attn_map.shape[0]),
                                          interpolation=cv2.INTER_CUBIC)
        
        # 归一化
        attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        class_norm = (class_feature_map - class_feature_map.min()) / (class_feature_map.max() - class_feature_map.min() + 1e-8)
        
        # 融合
        fused = alpha_attn * attn_norm + alpha_class * class_norm
        
        return fused
    
    def apply_morphological_operations(self, img, operation='close', kernel_size=5):
        """应用形态学操作（闭运算/开运算）"""
        img_uint8 = (img * 255).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'close':
            # 闭运算：先膨胀后腐蚀，填充小孔
            processed = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            # 开运算：先腐蚀后膨胀，去除小噪点
            processed = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
        else:
            processed = img_uint8
        
        return processed.astype(np.float32) / 255.0
    
    def apply_bilateral_filter(self, img, d=9, sigmaColor=75, sigmaSpace=75):
        """应用双边滤波（保边平滑）"""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 双边滤波
        filtered = cv2.bilateralFilter(img_uint8, d, sigmaColor, sigmaSpace)
        
        return filtered.astype(np.float32) / 255.0
    
    def optimize_last_layer(self, layer_idx=46, num_heads=3, predicted_class=0):
        """优化最后一层的可视化"""
        print(f"\n[*] 优化 Layer {layer_idx} 的可视化效果...")
        
        # 加载类别特征图
        class_feature_map = self.load_class_feature_map(predicted_class)
        
        for head_idx in range(num_heads):
            print(f"\n  处理 Head {head_idx}...")
            
            # 加载原始注意力图
            attn_map = self.load_attention_map(layer_idx, head_idx)
            
            if attn_map is None:
                print(f"    ⚠️ 跳过 Head {head_idx}")
                continue
            
            # 创建多个优化版本
            optimizations = {
                'original': attn_map,
                'percentile_clipping': self.enhance_contrast_percentile(attn_map, low=1, high=99),
                'clahe': self.enhance_contrast_clahe(attn_map),
                'adaptive': self.enhance_contrast_adaptive(attn_map),
                'bilateral_filter': self.apply_bilateral_filter(attn_map),
                'morphological_close': self.apply_morphological_operations(attn_map, 'close'),
            }
            
            # 如果有类别特征图，添加融合版本
            if class_feature_map is not None:
                optimizations['fused_with_class'] = self.fuse_with_class_feature(attn_map, class_feature_map)
                optimizations['fused_enhanced'] = self.enhance_contrast_adaptive(
                    self.fuse_with_class_feature(attn_map, class_feature_map)
                )
            
            # 保存所有优化版本
            for opt_name, opt_map in optimizations.items():
                self.save_visualization(opt_map, layer_idx, head_idx, opt_name)
            
            print(f"    ✓ 生成 {len(optimizations)} 个优化版本")
        
        # 创建对比网格
        self.create_comparison_grid(layer_idx, num_heads)
    
    def save_visualization(self, attn_map, layer_idx, head_idx, opt_name):
        """保存可视化结果"""
        # 应用colormap
        attn_uint8 = (attn_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
        
        # 调整到512x512
        if colored.shape[:2] != (512, 512):
            colored = cv2.resize(colored, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        
        # 保存
        output_path = self.output_dir / f"layer{layer_idx:02d}_head{head_idx}_{opt_name}.png"
        cv2.imwrite(str(output_path), colored)
    
    def create_comparison_grid(self, layer_idx, num_heads):
        """创建对比网格图"""
        print(f"\n  创建对比网格图...")
        
        # 优化方法列表
        opt_methods = [
            'original',
            'percentile_clipping',
            'clahe',
            'adaptive',
            'bilateral_filter',
            'morphological_close',
            'fused_with_class',
            'fused_enhanced',
        ]
        
        # 创建网格（rows=num_heads+1, cols=len(opt_methods)）
        rows = num_heads + 1  # +1 for original image
        cols = len(opt_methods)
        
        img_size = 512
        grid_width = cols * img_size
        grid_height = rows * img_size
        
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # 添加原图（第一行，居中）
        if self.original_img is not None:
            orig_img_pil = Image.fromarray(self.original_img)
            orig_img_resized = orig_img_pil.resize((img_size, img_size), Image.Resampling.LANCZOS)
            
            # 居中放置在第一行
            center_col = cols // 2
            grid_image.paste(orig_img_resized, (center_col * img_size, 0))
        
        # 添加优化后的注意力图
        for head_idx in range(num_heads):
            for col_idx, opt_name in enumerate(opt_methods):
                img_path = self.output_dir / f"layer{layer_idx:02d}_head{head_idx}_{opt_name}.png"
                
                if img_path.exists():
                    img = Image.open(img_path)
                    if img.size != (img_size, img_size):
                        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    
                    x = col_idx * img_size
                    y = (head_idx + 1) * img_size
                    grid_image.paste(img, (x, y))
        
        # 添加文字标签
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 列标签（优化方法）
        for col_idx, opt_name in enumerate(opt_methods):
            x = col_idx * img_size + 10
            y = 10
            draw.text((x, y), opt_name, fill='white', font=font_small, stroke_width=2, stroke_fill='black')
        
        # 行标签（原图 + 各个head）
        draw.text((10, 10), "Original Image", fill='white', font=font, stroke_width=2, stroke_fill='black')
        
        for head_idx in range(num_heads):
            y = (head_idx + 1) * img_size + 10
            draw.text((10, y), f"Head {head_idx}", fill='white', font=font, stroke_width=2, stroke_fill='black')
        
        # 保存网格图
        grid_path = self.output_dir / f"layer{layer_idx:02d}_optimization_comparison.png"
        grid_image.save(grid_path)
        
        print(f"  ✓ 对比网格图已保存: {grid_path}")
        
        return grid_path

def main():
    parser = argparse.ArgumentParser(description='优化最后一层的可视化效果')
    parser.add_argument('--base_dir', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final\0_n01443537_24scan_attn_map',
                       help='注意力图基础目录')
    parser.add_argument('--output_dir', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\last_layer_optimization',
                       help='输出目录')
    parser.add_argument('--original_img', type=str,
                       default=r'D:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG',
                       help='原始图像路径')
    parser.add_argument('--layer_idx', type=int, default=46, help='要优化的层索引')
    parser.add_argument('--num_heads', type=int, default=3, help='注意力头数量')
    parser.add_argument('--predicted_class', type=int, default=0, help='预测的类别索引')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Last Layer Visualization Optimization")
    print("=" * 80)
    
    optimizer = LastLayerOptimizer(args.base_dir, args.output_dir, args.original_img)
    
    # 优化最后一层
    optimizer.optimize_last_layer(
        layer_idx=args.layer_idx,
        num_heads=args.num_heads,
        predicted_class=args.predicted_class
    )
    
    print("\n" + "=" * 80)
    print("✅ 优化完成！")
    print(f"所有结果已保存到: {args.output_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()

