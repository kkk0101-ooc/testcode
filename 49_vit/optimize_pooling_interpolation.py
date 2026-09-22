"""
优化Max Pooling和插值参数
测试不同的pooling kernel size、stride和插值方法，找到显示清晰轮廓的最佳组合

参考论文：
- CAM/Grad-CAM生成小尺寸热力图，然后插值放大，所以更平滑
- 对NNS进行max pooling和插值处理，减少对比差异
- 在ImageNet上，NNS能描述轮廓和重要细节
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path

class PoolingInterpolationOptimizer:
    def __init__(self, attention_dir, original_img_path, output_dir):
        self.attention_dir = Path(attention_dir)
        self.original_img_path = original_img_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载原图
        self.original_img = cv2.imread(self.original_img_path)
        if self.original_img is not None:
            self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        
        print(f"\n[初始化]")
        print(f"  注意力目录: {self.attention_dir}")
        print(f"  原图: {self.original_img_path}")
        print(f"  输出目录: {self.output_dir}")
    
    def load_attention_map(self, layer_idx, head_idx=0):
        """加载注意力图（原始PNG）"""
        attn_path = self.attention_dir / str(layer_idx) / "input_attn" / f"head{head_idx}" / f"avg_cls_attn_head{head_idx}.png"
        
        if not attn_path.exists():
            return None
        
        img = cv2.imread(str(attn_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # 归一化到[0,1]
        return img.astype(np.float32) / 255.0
    
    def apply_max_pooling_and_interpolation(self, img, 
                                           pool_kernel=2, 
                                           pool_stride=1,
                                           pool_padding=1,
                                           interp_mode='bicubic',
                                           target_size=(512, 512)):
        """
        应用max pooling和插值
        
        参数：
        - pool_kernel: pooling kernel size (1, 2, 3, 4)
        - pool_stride: pooling stride (1表示重叠，>1表示下采样)
        - pool_padding: pooling padding
        - interp_mode: 插值模式 ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area')
        - target_size: 目标尺寸
        """
        # 转换为tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Max pooling（增强重要区域）
        if pool_kernel > 1:
            pooled = F.max_pool2d(
                img_tensor,
                kernel_size=pool_kernel,
                stride=pool_stride,
                padding=pool_padding
            )
        else:
            pooled = img_tensor
        
        # 插值到目标尺寸
        interpolated = F.interpolate(
            pooled,
            size=target_size,
            mode=interp_mode,
            align_corners=False if interp_mode != 'nearest' else None
        )
        
        # 转回numpy
        result = interpolated[0, 0].numpy()
        
        # 确保在[0,1]范围
        result = np.clip(result, 0, 1)
        
        return result
    
    def test_all_combinations(self, layer_idx, head_idx=0):
        """测试所有pooling和插值组合"""
        print(f"\n[测试] Layer {layer_idx}, Head {head_idx}")
        
        # 加载原始注意力图
        attn_map = self.load_attention_map(layer_idx, head_idx)
        if attn_map is None:
            print(f"  [错误] 注意力图不存在")
            return None
        
        print(f"  原始尺寸: {attn_map.shape}")
        
        # 参数组合
        test_configs = [
            # (kernel, stride, padding, interp_mode, 描述)
            # 论文推荐的组合
            (2, 1, 1, 'bicubic', 'paper_recommended'),  # 小kernel，重叠pooling，bicubic
            (3, 1, 1, 'bicubic', 'larger_kernel_bicubic'),
            (4, 1, 2, 'bicubic', 'large_kernel_bicubic'),
            
            # 不同stride（影响平滑度）
            (2, 1, 1, 'bicubic', 'stride1_bicubic'),  # 重叠pooling（更平滑）
            (2, 2, 0, 'bicubic', 'stride2_bicubic'),  # 不重叠pooling
            
            # 不同插值方法
            (2, 1, 1, 'bilinear', 'bilinear'),
            (2, 1, 1, 'nearest', 'nearest'),
            (2, 1, 1, 'area', 'area'),
            
            # 极端组合（更强的pooling）
            (3, 1, 1, 'bilinear', 'k3_bilinear'),
            (4, 1, 2, 'bilinear', 'k4_bilinear'),
            (5, 1, 2, 'bicubic', 'k5_bicubic'),
            
            # 无pooling（只插值）
            (1, 1, 0, 'bicubic', 'no_pooling_bicubic'),
            (1, 1, 0, 'bilinear', 'no_pooling_bilinear'),
            
            # 强pooling + 平滑插值
            (4, 1, 2, 'area', 'k4_area'),  # area可能更平滑
            (3, 1, 1, 'area', 'k3_area'),
        ]
        
        results = {}
        
        for kernel, stride, padding, interp_mode, name in test_configs:
            try:
                print(f"  - 测试 {name} (kernel={kernel}, stride={stride}, pad={padding}, interp={interp_mode})...")
                
                processed = self.apply_max_pooling_and_interpolation(
                    attn_map,
                    pool_kernel=kernel,
                    pool_stride=stride,
                    pool_padding=padding,
                    interp_mode=interp_mode,
                    target_size=(512, 512)
                )
                
                results[name] = {
                    'image': processed,
                    'config': (kernel, stride, padding, interp_mode)
                }
                
                # 保存
                self.save_heatmap(
                    processed, 
                    self.output_dir / f"layer{layer_idx:02d}_head{head_idx}_{name}.png",
                    overlay=True
                )
                
            except Exception as e:
                print(f"    [错误] {name}: {e}")
        
        print(f"  [完成] 测试了 {len(results)} 种组合")
        
        # 创建对比网格
        self.create_comparison_grid(layer_idx, head_idx, results)
        
        return results
    
    def save_heatmap(self, img, save_path, overlay=True):
        """保存热力图"""
        # 转换为uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 应用colormap
        colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
        
        # 保存纯热力图
        cv2.imwrite(str(save_path), colored)
        
        # 叠加到原图
        if overlay and self.original_img is not None:
            original_resized = cv2.resize(self.original_img, (512, 512))
            original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
            
            alpha = 0.5
            overlayed = cv2.addWeighted(original_bgr, alpha, colored, 1-alpha, 0)
            
            overlay_path = save_path.parent / (save_path.stem + "_overlay.png")
            cv2.imwrite(str(overlay_path), overlayed)
    
    def create_comparison_grid(self, layer_idx, head_idx, results):
        """创建对比网格"""
        print(f"  - 创建对比网格...")
        
        num_results = len(results)
        cols = 4
        rows = (num_results + cols - 1) // cols
        
        # 如果有原图，增加一行
        if self.original_img is not None:
            rows += 1
        
        img_size = 512
        grid_width = cols * img_size
        grid_height = rows * img_size
        
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # 添加原图（第一行居中）
        if self.original_img is not None:
            orig_pil = Image.fromarray(self.original_img)
            orig_resized = orig_pil.resize((img_size, img_size), Image.Resampling.LANCZOS)
            
            center_col = cols // 2
            grid_image.paste(orig_resized, (center_col * img_size, 0))
        
        # 添加处理结果
        start_row = 1 if self.original_img is not None else 0
        
        for idx, (name, data) in enumerate(results.items()):
            row = start_row + idx // cols
            col = idx % cols
            
            # 加载保存的图像
            img_path = self.output_dir / f"layer{layer_idx:02d}_head{head_idx}_{name}.png"
            if img_path.exists():
                img = Image.open(img_path)
                if img.size != (img_size, img_size):
                    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                
                x = col * img_size
                y = row * img_size
                grid_image.paste(img, (x, y))
        
        # 添加标签
        draw = ImageDraw.Draw(grid_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 原图标签
        if self.original_img is not None:
            draw.text((10, 10), "Original Image", fill='white', font=font,
                     stroke_width=2, stroke_fill='black')
        
        # 方法标签
        for idx, (name, data) in enumerate(results.items()):
            row = start_row + idx // cols
            col = idx % cols
            x = col * img_size + 10
            y = row * img_size + 10
            
            # 显示配置
            kernel, stride, padding, interp = data['config']
            label = f"{name}\nk={kernel},s={stride},p={padding}\n{interp}"
            
            draw.text((x, y), label, fill='white', font=font_small,
                     stroke_width=2, stroke_fill='black')
        
        # 保存网格
        grid_path = self.output_dir / f"layer{layer_idx:02d}_head{head_idx}_pooling_interp_comparison.png"
        grid_image.save(grid_path)
        
        print(f"  [保存] 对比网格: {grid_path}")
        
        return grid_path
    
    def find_best_combination(self, layer_idx, head_idx=0):
        """找到最佳组合（通过计算边缘强度和对比度）"""
        print(f"\n[评估] 寻找最佳组合...")
        
        # 加载所有结果
        result_files = list(self.output_dir.glob(f"layer{layer_idx:02d}_head{head_idx}_*.png"))
        result_files = [f for f in result_files if not f.name.endswith('_overlay.png') and 'comparison' not in f.name]
        
        scores = {}
        
        for file_path in result_files:
            # 加载图像
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # 计算指标
            # 1. 边缘强度（Sobel）
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_score = edge_magnitude.mean()
            
            # 2. 对比度（标准差）
            contrast_score = img.std()
            
            # 3. 边缘清晰度（Laplacian方差）
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            sharpness_score = laplacian.var()
            
            # 综合得分
            total_score = edge_score * 0.4 + contrast_score * 0.3 + sharpness_score * 0.3
            
            method_name = file_path.stem.replace(f"layer{layer_idx:02d}_head{head_idx}_", "")
            scores[method_name] = {
                'edge': edge_score,
                'contrast': contrast_score,
                'sharpness': sharpness_score,
                'total': total_score
            }
        
        # 排序
        sorted_methods = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print(f"\n  [排名] Top 5 最佳组合:")
        for i, (name, score) in enumerate(sorted_methods[:5], 1):
            print(f"    {i}. {name}")
            print(f"       边缘: {score['edge']:.2f}, 对比度: {score['contrast']:.2f}, "
                  f"清晰度: {score['sharpness']:.2f}, 总分: {score['total']:.2f}")
        
        # 保存评估结果
        eval_path = self.output_dir / f"layer{layer_idx:02d}_head{head_idx}_evaluation.txt"
        with open(eval_path, 'w') as f:
            f.write("Pooling & Interpolation 参数评估结果\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (name, score) in enumerate(sorted_methods, 1):
                f.write(f"{i}. {name}\n")
                f.write(f"   边缘强度: {score['edge']:.4f}\n")
                f.write(f"   对比度: {score['contrast']:.4f}\n")
                f.write(f"   清晰度: {score['sharpness']:.4f}\n")
                f.write(f"   总分: {score['total']:.4f}\n\n")
        
        print(f"\n  [保存] 评估结果: {eval_path}")
        
        return sorted_methods[0][0] if sorted_methods else None

def main():
    parser = argparse.ArgumentParser(description='优化Max Pooling和插值参数')
    parser.add_argument('--attention_dir', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final\0_n01443537_24scan_attn_map',
                       help='注意力图目录')
    parser.add_argument('--original_img', type=str,
                       default=r'D:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG',
                       help='原始图像路径')
    parser.add_argument('--output_dir', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\pooling_interp_optimized',
                       help='输出目录')
    parser.add_argument('--layer_idx', type=int, default=46, help='层索引')
    parser.add_argument('--head_idx', type=int, default=0, help='注意力头索引')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Max Pooling & 插值参数优化 - 显示清晰轮廓")
    print("=" * 80)
    print("\n参考论文方法:")
    print("  - CAM/Grad-CAM: 小尺寸热力图 + 插值放大 = 平滑")
    print("  - NNS改进: Max Pooling + 插值 = 平滑 + 保留细节")
    print("  - 目标: 在ImageNet上描述轮廓和重要细节")
    
    optimizer = PoolingInterpolationOptimizer(
        args.attention_dir,
        args.original_img,
        args.output_dir
    )
    
    # 测试所有组合
    results = optimizer.test_all_combinations(
        layer_idx=args.layer_idx,
        head_idx=args.head_idx
    )
    
    if results:
        # 自动评估找到最佳组合
        best_method = optimizer.find_best_combination(
            layer_idx=args.layer_idx,
            head_idx=args.head_idx
        )
        
        if best_method:
            print(f"\n" + "=" * 80)
            print(f"[推荐] 最佳组合: {best_method}")
            print(f"[查看] {args.output_dir}/layer{args.layer_idx:02d}_head{args.head_idx}_{best_method}_overlay.png")
            print("=" * 80)
    
    print(f"\n所有结果已保存到: {args.output_dir}")

if __name__ == '__main__':
    main()

