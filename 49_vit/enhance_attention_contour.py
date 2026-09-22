"""
超强注意力轮廓增强脚本
专门用于显示物体（鱼）的清晰轮廓
不修改训练代码，独立运行
"""
import os
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

class ContourEnhancer:
    def __init__(self, attention_dir, class_feature_dir, original_img_path, output_dir):
        self.attention_dir = Path(attention_dir)
        self.class_feature_dir = Path(class_feature_dir)
        self.original_img_path = original_img_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载原图
        self.original_img = cv2.imread(self.original_img_path)
        if self.original_img is not None:
            self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        
        print(f"\n[初始化]")
        print(f"  注意力目录: {self.attention_dir}")
        print(f"  类别特征目录: {self.class_feature_dir}")
        print(f"  原图: {self.original_img_path}")
        print(f"  输出目录: {self.output_dir}")
    
    def load_attention_map(self, layer_idx, head_idx=0):
        """加载注意力图"""
        attn_path = self.attention_dir / str(layer_idx) / "input_attn" / f"head{head_idx}" / f"avg_cls_attn_head{head_idx}.png"
        
        if not attn_path.exists():
            return None
        
        img = cv2.imread(str(attn_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        return img.astype(np.float32) / 255.0
    
    def load_class_feature(self, class_idx=0):
        """加载类别特征图"""
        class_path = self.class_feature_dir / f"{class_idx}.png"
        
        if not class_path.exists():
            return None
        
        img = cv2.imread(str(class_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        return img.astype(np.float32) / 255.0
    
    def method1_extreme_percentile(self, img):
        """方法1: 极端百分位裁剪（突出最重要的区域）"""
        p_low = np.percentile(img, 0.1)  # 更极端
        p_high = np.percentile(img, 99.9)
        
        img_clipped = np.clip(img, p_low, p_high)
        img_norm = (img_clipped - p_low) / (p_high - p_low + 1e-8)
        
        # 应用gamma校正增强对比度
        gamma = 0.5  # < 1 增强暗部
        img_enhanced = np.power(img_norm, gamma)
        
        return img_enhanced
    
    def method2_edge_aware(self, img):
        """方法2: 边缘感知增强"""
        # 1. Sobel边缘检测
        img_uint8 = (img * 255).astype(np.uint8)
        
        sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        
        # 边缘强度
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)
        
        # 2. 将边缘信息融合回原图
        alpha_edge = 0.4
        alpha_original = 0.6
        
        enhanced = alpha_original * img + alpha_edge * edge_magnitude
        enhanced = enhanced / (enhanced.max() + 1e-8)
        
        return enhanced
    
    def method3_top_k_masking(self, img, top_k_ratio=0.3):
        """方法3: Top-K遮罩（只保留最重要的区域）"""
        # 计算阈值
        threshold = np.percentile(img, (1 - top_k_ratio) * 100)
        
        # 创建二值遮罩
        mask = (img >= threshold).astype(np.float32)
        
        # 形态学闭运算，连接断开的区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask = mask_closed.astype(np.float32) / 255.0
        
        # 应用遮罩
        masked = img * mask
        
        # 归一化
        if masked.max() > 1e-8:
            masked = masked / masked.max()
        
        return masked
    
    def method4_class_feature_guided(self, attn_map, class_feature):
        """方法4: 类别特征强引导（主要用类别特征）"""
        if class_feature is None:
            return attn_map
        
        # 确保尺寸一致
        if attn_map.shape != class_feature.shape:
            class_feature = cv2.resize(class_feature, 
                                      (attn_map.shape[1], attn_map.shape[0]),
                                      interpolation=cv2.INTER_CUBIC)
        
        # 使用类别特征作为主导（90%），注意力作为辅助（10%）
        alpha_class = 0.9
        alpha_attn = 0.1
        
        # 归一化
        attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        class_norm = (class_feature - class_feature.min()) / (class_feature.max() - class_feature.min() + 1e-8)
        
        fused = alpha_class * class_norm + alpha_attn * attn_norm
        
        return fused
    
    def method5_watershed_segmentation(self, img):
        """方法5: 分水岭分割（清晰的目标分离）"""
        # 转换为uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 1. 阈值分割
        _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 3. 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 4. 距离变换找前景
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        
        sure_fg = sure_fg.astype(np.uint8)
        
        # 5. 结合原图，突出前景
        result = img.copy()
        result[sure_fg == 0] = result[sure_fg == 0] * 0.3  # 抑制背景
        
        # 归一化
        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        
        return result
    
    def method6_gradcam_style(self, attn_map):
        """方法6: GradCAM风格（突出激活区域）"""
        # 1. ReLU激活（去除负值）
        attn_relu = np.maximum(attn_map, 0)
        
        # 2. 归一化
        if attn_relu.max() > 1e-8:
            attn_norm = attn_relu / attn_relu.max()
        else:
            attn_norm = attn_relu
        
        # 3. 应用阈值，只保留强激活区域
        threshold = 0.5
        attn_norm[attn_norm < threshold] = 0
        
        # 4. 再次归一化
        if attn_norm.max() > 1e-8:
            attn_norm = attn_norm / attn_norm.max()
        
        return attn_norm
    
    def method7_superpixel_attention(self, img):
        """方法7: 超像素注意力（区域聚合）"""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 使用SLIC超像素分割
        # 注意：需要 scikit-image
        try:
            from skimage.segmentation import slic
            from skimage.segmentation import mark_boundaries
            
            # 转换为3通道
            img_3ch = np.stack([img_uint8] * 3, axis=-1)
            
            # SLIC分割
            segments = slic(img_3ch, n_segments=100, compactness=10, sigma=1)
            
            # 计算每个超像素的平均注意力
            result = np.zeros_like(img)
            for segment_id in np.unique(segments):
                mask = segments == segment_id
                avg_val = img[mask].mean()
                result[mask] = avg_val
            
            return result
        except ImportError:
            print("  [警告] scikit-image未安装，跳过超像素方法")
            return img
    
    def method8_multiscale_fusion(self, img):
        """方法8: 多尺度融合（金字塔融合）"""
        # 构建高斯金字塔
        pyramid = [img]
        for i in range(3):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        
        # 上采样并融合
        result = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            result = cv2.pyrUp(result)
            # 确保尺寸匹配
            if result.shape != pyramid[i].shape:
                result = cv2.resize(result, (pyramid[i].shape[1], pyramid[i].shape[0]))
            result = 0.5 * result + 0.5 * pyramid[i]
        
        return result
    
    def method9_adaptive_histogram_equalization(self, img):
        """方法9: 自适应直方图均衡化（局部对比度增强）"""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # CLAHE with stronger settings
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(img_uint8)
        
        return enhanced.astype(np.float32) / 255.0
    
    def method10_combined_best(self, attn_map, class_feature):
        """方法10: 组合最佳方法"""
        # Step 1: 类别特征强引导
        guided = self.method4_class_feature_guided(attn_map, class_feature)
        
        # Step 2: 极端百分位裁剪
        percentile = self.method1_extreme_percentile(guided)
        
        # Step 3: 边缘感知增强
        edge_aware = self.method2_edge_aware(percentile)
        
        # Step 4: Top-K遮罩
        top_k = self.method3_top_k_masking(edge_aware, top_k_ratio=0.4)
        
        # Step 5: 自适应直方图均衡
        final = self.method9_adaptive_histogram_equalization(top_k)
        
        return final
    
    def save_heatmap(self, img, save_path, overlay_original=True):
        """保存热力图"""
        # 确保在[0,1]范围
        img = np.clip(img, 0, 1)
        
        # 转换为uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 应用colormap
        colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
        
        # 调整大小到512x512
        if colored.shape[:2] != (512, 512):
            colored = cv2.resize(colored, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        
        # 可选：叠加到原图
        if overlay_original and self.original_img is not None:
            original_resized = cv2.resize(self.original_img, (512, 512))
            original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
            
            # 混合
            alpha = 0.5
            overlayed = cv2.addWeighted(original_bgr, alpha, colored, 1-alpha, 0)
            
            # 保存两个版本
            cv2.imwrite(str(save_path), colored)
            overlay_path = save_path.parent / (save_path.stem + "_overlay.png")
            cv2.imwrite(str(overlay_path), overlayed)
        else:
            cv2.imwrite(str(save_path), colored)
    
    def enhance_all_methods(self, layer_idx=46, head_idx=0, predicted_class=0):
        """使用所有方法增强"""
        print(f"\n[处理] Layer {layer_idx}, Head {head_idx}")
        
        # 加载注意力图和类别特征
        attn_map = self.load_attention_map(layer_idx, head_idx)
        class_feature = self.load_class_feature(predicted_class)
        
        if attn_map is None:
            print(f"  [错误] 注意力图不存在")
            return
        
        if class_feature is None:
            print(f"  [警告] 类别特征图不存在，某些方法会退化")
        
        # 应用所有方法
        methods = {
            'extreme_percentile': lambda: self.method1_extreme_percentile(attn_map),
            'edge_aware': lambda: self.method2_edge_aware(attn_map),
            'top_k_masking': lambda: self.method3_top_k_masking(attn_map, 0.3),
            'class_feature_guided': lambda: self.method4_class_feature_guided(attn_map, class_feature),
            'watershed': lambda: self.method5_watershed_segmentation(attn_map),
            'gradcam_style': lambda: self.method6_gradcam_style(attn_map),
            'superpixel': lambda: self.method7_superpixel_attention(attn_map),
            'multiscale_fusion': lambda: self.method8_multiscale_fusion(attn_map),
            'adaptive_hist_eq': lambda: self.method9_adaptive_histogram_equalization(attn_map),
            'combined_best': lambda: self.method10_combined_best(attn_map, class_feature),
        }
        
        results = {}
        for method_name, method_func in methods.items():
            try:
                print(f"  - 应用 {method_name}...")
                enhanced = method_func()
                results[method_name] = enhanced
                
                # 保存
                save_path = self.output_dir / f"layer{layer_idx:02d}_head{head_idx}_{method_name}.png"
                self.save_heatmap(enhanced, save_path, overlay_original=True)
                
            except Exception as e:
                print(f"    [错误] {method_name}: {e}")
        
        print(f"  [完成] 生成 {len(results)} 个增强版本")
        
        # 创建对比网格
        self.create_comparison_grid(layer_idx, head_idx, results)
        
        return results
    
    def create_comparison_grid(self, layer_idx, head_idx, results):
        """创建对比网格"""
        print(f"  - 创建对比网格...")
        
        # 计算网格大小
        num_methods = len(results)
        cols = 4
        rows = (num_methods + cols - 1) // cols
        
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
        
        # 添加增强结果
        start_row = 1 if self.original_img is not None else 0
        
        for idx, (method_name, enhanced) in enumerate(results.items()):
            row = start_row + idx // cols
            col = idx % cols
            
            # 加载保存的图像
            img_path = self.output_dir / f"layer{layer_idx:02d}_head{head_idx}_{method_name}.png"
            if img_path.exists():
                img = Image.open(img_path)
                if img.size != (img_size, img_size):
                    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                
                x = col * img_size
                y = row * img_size
                grid_image.paste(img, (x, y))
        
        # 添加标签
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 原图标签
        if self.original_img is not None:
            draw.text((10, 10), "Original Image", fill='white', font=font, 
                     stroke_width=2, stroke_fill='black')
        
        # 方法标签
        for idx, method_name in enumerate(results.keys()):
            row = start_row + idx // cols
            col = idx % cols
            x = col * img_size + 10
            y = row * img_size + 10
            
            # 简化方法名
            display_name = method_name.replace('_', ' ').title()
            draw.text((x, y), display_name, fill='white', font=font,
                     stroke_width=2, stroke_fill='black')
        
        # 保存网格
        grid_path = self.output_dir / f"layer{layer_idx:02d}_head{head_idx}_contour_enhancement_grid.png"
        grid_image.save(grid_path)
        
        print(f"  [保存] 对比网格: {grid_path}")

def main():
    parser = argparse.ArgumentParser(description='超强注意力轮廓增强')
    parser.add_argument('--attention_dir', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final\0_n01443537_24scan_attn_map',
                       help='注意力图目录')
    parser.add_argument('--class_feature_dir', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final\0_n01443537_24scan_RGB\50',
                       help='类别特征图目录')
    parser.add_argument('--original_img', type=str,
                       default=r'D:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG',
                       help='原始图像路径')
    parser.add_argument('--output_dir', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\contour_enhanced',
                       help='输出目录')
    parser.add_argument('--layer_idx', type=int, default=46, help='层索引')
    parser.add_argument('--head_idx', type=int, default=0, help='注意力头索引')
    parser.add_argument('--predicted_class', type=int, default=0, help='预测类别（鱼=0）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("超强注意力轮廓增强 - 显示鱼的清晰轮廓")
    print("=" * 80)
    
    enhancer = ContourEnhancer(
        args.attention_dir,
        args.class_feature_dir,
        args.original_img,
        args.output_dir
    )
    
    # 增强指定层
    enhancer.enhance_all_methods(
        layer_idx=args.layer_idx,
        head_idx=args.head_idx,
        predicted_class=args.predicted_class
    )
    
    print("\n" + "=" * 80)
    print("[完成] 所有增强结果已保存到:", args.output_dir)
    print("=" * 80)

if __name__ == '__main__':
    main()

