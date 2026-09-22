"""
统一类别特征图（0-9.png）的颜色，使它们使用相同的归一化范围
这样所有类别特征图的颜色就会一致
"""
import os
import cv2
import numpy as np
from PIL import Image

# 配置
class_feature_dir = r"D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_final\0_n01443537_24scan_RGB\50"
output_dir = class_feature_dir  # 覆盖原文件，或创建新目录

print(f"类别特征图目录: {class_feature_dir}")

# 加载所有类别特征图
class_images = {}
class_arrays = {}
all_values = []

if not os.path.exists(class_feature_dir):
    print(f"错误：目录不存在: {class_feature_dir}")
    exit(1)

# 第一步：加载所有图像并收集所有像素值
print("\n步骤1: 加载所有类别特征图...")
for i in range(10):
    img_path = os.path.join(class_feature_dir, f"{i}.png")
    if os.path.exists(img_path):
        # 读取图像（可能是RGB热力图）
        img = cv2.imread(img_path)
        if img is not None:
            # 如果是RGB热力图，需要反向映射到原始值
            # 或者直接使用灰度值
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            class_images[i] = img
            class_arrays[i] = gray
            all_values.extend(gray.flatten())
            print(f"  ✓ 类别 {i}: {img.shape}")
        else:
            print(f"  ✗ 类别 {i}: 无法读取")
    else:
        print(f"  ✗ 类别 {i}: 文件不存在")

if len(class_images) == 0:
    print("错误：没有找到任何类别特征图")
    exit(1)

# 计算全局最小值和最大值
global_min = np.min(all_values)
global_max = np.max(all_values)
global_range = global_max - global_min

print(f"\n步骤2: 计算全局范围...")
print(f"  全局最小值: {global_min}")
print(f"  全局最大值: {global_max}")
print(f"  全局范围: {global_range}")

# 第二步：使用全局范围重新归一化所有图像
print(f"\n步骤3: 使用全局范围重新归一化所有图像...")

# 创建备份目录
backup_dir = os.path.join(class_feature_dir, "backup_original")
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)
    print(f"  创建备份目录: {backup_dir}")

for i in range(10):
    if i in class_arrays:
        # 备份原图
        original_path = os.path.join(class_feature_dir, f"{i}.png")
        backup_path = os.path.join(backup_dir, f"{i}.png")
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(original_path, backup_path)
        
        # 获取当前图像的灰度值
        gray = class_arrays[i]
        
        # 归一化到[0, 1]使用全局范围
        if global_range > 1e-8:
            normalized = (gray.astype(np.float32) - global_min) / global_range
        else:
            normalized = gray.astype(np.float32) / 255.0
        
        # 确保在[0, 1]范围内
        normalized = np.clip(normalized, 0, 1)
        
        # 转换为uint8
        normalized_uint8 = (normalized * 255).astype(np.uint8)
        
        # 应用JET colormap（与原始可视化一致）
        heat_img = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
        
        # 保存
        output_path = os.path.join(class_feature_dir, f"{i}.png")
        cv2.imwrite(output_path, heat_img)
        print(f"  ✓ 类别 {i}: 已重新归一化并保存")

print(f"\n✅ 完成！所有类别特征图已使用统一的归一化范围")
print(f"   原图备份在: {backup_dir}")
print(f"   现在所有类别特征图的颜色应该一致了！")

