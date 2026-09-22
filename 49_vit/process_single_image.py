"""
处理单张图像的NNS可视化
不影响训练，独立运行
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import argparse
from pathlib import Path
from NNS.Scannerpic1107_cls_nonet import NNS_NONET
from model.vit_final import VisionTransformer

def process_single_image(img_path, model_path, output_dir, device='cuda:0'):
    """处理单张图像"""
    print(f"\n{'='*80}")
    print(f"处理单张图像 NNS 可视化")
    print(f"{'='*80}")
    
    # 检查图像是否存在
    if not os.path.exists(img_path):
        print(f"[错误] 图像不存在: {img_path}")
        return False
    
    print(f"\n[输入]")
    print(f"  图像: {img_path}")
    print(f"  模型: {model_path}")
    print(f"  输出: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"\n[加载模型]")
    model = VisionTransformer(
        image_size=64,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=10
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"  ✓ 模型加载完成 (depth=12)")
    
    # 创建Scanner
    print(f"\n[初始化 NNS Scanner]")
    scanner = NNS_NONET(
        model=model,
        model_name='depth12_single',
        save_root=output_dir,
        device=device,
        visual_attention=True,
        visual_value=False
    )
    
    print(f"  ✓ Scanner初始化完成")
    
    # 处理图像
    print(f"\n[处理图像]")
    try:
        scanner.ScanOnePic(img_path)
        print(f"\n  ✓ 处理完成！")
        return True
    except Exception as e:
        print(f"\n  × 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='处理单张图像的NNS可视化')
    parser.add_argument('--img_path', type=str,
                       default=r'D:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\scanimg\n01443537\n01443537_102.JPEG',
                       help='输入图像路径')
    parser.add_argument('--model_path', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\checkpoint\vit_depth12_epoch49_final.pth',
                       help='模型路径')
    parser.add_argument('--output_dir', type=str,
                       default=r'D:\project\vit_114\vitnns\49_vit\scan_results\scan_vit\114VisionTransformer_tiny200ILSVRC\depth12_single',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    
    args = parser.parse_args()
    
    success = process_single_image(
        args.img_path,
        args.model_path,
        args.output_dir,
        args.device
    )
    
    if success:
        print(f"\n{'='*80}")
        print(f"✅ 完成！结果保存在: {args.output_dir}")
        print(f"{'='*80}")
        
        # 提示下一步
        img_name = Path(args.img_path).stem
        print(f"\n下一步:")
        print(f"  1. 查看注意力图:")
        print(f"     {args.output_dir}\\*{img_name}*scan_attn_map\\")
        print(f"  2. 运行轮廓增强:")
        print(f"     python enhance_attention_contour.py --attention_dir <attn_dir> --original_img {args.img_path}")
    else:
        print(f"\n{'='*80}")
        print(f"❌ 处理失败")
        print(f"{'='*80}")

if __name__ == '__main__':
    main()
