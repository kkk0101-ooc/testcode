"""
为ImageNet图片运行ViT-Base-16的NNS扫描
使用timm预训练模型
"""

import torch
import timm
from PIL import Image
from pathlib import Path
from torchvision import transforms
import sys

# 添加NNS路径
sys.path.insert(0, str(Path(__file__).parent))
from NNS.Scannerpic1107_cls_nonet import NNS_NONET

def scan_imagenet_images(image_paths, device='cuda'):
    """
    扫描ImageNet图片
    
    Args:
        image_paths: 图片路径列表
        device: 'cuda' or 'cpu'
    """
    
    print("="*80)
    print("ImageNet ViT-Base-16 NNS扫描")
    print("="*80)
    print()
    
    # 设备设置
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print()
    
    # 加载ViT-Base-16模型
    print("加载ViT-Base-16模型...")
    model_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
    
    try:
        model = timm.create_model(model_name, pretrained=True)
        model.to(device)
        model.eval()
        print(f"  OK 模型已加载: {model_name}")
    except Exception as e:
        print(f"  错误: {e}")
        print("  尝试使用基础版本...")
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.to(device)
        model.eval()
        print(f"  OK 使用基础ViT-Base模型")
    
    print()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.230])
    ])
    
    # 处理每张图片
    for idx, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        
        print(f"处理图片 {idx+1}/{len(image_paths)}: {img_path.name}")
        
        if not img_path.exists():
            print(f"  错误: 文件不存在")
            continue
        
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            print(f"  图片大小: {image.size} -> 张量: {image_tensor.shape}")
        except Exception as e:
            print(f"  错误: 无法加载图片 - {e}")
            continue
        
        # 提取图片ID（从文件名）
        # 例如: ILSVRC2012_val_00000441.JPEG -> 00000441
        img_name = img_path.stem  # ILSVRC2012_val_00000441
        parts = img_name.split('_')
        img_id = parts[-1] if len(parts) > 0 else "unknown"
        
        # 先运行一次推理获取预测
        print(f"  运行推理...")
        try:
            with torch.no_grad():
                output = model(image_tensor)
                pred = output.argmax(dim=1).item()
                print(f"  预测类别: {pred}")
        except Exception as e:
            print(f"  警告: 推理失败 - {e}")
            pred = 0
        
        # 创建扫描器
        print(f"  创建NNS扫描器...")
        full_img_name = f"{idx}_{img_name}_pred{pred}"
        
        # 提取模型参数
        params = {
            'cls_token': model.cls_token,
            'pos_embed': model.pos_embed
        }
        
        try:
            scanner = NNS_NONET(
                network=model,
                imageTensor=image_tensor,
                img_name=full_img_name,
                id=idx,
                device=device,
                model_name="vit_base_patch16_224_pretrained",
                model_type="ImageNet",
                dataset="114VisionTransformer_ImageNet",
                patch_size=16,
                num_heads=12,
                depth=12,
                params=params
            )
            
            print(f"  输出目录: {scanner.rootdir}")
            
            # 运行扫描
            print(f"  开始扫描...")
            with torch.no_grad():
                scanner.scan_cls_nonet(model, image_tensor, idx, full_img_name)
            
            print(f"  OK 扫描完成!")
            print()
            
        except Exception as e:
            print(f"  错误: 扫描失败 - {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("="*80)
    print("全部完成!")
    print("="*80)

def main():
    """主函数"""
    
    # 要扫描的两张图片
    images = [
        "datas/ILSVRC2012_val/ILSVRC2012_val_00000441.JPEG",
        "datas/ILSVRC2012_val/ILSVRC2012_val_00006100.JPEG"
    ]
    
    # 验证图片存在
    print("验证图片...")
    all_exist = True
    for img in images:
        p = Path(img)
        if p.exists():
            print(f"  OK {p.name}")
        else:
            print(f"  X 不存在: {img}")
            all_exist = False
    
    if not all_exist:
        print("\n错误: 部分图片不存在")
        return
    
    print()
    print("开始扫描...")
    print("注意: 每张图片约需30-60秒")
    print()
    
    scan_imagenet_images(images)

if __name__ == "__main__":
    main()
