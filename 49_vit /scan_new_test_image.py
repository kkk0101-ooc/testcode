"""简化的新图片NNS扫描脚本"""
import sys
import os
sys.path.append('.')

import torch
from torchvision import transforms
from PIL import Image
from NNS.Scannerpic1107_cls_nonet import NNS_NONET

def main():
    # 图片路径
    image_path = r"D:\project\vit_114\20251114142550.png"
    model_path = r"d:\project\vit_114\NNS_model_and_dataset\models\vit_3channel_mutlihead\tiny200ILSVRC_VisionTransformer_32imgsize_10classes_45threshold_3heads_12depth_4patch_size_VisionTransformer_200_final.pth"
    
    print(f"Processing image: {image_path}")
    print(f"Using model: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return
    
    # 加载图像
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型 - 处理可能的字典格式
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # 需要重建模型架构
            print("Model is state_dict format, need to reconstruct...")
            from models.vit import VisionTransformer
            model = VisionTransformer(
                img_size=32,
                patch_size=4,
                in_chans=3,
                num_classes=10,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4.0
            )
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("Unknown checkpoint format")
            return
    else:
        model = checkpoint
    
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # 创建NNS扫描器
    scanner = NNS_NONET(
        network=model,
        imageTensor=image_tensor,
        img_name="new_test_image_20251114",
        id=0,
        device=device,
        model_name="VisionTransformer_200_final",
        model_type="tiny200ILSVRC", 
        dataset="114VisionTransformer_tiny200ILSVRC",
        patch_size=4,
        num_heads=3,
        depth=12,
        params=None
    )
    
    print("NNS Scanner created successfully")
    print(f"Output directory: {scanner.rootdir}")
    
    # 运行扫描
    print("Starting NNS scan...")
    try:
        with torch.no_grad():
            # 这里直接调用扫描方法，而不是scan_cls_nonet
            scanner.scan_cls_nonet(model, image_tensor, 0, "new_test_image_20251114")
        print("NNS scan completed successfully!")
        print(f"Results saved to: {scanner.rootdir}")
    except Exception as e:
        print(f"ERROR during scanning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
