"""直接运行NNS扫描 - 使用刚训练好的depth=12模型"""
import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.vit_final import VisionTransformer
from NNS.Scannerpic1107_cls_nonet import NNS_NONET

def main():
    # 配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'设备: {device}')
    
    # 模型参数
    depth = 12
    num_heads = 3
    patch_size = 4
    classes = 10
    
    # 鱼图像路径
    fish_image = r"d:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\test111\n01443537\n01443537_24.JPEG"
    
    # 模型路径（使用刚训练好的final模型）
    model_path = r"d:\project\vit_114\NNS_model_and_dataset\models\vit_3channel_mutlihead\tiny200ILSVRC_VisionTransformer_32imgsize_10classes_45threshold_3heads_12depth_4patch_size_VisionTransformer_200_final.pth"
    
    print(f"模型路径: {model_path}")
    print(f"图像路径: {fish_image}")
    
    # 检查文件存在
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    if not os.path.exists(fish_image):
        print(f"ERROR: Image file not found: {fish_image}")
        return
    
    # 创建模型
    print(f"创建模型: depth={depth}, num_heads={num_heads}, patch_size={patch_size}")
    model = VisionTransformer(
        img_size=[32], 
        patch_size=patch_size, 
        num_classes=classes, 
        embed_dim=192, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_ratio=4,
        qkv_bias=False
    )
    
    # 加载权重
    print("加载模型权重...")
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # 提取参数
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.clone()
    
    # 加载和预处理图像
    print("加载图像...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    
    image = Image.open(fish_image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 获取预测
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[pred].item()
        print(f"预测类别: {pred} (置信度: {confidence:.2%})")
    
    # 获取模型层数
    child = list(model.children())
    LN = len(child)
    print(f"模型层数: {LN}")
    
    # 创建NNS对象
    print(f"开始NNS扫描（depth={depth}层）...")
    name = "n01443537_24"
    model_name = "VisionTransformer_200_final"
    
    nns = NNS_NONET(
        model, image_tensor[0], name, 0, device, 
        model_name, "VisionTransformer", 'tiny200ILSVRC', 
        patch_size, num_heads, depth, params
    )
    
    # 运行扫描
    print(f"输出目录: {nns.rootdir}")
    nns.get_scanpic(LN)
    print("SUCCESS: NNS scan completed!")
    print(f"Results saved to: {nns.rootdir}")
    
    return nns.rootdir

if __name__ == "__main__":
    result_dir = main()
    print(f"\n{'='*60}")
    print("SUCCESS: Scan completed! Result directory:")
    print(f"{result_dir}")
    print(f"{'='*60}")
