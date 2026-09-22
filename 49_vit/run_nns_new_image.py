"""对新图片进行NNS扫描 - n02085620_1.JPEG"""
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
    print(f'Device: {device}')
    
    # 模型参数
    depth = 12
    num_heads = 3
    patch_size = 4
    classes = 10
    
    # 新图像路径
    new_image = r"d:\project\vit_114\NNS_model_and_dataset\dataset\tiny-imagenet-5\scanimg\n02085620\n02085620_1.JPEG"
    
    # 模型路径
    model_path = r"d:\project\vit_114\NNS_model_and_dataset\models\vit_3channel_mutlihead\tiny200ILSVRC_VisionTransformer_32imgsize_10classes_45threshold_3heads_12depth_4patch_size_VisionTransformer_200_final.pth"
    
    print(f"Model path: {model_path}")
    print(f"New image path: {new_image}")
    
    # 检查文件存在
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    if not os.path.exists(new_image):
        print(f"ERROR: Image file not found: {new_image}")
        return
    
    # 创建模型
    print(f"Creating model: depth={depth}, num_heads={num_heads}, patch_size={patch_size}")
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
    print("Loading model weights...")
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
    print("Loading new image...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    
    image = Image.open(new_image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 获取预测
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[pred].item()
        print(f"Prediction: class {pred} (confidence: {confidence:.2%})")
    
    # 获取模型层数
    child = list(model.children())
    LN = len(child)
    print(f"Model layers: {LN}")
    
    # 创建NNS对象
    print(f"Starting NNS scan for new image (depth={depth} layers)...")
    name = "n02085620_1_new"
    model_name = "VisionTransformer_200_final_new"
    
    nns = NNS_NONET(
        model, image_tensor[0], name, 0, device, 
        model_name, "VisionTransformer", 'tiny200ILSVRC', 
        patch_size, num_heads, depth, params
    )
    
    print(f"Output directory: {nns.rootdir}")
    
    # 确保目录存在
    os.makedirs(nns.rootdir, exist_ok=True)
    os.makedirs(nns.scan_attn_map, exist_ok=True)
    
    # 运行扫描
    print("Running NNS scan for new image...")
    nns.get_scanpic(50)  # 使用50确保所有层都被处理
    
    print("SUCCESS: NNS scan for new image completed!")
    print(f"Results saved to: {nns.rootdir}")
    
    # 检查生成的结果
    print("\nChecking generated results...")
    
    # 检查scan_attn_map目录
    if os.path.exists(nns.scan_attn_map):
        layer_dirs = []
        for item in os.listdir(nns.scan_attn_map):
            item_path = os.path.join(nns.scan_attn_map, item)
            if os.path.isdir(item_path) and item.isdigit():
                layer_dirs.append(int(item))
        layer_dirs.sort()
        
        if layer_dirs:
            print(f"SUCCESS: Found {len(layer_dirs)} attention layers: {layer_dirs}")
        else:
            print("WARNING: No attention layer directories found")
    else:
        print("WARNING: scan_attn_map directory not created")
    
    return nns.rootdir

if __name__ == "__main__":
    result_dir = main()
    print(f"\n{'='*80}")
    print("NNS SCAN FOR NEW IMAGE COMPLETED!")
    print(f"{'='*80}")
    print(f"Result directory: {result_dir}")
    print(f"{'='*80}")
