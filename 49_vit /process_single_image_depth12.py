"""
处理单个图像文件用于NNS可视化 - 支持depth=12
"""
import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.vit_final import VisionTransformer
from NNS.Scannerpic1107_cls_nonet import NNS_NONET

def find_latest_model(base_path, depth, num_heads, patch_size, classes, threshold):
    """查找最新训练的模型"""
    # 构建模型目录模式
    model_dir_pattern = f'tiny200ILSVRC_VisionTransformer_32imgsize_{classes}classes_{threshold}threshold_{num_heads}heads_{depth}depth_{patch_size}patch_size_'
    
    modeldir = os.path.join(base_path, 'models', 'vit_3channel_mutlihead', model_dir_pattern)
    
    if not os.path.exists(modeldir):
        raise FileNotFoundError(f"模型目录不存在: {modeldir}")
    
    # 查找所有.pth文件
    model_files = []
    for f in os.listdir(modeldir):
        if f.endswith('.pth'):
            full_path = os.path.join(modeldir, f)
            model_files.append((os.path.getmtime(full_path), f, full_path))
    
    if not model_files:
        raise FileNotFoundError(f"在 {modeldir} 中找不到模型文件")
    
    # 按修改时间排序，返回最新的
    model_files.sort(reverse=True)
    return model_files[0][2], model_files[0][1]

def process_single_image(image_path, args):
    """处理单个图像文件"""
    
    # 设置设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f'设备: {device}')
    
    # 加载模型
    print(f"加载模型: {args.model}")
    print(f"模型配置: depth={args.depth}, num_heads={args.num_heads}, patch_size={args.patch_size}")
    
    model = VisionTransformer(
        img_size=[32], 
        patch_size=args.patch_size, 
        num_classes=args.classes, 
        embed_dim=192, 
        depth=args.depth, 
        num_heads=args.num_heads, 
        mlp_ratio=4,
        qkv_bias=False
    )
    
    # 查找模型权重
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    if os.path.exists(os.path.join(project_root, 'NNS_model_and_dataset')):
        base_path = os.path.join(project_root, 'NNS_model_and_dataset')
    elif os.path.exists('/nas/NNS_model_and_dataset'):
        base_path = '/nas/NNS_model_and_dataset'
    else:
        base_path = os.path.join(project_root, 'NNS_model_and_dataset')
    
    try:
        model_path, model_filename = find_latest_model(
            base_path, args.depth, args.num_heads, 
            args.patch_size, args.classes, args.threshold
        )
        model_name = ".".join(model_filename.split('.')[:-1])
        print(f"使用模型: {model_name}")
        print(f"模型路径: {model_path}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n可用的模型目录:")
        models_base = os.path.join(base_path, 'models', 'vit_3channel_mutlihead')
        if os.path.exists(models_base):
            for item in os.listdir(models_base):
                if 'tiny200ILSVRC' in item:
                    print(f"  - {item}")
        sys.exit(1)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # 提取参数
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.data.clone()
    
    # 加载和预处理图像
    print(f"\n加载图像: {image_path}")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 获取预测
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[pred].item()
        print(f"预测类别: {pred} (置信度: {confidence:.2%})")
    
    # 提取文件名
    img_basename = os.path.basename(image_path)
    name = os.path.splitext(img_basename)[0]
    
    # 获取模型层数
    child = list(model.children())
    LN = len(child)
    print(f"模型层数: {LN}")
    
    # 创建NNS对象
    print(f"\n开始NNS扫描（depth={args.depth}层）...")
    nns = NNS_NONET(
        model, image_tensor[0], name, 0, device, 
        model_name, args.model, 'tiny200ILSVRC', 
        args.patch_size, args.num_heads, args.depth, params
    )
    
    # 运行扫描
    if not os.path.exists(nns.rootdir):
        print(f"输出目录: {nns.rootdir}")
        nns.get_scanpic(LN)
        print(f"\n✓ 扫描完成，结果保存在: {nns.rootdir}")
    else:
        print(f"结果已存在: {nns.rootdir}")
    
    return nns.rootdir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='处理单个图像文件 - 支持depth=12')
    parser.add_argument('--image_path', type=str, required=True,
                        help='图像文件路径')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备编号')
    parser.add_argument('--model', type=str, default='VisionTransformer',
                        help='模型名称')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch大小')
    parser.add_argument('--num_heads', type=int, default=3,
                        help='注意力头数')
    parser.add_argument('--depth', type=int, default=12,
                        help='模型深度')
    parser.add_argument('--classes', type=int, default=10,
                        help='类别数')
    parser.add_argument('--threshold', type=int, default=45,
                        help='训练阈值（用于查找模型）')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NNS单图像处理 - Depth=12模型")
    print("="*80)
    
    result_dir = process_single_image(args.image_path, args)
    print(f"\n{'='*80}")
    print(f"✅ 完成！结果目录: {result_dir}")
    print(f"{'='*80}")
