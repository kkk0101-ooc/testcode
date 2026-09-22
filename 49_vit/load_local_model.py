"""
从本地加载ViT模型
"""

import torch
from pathlib import Path

def load_vit_model(device='cuda'):
    """
    从本地加载ViT模型
    
    Args:
        device: 'cuda' 或 'cpu'
    
    Returns:
        model: 加载的模型
        device: 实际使用的设备
    """
    
    # 模型路径
    model_dir = Path(__file__).parent / 'model'
    model_path = model_dir / 'vit_base_patch16_224_full.pth'
    
    # 检查文件是否存在
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行: python download_model_to_local.py")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 选择设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'
    
    device = torch.device(device)
    
    # 加载模型
    print(f"从本地加载模型: {model_path.name}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    
    print(f"  模型已加载到: {device}")
    
    return model, device

if __name__ == "__main__":
    # 测试加载
    model, device = load_vit_model()
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("模型加载成功!")
