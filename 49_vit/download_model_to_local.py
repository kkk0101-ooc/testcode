"""
下载ViT模型并保存到本地model文件夹
"""

import torch
import timm
from pathlib import Path
import shutil
import os

def download_and_save_model():
    """下载模型并保存到本地"""
    
    print("="*60)
    print("下载ViT模型到本地文件夹")
    print("="*60)
    print()
    
    # 创建model目录
    model_dir = Path(__file__).parent / 'model'
    model_dir.mkdir(exist_ok=True)
    print(f"模型保存目录: {model_dir}")
    print()
    
    # 模型名称
    model_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
    
    print(f"正在下载模型: {model_name}")
    print("首次下载需要时间，请耐心等待...")
    print("模型大小: 约330 MB")
    print()
    
    try:
        # 加载模型（会自动下载到缓存）
        print("步骤1: 通过timm加载模型...")
        model = timm.create_model(model_name, pretrained=True)
        print("  模型加载成功")
        print()
        
        # 保存模型权重
        print("步骤2: 保存模型权重...")
        model_save_path = model_dir / 'vit_base_patch16_224_weights.pth'
        torch.save(model.state_dict(), model_save_path)
        size_mb = model_save_path.stat().st_size / (1024 * 1024)
        print(f"  权重已保存: {model_save_path}")
        print(f"  文件大小: {size_mb:.2f} MB")
        print()
        
        # 保存完整模型（包含结构）
        print("步骤3: 保存完整模型...")
        full_model_path = model_dir / 'vit_base_patch16_224_full.pth'
        torch.save(model, full_model_path)
        size_mb = full_model_path.stat().st_size / (1024 * 1024)
        print(f"  完整模型已保存: {full_model_path}")
        print(f"  文件大小: {size_mb:.2f} MB")
        print()
        
        # 保存模型配置
        print("步骤4: 保存模型配置...")
        config_path = model_dir / 'model_config.txt'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(f"模型名称: {model_name}\n")
            f.write(f"模型架构: ViT-Base-16\n")
            f.write(f"参数量: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"输入大小: 224x224\n")
            f.write(f"Patch size: 16x16\n")
            f.write(f"层数: 12\n")
            f.write(f"注意力头数: 12\n")
            f.write(f"隐藏维度: 768\n")
            f.write(f"\n文件说明:\n")
            f.write(f"- vit_base_patch16_224_weights.pth: 仅权重文件\n")
            f.write(f"- vit_base_patch16_224_full.pth: 完整模型（推荐）\n")
        print(f"  配置已保存: {config_path}")
        print()
        
        # 创建加载示例脚本
        print("步骤5: 创建加载示例...")
        example_path = model_dir / 'load_model_example.py'
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write('''"""
从本地加载ViT模型的示例代码
"""

import torch
from pathlib import Path

def load_model_from_local(use_gpu=True):
    """从本地加载模型"""
    
    # 模型路径
    model_dir = Path(__file__).parent
    model_path = model_dir / 'vit_base_patch16_224_full.pth'
    
    # 检查文件是否存在
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 选择设备
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print(f"从本地加载模型: {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    print(f"模型已加载到: {device}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, device

if __name__ == "__main__":
    model, device = load_model_from_local()
    print("模型加载成功!")
''')
        print(f"  示例脚本已保存: {example_path}")
        print()
        
        print("="*60)
        print("完成!")
        print("="*60)
        print()
        print("生成的文件:")
        print(f"  1. vit_base_patch16_224_weights.pth - 权重文件")
        print(f"  2. vit_base_patch16_224_full.pth - 完整模型（推荐使用）")
        print(f"  3. model_config.txt - 模型配置说明")
        print(f"  4. load_model_example.py - 加载示例代码")
        print()
        print(f"模型目录: {model_dir}")
        print()
        print("使用方法:")
        print("  运行 update_scripts_use_local_model.py 更新现有脚本")
        
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        print()
        print("可能的原因:")
        print("  1. 网络连接问题")
        print("  2. 磁盘空间不足")
        print("  3. 权限问题")
        print()
        print("解决方案:")
        print("  1. 检查网络连接")
        print("  2. 使用镜像源: export HF_ENDPOINT=https://hf-mirror.com")
        print("  3. 确保有足够的磁盘空间（至少1GB）")
        
        return False

if __name__ == "__main__":
    success = download_and_save_model()
    
    if success:
        print("\n下一步: 运行 update_scripts_use_local_model.py 更新脚本")
    else:
        print("\n下载失败，请检查错误信息并重试")
