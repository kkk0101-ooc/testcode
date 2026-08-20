"""
更新脚本，使用本地模型文件
"""

from pathlib import Path
import shutil

def create_model_loader():
    """创建模型加载器模块"""
    
    loader_path = Path(__file__).parent / 'load_local_model.py'
    
    code = '''"""
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
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    print(f"  模型已加载到: {device}")
    
    return model, device

if __name__ == "__main__":
    # 测试加载
    model, device = load_vit_model()
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("模型加载成功!")
'''
    
    with open(loader_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"已创建: {loader_path}")
    return loader_path

def update_generate_all_ppt_figures():
    """更新 generate_all_ppt_figures.py"""
    
    script_path = Path(__file__).parent / 'generate_all_ppt_figures.py'
    
    if not script_path.exists():
        print(f"警告: 文件不存在: {script_path}")
        return
    
    # 备份原文件
    backup_path = script_path.with_suffix('.py.backup')
    shutil.copy2(script_path, backup_path)
    print(f"已备份: {backup_path.name}")
    
    # 读取原文件
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换模型加载部分
    old_code = """    # 加载模型
        print("加载ViT-Base-16模型...")
        model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        model.to(device)
        model.eval()
        print("  OK\\n")"""
    
    new_code = """    # 加载模型（从本地）
        print("加载ViT-Base-16模型（本地文件）...")
        from load_local_model import load_vit_model
        model, device = load_vit_model('cuda' if torch.cuda.is_available() else 'cpu')
        print("  OK\\n")"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"已更新: {script_path.name}")
    else:
        print(f"跳过: {script_path.name} (未找到匹配代码)")

def update_run_new_images_experiments():
    """更新 run_new_images_experiments.py"""
    
    script_path = Path(__file__).parent / 'run_new_images_experiments.py'
    
    if not script_path.exists():
        print(f"警告: 文件不存在: {script_path}")
        return
    
    # 备份原文件
    backup_path = script_path.with_suffix('.py.backup')
    shutil.copy2(script_path, backup_path)
    print(f"已备份: {backup_path.name}")
    
    # 读取原文件
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换模型加载部分
    old_code = """        # 加载模型
        print("加载ViT-Base-16模型...")
        self.model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        print("  模型加载完成\\n")"""
    
    new_code = """        # 加载模型（从本地）
        print("加载ViT-Base-16模型（本地文件）...")
        from load_local_model import load_vit_model
        self.model, self.device = load_vit_model('cuda' if torch.cuda.is_available() else 'cpu')
        print("  模型加载完成\\n")"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"已更新: {script_path.name}")
    else:
        print(f"跳过: {script_path.name} (未找到匹配代码)")

def main():
    """主函数"""
    
    print("="*60)
    print("更新脚本使用本地模型")
    print("="*60)
    print()
    
    # 检查模型文件是否存在
    model_dir = Path(__file__).parent / 'model'
    model_file = model_dir / 'vit_base_patch16_224_full.pth'
    
    if not model_file.exists():
        print("错误: 本地模型文件不存在")
        print(f"请先运行: python download_model_to_local.py")
        print()
        return
    
    print(f"找到本地模型: {model_file}")
    print()
    
    # 创建加载器
    print("步骤1: 创建模型加载器...")
    loader_path = create_model_loader()
    print()
    
    # 更新脚本
    print("步骤2: 更新现有脚本...")
    update_generate_all_ppt_figures()
    update_run_new_images_experiments()
    print()
    
    print("="*60)
    print("完成!")
    print("="*60)
    print()
    print("更新内容:")
    print("  1. 创建了 load_local_model.py - 本地模型加载器")
    print("  2. 更新了 generate_all_ppt_figures.py")
    print("  3. 更新了 run_new_images_experiments.py")
    print()
    print("原文件已备份为 *.py.backup")
    print()
    print("现在可以运行脚本，将使用本地模型文件")

if __name__ == "__main__":
    main()
