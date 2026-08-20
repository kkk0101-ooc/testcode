"""
调试timm模型结构
"""
import torch
import timm

# 加载模型
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

print("="*70)
print("ViT-base-16 模型结构")
print("="*70)

print("\n1. 顶层children:")
for i, (name, child) in enumerate(model.named_children()):
    print(f"  {i}. {name}: {type(child).__name__}")

print("\n2. 所有模块:")
for name, module in model.named_modules():
    if len(name.split('.')) <= 2:  # 只显示前两层
        print(f"  {name}: {type(module).__name__}")

print("\n3. Blocks结构:")
if hasattr(model, 'blocks'):
    print(f"  blocks类型: {type(model.blocks).__name__}")
    print(f"  blocks长度: {len(model.blocks)}")
    if len(model.blocks) > 0:
        print(f"  第一个block: {type(model.blocks[0]).__name__}")
        for name, child in model.blocks[0].named_children():
            print(f"    - {name}: {type(child).__name__}")

print("\n4. 测试forward:")
x = torch.randn(1, 3, 224, 224)
try:
    with torch.no_grad():
        out = model(x)
    print(f"  输出shape: {out.shape}")
    print(f"  ✓ Forward成功")
except Exception as e:
    print(f"  ✗ Forward失败: {e}")

print("\n" + "="*70)

