"""
测试NNS问题
"""
import sys
import torch
import timm
from model.vit_final import VisionTransformer

# 加载timm模型
print("加载timm模型...")
timm_model = timm.create_model('vit_base_patch16_224', pretrained=True)
timm_model.eval()

print("\n1. timm模型的children:")
children_list = list(timm_model.children())
print(f"   children数量: {len(children_list)}")
for i, child in enumerate(children_list):
    print(f"   {i}. {type(child).__name__}")

print("\n2. 检查是否有PatchEmbed:")
has_patch_embed = False
for child in timm_model.children():
    if type(child).__name__ == 'PatchEmbed':
        has_patch_embed = True
        print(f"   ✓ 找到PatchEmbed")
        break

if not has_patch_embed:
    print(f"   ✗ 没有找到PatchEmbed作为直接child")
    print(f"   检查named_children:")
    for name, child in timm_model.named_children():
        print(f"     - {name}: {type(child).__name__}")
        if name == 'patch_embed':
            print(f"       ✓ 找到patch_embed")

print("\n3. 检查是否有blocks (ModuleList):")
has_blocks = False
for child in timm_model.children():
    if type(child).__name__ in ['ModuleList', 'Sequential']:
        has_blocks = True
        print(f"   ✓ 找到{type(child).__name__}，长度: {len(child)}")
        break

if not has_blocks:
    print(f"   ✗ 没有找到ModuleList作为直接child")
    if hasattr(timm_model, 'blocks'):
        print(f"   但是model.blocks存在，类型: {type(timm_model.blocks).__name__}")

print("\n结论:")
print("  timm模型的结构与NNS期望的不同！")
print("  NNS期望: PatchEmbed -> ModuleList(Blocks) -> Linear")
print("  timm实际: 使用named_children访问patch_embed和blocks")

