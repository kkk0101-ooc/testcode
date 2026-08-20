import timm

print("加载timm模型...")
model = timm.create_model('vit_base_patch16_224', pretrained=True)

print("\n=== Children ===")
children_list = list(model.children())
print(f"数量: {len(children_list)}")
for i, c in enumerate(children_list):
    print(f"  {i}. {type(c).__name__}")

print("\n=== Named Children ===")
for n, c in model.named_children():
    print(f"  {n}: {type(c).__name__}")
    if n == 'blocks':
        print(f"    -> blocks数量: {len(c)}")
        if len(c) > 0:
            print(f"    -> 第一个block类型: {type(c[0]).__name__}")

print("\n=== 结论 ===")
print("timm模型使用named_children访问:")
print("  - patch_embed: PatchEmbed")
print("  - blocks: Sequential (包含12个Block)")
print("  - norm: LayerNorm")
print("  - head: Linear")

