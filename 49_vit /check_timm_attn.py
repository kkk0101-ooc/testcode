import timm

m = timm.create_model('vit_base_patch16_224', pretrained=True)
block0 = m.blocks[0]

print("Block 0 children:")
for i, child in enumerate(block0.children()):
    print(f"  {i}. {type(child).__name__}")

attn = list(block0.children())[1]
print(f"\nAttention layer children (共{len(list(attn.children()))}个):")
for i, child in enumerate(attn.children()):
    print(f"  {i}. {type(child).__name__}")
    if hasattr(child, 'in_features'):
        print(f"     -> in_features={child.in_features}, out_features={child.out_features}")

print("\nAttention layer named_children:")
for name, child in attn.named_children():
    print(f"  {name}: {type(child).__name__}")

