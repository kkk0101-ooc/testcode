from load_local_model import load_vit_model

model, device = load_vit_model('cuda')

print("模型结构中的attention相关层:")
for name, module in model.named_modules():
    if 'attn' in name.lower():
        print(f"  {name}: {module.__class__.__name__}")
