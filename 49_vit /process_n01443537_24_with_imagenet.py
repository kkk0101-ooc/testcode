"""
使用ImageNet + ViT-base-16处理n01443537_24.JPEG图像
方便对比不同模型在同一张图上的效果
"""
import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import timm

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.vit_final import VisionTransformer
from NNS.Scannerpic1107_cls_nonet import NNS_NONET

# 配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_path = r"datas\tiny-imagenet-5-scanner\n01443537\n01443537_24.JPEG"
nns_resize = 64

print("="*70)
print("使用ImageNet + ViT-base-16处理n01443537_24.JPEG")
print("="*70)

# 加载timm模型
print("\n1. 加载ViT-base-16预训练模型...")
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
model.eval()
model = model.to(device)
print("   ✓ 模型已加载")

# 提取参数
print("\n2. 提取模型参数...")
params = {}
state_dict = model.state_dict()

if 'cls_token' in state_dict:
    params['cls_token'] = state_dict['cls_token'].clone()
if 'pos_embed' in state_dict:
    params['pos_embed'] = state_dict['pos_embed'].clone()

for name, param in model.named_parameters():
    params[name] = param.data.clone()
print("   ✓ 参数已提取")

# 加载图像
print(f"\n3. 加载图像: {image_path}")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)
print(f"   ✓ 图像已加载: {image_tensor.shape}")

# 获取预测
print("\n4. 获取模型预测...")
with torch.no_grad():
    output = model(image_tensor)
    pred = torch.argmax(output, dim=1).item()
    print(f"   预测类别: {pred}")

# 降采样（节省内存）
print(f"\n5. 降采样到{nns_resize}×{nns_resize}以节省内存...")
image_for_nns = image_tensor[0]
if image_for_nns.shape[-1] > nns_resize:
    import torch.nn.functional as F
    image_for_nns = F.interpolate(
        image_for_nns.unsqueeze(0),
        size=(nns_resize, nns_resize),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    print(f"   ✓ 已降采样: {image_for_nns.shape}")

# 提取文件名
img_basename = os.path.basename(image_path)
name = os.path.splitext(img_basename)[0]

# 获取模型层数
child = list(model.children())
LN = len(child)

# 创建NNS对象
print(f"\n6. 开始NNS扫描（12层）...")
print(f"   这可能需要1-2分钟...")
nns = NNS_NONET(
    model, image_for_nns, name, 0, device,
    "vit_base_patch16_224_pretrained_n01443537_24", 
    'VisionTransformer', 'ImageNet',
    16, 12, 12, params  # patch_size=16, num_heads=12, depth=12
)

# 运行扫描
if not os.path.exists(nns.rootdir):
    nns.get_scanpic(LN)
    print(f"   ✓ NNS扫描完成")
else:
    print(f"   结果已存在: {nns.rootdir}")

print(f"\n✓ 处理完成！")
print(f"结果保存在: {nns.rootdir}")
print("="*70)

