import time
import numpy as np
import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import math
import torch.nn.functional as F
import torch.distributed as dist
import argparse
from PIL import Image
import datetime as dt
import sys
from model.vit_final import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from NNS.Scannerpic1107_cls_nonet import NNS_NONET
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
# Notice: 这里的64，32是img_size大小
from function.dataset32 import *
from function.helper import *


parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, default='1',
                    help='cuda number')
#parser.add_argument('--acc', '-ac', type=str, default='03', #8 00-04 indexarr = [0,30,50,60,70]
                    #help='model accuracy')
parser.add_argument('--patch_size', default=4, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--depth', default=6, type=int)
parser.add_argument('--count', '-c', type=int, default=250,
                    help='sample number')
parser.add_argument('--model', type=str, default='VisionTransformer',
                    help='VisionTransformer alexnet32 vgg1632 vgg1132 resnet mlp')
parser.add_argument('--dataset', type=str, default='tiny200ILSVRC',
                    help='ILSVRC32 tiny200ILSVRC CUB VOC MNIST ImageNet')
#parser.add_argument('--mark', type=str, default='epoch') #test test1
#parser.add_argument('--accmark', type=str, default='00') #test test1
parser.add_argument('--classes', type=int, default=10) #test 10,25, 1000 for ImageNet
parser.add_argument('--use_all_class', default=False, type=bool)
parser.add_argument('--process_wrong_classification', default=False, type=bool,
                    help='是否处理分类错误的图像（默认False，只处理分类正确的图像）')
parser.add_argument('--use_pretrained', default=False, type=bool,
                    help='是否使用预训练模型（ImageNet）')
parser.add_argument('--imagenet_path', type=str, default='',
                    help='ImageNet数据集路径（val目录）')
parser.add_argument('--nns_resize', type=int, default=None,
                    help='NNS处理时的图像尺寸（默认None，使用原始尺寸。对于大图像建议设置为64或128以节省内存）')
args = parser.parse_args()

device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
print('device:'+args.device+','+str(torch.cuda.is_available()))
print(args.model+'_'+args.dataset)





def train_scanenr():

    classes=args.classes
    COUNT=args.count
    # 获取项目根目录（自动检测Windows或Linux路径）
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # 尝试Windows路径
    if os.path.exists(os.path.join(project_root, 'NNS_model_and_dataset')):
        base_path = os.path.join(project_root, 'NNS_model_and_dataset')
    # 尝试Linux路径
    elif os.path.exists('/nas/NNS_model_and_dataset'):
        base_path = '/nas/NNS_model_and_dataset'
    else:
        # 默认使用项目根目录
        base_path = os.path.join(project_root, 'NNS_model_and_dataset')
    
    if args.model == 'VisionTransformer':
        # ImageNet + ViT-base-16 预训练模型
        if args.dataset == "ImageNet":
            # 使用自定义ViT-base-16架构（兼容timm的预训练权重）
            print("初始化ViT-base-16模型架构...")
            model = VisionTransformer(img_size=[224], patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
            
            # 尝试加载timm的预训练权重
            try:
                import timm
                print("尝试加载timm预训练权重...")
                timm_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
                timm_model.eval()
                
                # 映射timm参数到我们的模型
                print("映射预训练权重...")
                state_dict = timm_model.state_dict()
                our_state_dict = model.state_dict()
                
                # 参数名映射（timm -> 我们的模型）
                param_mapping = {
                    'patch_embed.proj.weight': 'patch_embed.proj.weight',
                    'patch_embed.proj.bias': 'patch_embed.proj.bias',
                    'cls_token': 'cls_token',
                    'pos_embed': 'pos_embed',
                }
                
                # 映射blocks
                for i in range(12):
                    param_mapping[f'blocks.{i}.norm1.weight'] = f'blocks.{i}.attn.norm.weight' if hasattr(model.blocks[i].attn, 'norm') else None
                    param_mapping[f'blocks.{i}.norm1.bias'] = f'blocks.{i}.attn.norm.bias' if hasattr(model.blocks[i].attn, 'norm') else None
                    param_mapping[f'blocks.{i}.attn.qkv.weight'] = None  # 我们的模型结构不同
                    param_mapping[f'blocks.{i}.attn.qkv.bias'] = None
                    param_mapping[f'blocks.{i}.attn.proj.weight'] = None
                    param_mapping[f'blocks.{i}.attn.proj.bias'] = None
                    param_mapping[f'blocks.{i}.norm2.weight'] = None
                    param_mapping[f'blocks.{i}.norm2.bias'] = None
                    param_mapping[f'blocks.{i}.mlp.fc1.weight'] = None
                    param_mapping[f'blocks.{i}.mlp.fc1.bias'] = None
                    param_mapping[f'blocks.{i}.mlp.fc2.weight'] = None
                    param_mapping[f'blocks.{i}.mlp.fc2.bias'] = None
                
                # 由于timm的ViT结构与我们自定义的ViT结构不同，我们直接使用timm模型
                # 但需要适配NNS代码
                print("使用timm模型（注意：需要适配NNS代码）...")
                model = timm_model
                model.eval()
                
            except ImportError:
                print("警告: timm未安装，使用随机初始化的模型")
                print("建议安装timm: pip install timm")
            except Exception as e:
                print(f"加载预训练权重失败: {e}")
                print("使用随机初始化的模型")
            
            # ImageNet数据路径
            if args.imagenet_path:
                traindir = args.imagenet_path
            else:
                # 尝试常见的ImageNet路径
                possible_paths = [
                    os.path.join(base_path, 'dataset', 'imagenet', 'val'),
                    os.path.join(base_path, 'dataset', 'ImageNet', 'val'),
                    '/data/imagenet/val',
                    '/dataset/imagenet/val',
                    'D:/data/imagenet/val',
                    'D:/ImageNet/val',
                ]
                traindir = None
                for path in possible_paths:
                    if os.path.exists(path):
                        traindir = path
                        break
                
                if traindir is None:
                    raise FileNotFoundError(f"找不到ImageNet数据目录，请使用--imagenet_path指定路径，或检查以下路径: {possible_paths}")
            
            print(f"使用ImageNet数据目录: {traindir}")
            modeldir = None  # 预训练模型不需要从文件加载
            args.classes = 1000  # ImageNet有1000个类别
            args.patch_size = 16  # ViT-base-16使用16x16的patch
            args.num_heads = 12
            args.depth = 12
                
        elif args.dataset=="tiny200ILSVRC":
            # 根据参数动态构建模型目录路径
            # 格式: tiny200ILSVRC_VisionTransformer_32imgsize_10classes_{threshold}threshold_{num_heads}heads_{depth}depth_{patch_size}patch_size_
            
            # 尝试多个可能的threshold值（0, 30, 40, 45等）
            possible_thresholds = [0, 30, 35, 40, 45]
            possible_modeldirs = []
            
            # 优先使用mutlihead目录
            for threshold in possible_thresholds:
                modeldir_name = f'tiny200ILSVRC_VisionTransformer_32imgsize_{args.classes}classes_{threshold}threshold_{args.num_heads}heads_{args.depth}depth_{args.patch_size}patch_size_'
                possible_modeldirs.append(
                    os.path.join(base_path, 'models', 'vit_3channel_mutlihead', modeldir_name)
                )
            
            # 也尝试vit_3channel目录
            for threshold in possible_thresholds:
                modeldir_name = f'tiny200ILSVRC_VisionTransformer_32imgsize_{args.classes}classes_{threshold}threshold_{args.num_heads}heads_{args.depth}depth_{args.patch_size}patch_size_'
                possible_modeldirs.append(
                    os.path.join(base_path, 'models', 'vit_3channel', modeldir_name)
                )
            
            modeldir = None
            for md in possible_modeldirs:
                if os.path.exists(md):
                    modeldir = md
                    print(f"找到模型目录: {modeldir}")
                    break
            
            if modeldir is None:
                print(f"警告: 未找到匹配的模型目录，尝试的路径:")
                for md in possible_modeldirs[:5]:  # 只显示前5个
                    print(f"  - {md}")
                # 如果找不到，尝试查找任何包含depth的目录
                mutlihead_dir = os.path.join(base_path, 'models', 'vit_3channel_mutlihead')
                if os.path.exists(mutlihead_dir):
                    all_dirs = [d for d in os.listdir(mutlihead_dir) if f'{args.depth}depth' in d]
                    if all_dirs:
                        modeldir = os.path.join(mutlihead_dir, all_dirs[0])
                        print(f"使用找到的目录: {modeldir}")
                
                if modeldir is None:
                    raise FileNotFoundError(f"找不到depth={args.depth}的模型目录，请检查路径")
            
            # 数据路径 - 优先使用test111目录（包含指定的n01443537_24图像）
            alt_paths = [
                os.path.join(base_path, 'dataset', 'tiny-imagenet-5', 'test111'),  # 1张图片（n01443537_24）
                os.path.join(base_path, 'dataset', 'tiny-imagenet-5', 'scanner'),  # 250张图片
                os.path.join(base_path, 'dataset', 'tiny-imagenet-5', 'val'),      # 1250张图片
            ]
            traindir = None
            for alt in alt_paths:
                if os.path.exists(alt):
                    traindir = alt
                    print(f"使用数据目录: {traindir}")
                    break
            if traindir is None:
                raise FileNotFoundError(f"找不到数据目录，请检查以下路径: {alt_paths}")
            
            model = VisionTransformer(img_size=[32],patch_size=args.patch_size, num_classes=args.classes, embed_dim=192, depth=args.depth, num_heads=args.num_heads, mlp_ratio=4,
                                            qkv_bias=False)

        if args.dataset == 'MNIST':       
            modeldir = os.path.join(base_path, 'models', 'vit_3channel', 'MNIST_VisionTransformer_28imgsize_10classes_80threshold_1heads_3depth_4patch_size_')
            traindir = os.path.join(base_path, 'dataset', 'mnist_scanner')
            model = VisionTransformer(img_size=[28],patch_size=args.patch_size, in_chans=1, num_classes=args.classes, embed_dim=192, depth=3, num_heads=1, mlp_ratio=4,
                                            qkv_bias=False)
            
            
            
    child = list(model.children())
    LN = len(child)


    if args.dataset == "ImageNet":
        # 使用ImageNet数据集加载器（224x224）
        from function.dataset224 import ScanDataSet224_num, ScanDataSet224
        if args.use_all_class:
            dataloader = torch.utils.data.DataLoader(ScanDataSet224(traindir, args.classes), batch_size=1, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(ScanDataSet224_num(traindir, args.classes), batch_size=1, shuffle=False)
    elif args.dataset=="tiny200ILSVRC":
        if args.use_all_class:
            dataloader = torch.utils.data.DataLoader(ScanDataSet(traindir, ''), batch_size=1, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(ScanDataSet_num(traindir, args.classes), batch_size=1, shuffle=False)
    elif args.dataset == 'MNIST': 
        dataloader = torch.utils.data.DataLoader(MnistDataset(traindir),batch_size=1, shuffle=True)
    
    
    if modeldir is not None:
        print(f"模型目录: {modeldir}")
        print(f"数据目录: {traindir}")
        if not os.path.exists(modeldir):
            raise FileNotFoundError(f"模型目录不存在: {modeldir}")
        if not os.path.exists(traindir):
            raise FileNotFoundError(f"数据目录不存在: {traindir}")
        
        model_files = [f for f in os.listdir(modeldir) if f.endswith('.pth')]
        
        # 如果目录中没有文件，检查上一级目录（因为train_net_simple.py保存时可能没有加斜杠）
        if not model_files:
            parent_dir = os.path.dirname(modeldir)
            parent_files = [f for f in os.listdir(parent_dir) if f.endswith('.pth') and os.path.basename(modeldir) in f]
            if parent_files:
                print(f"在上一级目录找到模型文件: {parent_files}")
                # 将文件复制或移动到正确位置，或者直接使用
                model_files = [os.path.join(parent_dir, f) for f in parent_files]
            else:
                raise FileNotFoundError(f"在 {modeldir} 及其父目录中找不到模型文件(.pth)")
        else:
            model_files = [os.path.join(modeldir, f) for f in model_files]
        
        print(f"找到 {len(model_files)} 个模型文件")
        
        # model_files现在可能是完整路径或文件名
        models_to_process = []
        for m in model_files:
            if os.path.isabs(m) or os.path.dirname(m):
                # 已经是完整路径
                model_path = m
                full_model_name = os.path.splitext(os.path.basename(m))[0]
            else:
                # 只是文件名
                model_path = os.path.join(modeldir, m)
                full_model_name = ".".join(m.split('.')[:-1])
            
            # 缩短模型名称以避免Windows路径长度限制（260字符）
            # 提取关键信息：depth, epoch, final标记
            short_name = full_model_name
            if 'final' in full_model_name.lower():
                # 如果是final模型，使用简短名称
                short_name = f"depth{args.depth}_final"
            else:
                # 提取epoch号
                import re
                epoch_match = re.search(r'_(\d+)_', full_model_name)
                if epoch_match:
                    epoch = epoch_match.group(1)
                    short_name = f"depth{args.depth}_ep{epoch}"
                else:
                    # 如果找不到epoch，使用哈希值的前8位
                    import hashlib
                    short_name = f"depth{args.depth}_{hashlib.md5(full_model_name.encode()).hexdigest()[:8]}"
            
            models_to_process.append((model_path, short_name))
        
        # 如果指定了depth=12，优先使用final模型
        if args.depth == 12:
            final_models = [m for m in models_to_process if 'final' in m[1].lower()]
            if final_models:
                # 找到对应的原始文件名
                final_path = final_models[0][0]
                original_name = os.path.splitext(os.path.basename(final_path))[0]
                print(f"找到final模型，优先使用: {final_models[0][1]} (原名称: {original_name})")
                models_to_process = final_models[:1]  # 只使用final模型
    else:
        # 使用预训练模型（如ImageNet）
        print(f"数据目录: {traindir}")
        if not os.path.exists(traindir):
            raise FileNotFoundError(f"数据目录不存在: {traindir}")
        
        # 预训练模型已经加载，直接使用
        model = model.to(device)
        print(f"预训练模型已加载到设备: {device}")
        
        # 提取参数用于NNS（适配timm模型结构）
        params = {}
        state_dict = model.state_dict()
        
        # 提取cls_token和pos_embed（timm模型使用这些名称）
        if 'cls_token' in state_dict:
            params['cls_token'] = state_dict['cls_token'].clone()
        else:
            # 尝试其他可能的名称
            for key in state_dict.keys():
                if 'cls_token' in key.lower():
                    params['cls_token'] = state_dict[key].clone()
                    break
        
        if 'pos_embed' in state_dict:
            params['pos_embed'] = state_dict['pos_embed'].clone()
        else:
            # 尝试其他可能的名称
            for key in state_dict.keys():
                if 'pos_embed' in key.lower():
                    params['pos_embed'] = state_dict[key].clone()
                    break
        
        # 如果找不到，尝试从模型属性获取
        if 'cls_token' not in params:
            if hasattr(model, 'cls_token'):
                params['cls_token'] = model.cls_token.data.clone()
            elif hasattr(model, 'pos_embed'):
                # 有些模型可能没有cls_token，创建一个
                print("警告: 模型没有cls_token，将创建一个")
                params['cls_token'] = torch.zeros(1, 1, 768)
        
        if 'pos_embed' not in params:
            if hasattr(model, 'pos_embed'):
                params['pos_embed'] = model.pos_embed.data.clone()
        
        # 复制所有参数（用于NNS的其他部分）
        for name, param in model.named_parameters():
            params[name] = param.data.clone()
        
        models_to_process = [(None, "vit_base_patch16_224_pretrained")]
    
    for model_path, model_name in models_to_process:
        if model_path is not None:
            print(f"\n处理模型: {model_name}")
            params = load_checkpoint_and_return_params(model_path, model)
            model = model.to(device)
            print(f"模型已加载到设备: {device}")
        else:
            print(f"\n使用预训练模型: {model_name}")


        pbar = tqdm(total=COUNT)
        ii = 0
        real = 0
        wrong_count = 0
        correct_count = 0
        
        for i, (id, img_name, image, target) in enumerate(dataloader):

            if ii>=COUNT:
                break

            image = image.to(device)
            target = target.to(device)
            output = model(image)
            out = torch.argsort(output)[0,-1:]
            is_correct = (out == target)

            # 根据参数决定是否处理分类错误的图像
            if not is_correct and not args.process_wrong_classification:
                print(f"跳过分类错误的图像: {img_name[0]} (预测: {out.item()}, 实际: {target.item()})")
                wrong_count += 1
                continue

            # 提取文件名（处理Windows和Linux路径）
            img_path = img_name[0]
            img_basename = os.path.basename(img_path)  # 获取文件名（如 n03976467_9546.JPEG）
            name = os.path.splitext(img_basename)[0]  # 去掉扩展名（如 n03976467_9546）
            status = "[OK]" if is_correct else "[WRONG]"  # 使用ASCII字符避免编码问题
            print(f"{status} {img_path}*{name} (预测: {out.item()}, 实际: {target.item()})")
            
            if is_correct:
                correct_count += 1
            else:
                wrong_count += 1
                # 对于分类错误的图像，在名称中添加标记
                name = f"{name}_wrong_pred{out.item()}"

            # 如果需要降采样处理（节省内存）
            image_for_nns = image[0]
            if args.nns_resize is not None and image_for_nns.shape[-1] > args.nns_resize:
                import torch.nn.functional as F
                original_size = image_for_nns.shape[-1]
                image_for_nns_resized = F.interpolate(
                    image_for_nns.unsqueeze(0), 
                    size=(args.nns_resize, args.nns_resize), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                print(f"  注意: 图像从 {original_size}×{original_size} 降采样到 {args.nns_resize}×{args.nns_resize} 以节省内存")
                image_for_nns = image_for_nns_resized
            
            nns = NNS_NONET(model,image_for_nns,name,ii,device,model_name,args.model,args.dataset,args.patch_size,args.num_heads,args.depth,params)
            #不重复跑
            print(f"  检查目录: {nns.rootdir}")
            if not os.path.exists(nns.rootdir):
                print(f"  目录不存在，开始NNS处理...")
                nns.get_scanpic(LN)
                print(f"  NNS处理完成")
            else:
                print(f"  目录已存在，跳过NNS处理")

            ii = ii + 1
            torch.cuda.empty_cache()
            pbar.update(1)
        
        print(f"\n处理完成统计:")
        print(f"  分类正确: {correct_count} 张")
        print(f"  分类错误: {wrong_count} 张 (已{'处理' if args.process_wrong_classification else '跳过'})")
        print(f"  总计: {ii} 张图像")

        pbar.close()




def load_checkpoint_and_return_params(checkpoint, model, optimizer=None):   
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location=device)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])
    # print(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict']) 
    
    params = checkpoint['state_dict']
    
    return params

if __name__ == '__main__':
    train_scanenr()
