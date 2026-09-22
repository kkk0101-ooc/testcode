import torch
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr
import os
from tempfile import tempdir
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sys
from itertools import product
from importlib import import_module
from PIL import Image
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from scipy.spatial.distance import cdist

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


'''
1 Conv2d

---Block 1---
2 Attention
3 Linear
4 Relu
5 Linear

---Block 2---
6 Attention
7 Linear
8 Relu
9 Linear

---Block 3---
10 Attention
11 Linear
12 Relu
13 Linear

---Block 4---
14 Attention
15 Linear
16 Relu
17 Linear

---Block 5---
18 Attention
19 Linear
20 Relu
21 Linear

---Block 6---
22 Attention
23 Linear
24 Relu
25 Linear

26 Linear
'''


saveTensor = True
saveFeature = True
savePicure = True
saveSumTen = True
pic_RGB = True

class NNS_NONET():
    def __init__(self, network, imageTensor, img_name,id, device,model_name,model_type,dataset,patch_size,num_heads,depth,params):
        self.network = network
        self.img_name = img_name
        patch_height, patch_width = pair(patch_size)
        self.imageTensor = imageTensor # torch.Size([1, 28, 28]) (c,h,w)
        self.c = imageTensor.shape[0]
        self.h = imageTensor.shape[1]
        self.w = imageTensor.shape[2]
        self.device = device
        self.original_img_size = imageTensor.size()
        # self.pos_embedding = param_posembed #(1,16,64)
        # self.heads = heads#1
        # self.scale = dim_head ** -0.5
        self.params = params
        self.patch_size = patch_size
        self.heads = num_heads
        self.depth = depth
        self.cls_token = params['cls_token']
        self.pos_embed = params['pos_embed']
        self.embed_dim = self.pos_embed.shape[-1]
        self.scale = self.embed_dim ** -0.5
        
        # 保存类别权重，用于加权注意力可视化
        self.class_weights = None  # 最后的Linear层输出的类别权重
        self.predicted_class = None  # 预测的类别
        self.class_feature_map = None  # 类别特征图 (10, H, W) - 用于加权注意力
        
        # 自动检测输出路径（Windows或Linux）
        import os
        # 尝试多个可能的输出路径
        possible_output_bases = [
            '/hy-tmp/1_vit_final/datas/scan_vit/',  # Linux路径
            os.path.join(os.path.expanduser('~'), 'scan_results', 'scan_vit'),  # Windows用户目录
            os.path.join('.', 'scan_results', 'scan_vit'),  # 当前目录
        ]
        output_base = None
        for base in possible_output_bases:
            if os.path.exists(os.path.dirname(base)) or base.startswith('.'):
                output_base = base
                break
        if output_base is None:
            # 默认使用当前目录
            output_base = os.path.join('.', 'scan_results', 'scan_vit')
            os.makedirs(output_base, exist_ok=True)
        
        # 清理img_name，只使用文件名（去除路径和非法字符）
        # img_name可能是完整路径或文件名，需要提取纯文件名
        if isinstance(img_name, str):
            # 如果包含路径分隔符，提取文件名
            if os.sep in img_name or '/' in img_name or '\\' in img_name:
                img_name = os.path.basename(img_name)
            # 移除文件扩展名（如果有）
            img_name = os.path.splitext(img_name)[0]
            # 替换可能的非法字符（Windows不允许 : < > | " ? *）
            invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
            for char in invalid_chars:
                img_name = img_name.replace(char, '_')
        
        self.rootdir = os.path.join(output_base, '114' + str(model_type) + '_' + str(dataset), str(model_name), str(id) + '_' + str(img_name))


        if pic_RGB:
            self.scan_path_pic = os.path.join(self.rootdir, 'scan_RGB/')
        else:
            self.scan_path_pic = os.path.join(self.rootdir, 'scan_gray/')
        self.scan_path_ten = os.path.join(self.rootdir, 'scan_ten/')
        self.scan_path_attn = os.path.join(self.rootdir, 'scan_attn_ten/')
        self.scan_path_fea = os.path.join(self.rootdir, 'scan_fea/')
        self.scan_path_sumten = os.path.join(self.rootdir, 'scan_sumten/')
        self.scan_path_ori_pic = os.path.join(self.rootdir, 'scan_ori/')
        self.scan_attn_map = os.path.join(self.rootdir, 'scan_attn_map/')
        self.scan_v = os.path.join(self.rootdir, 'scan_v/')
        

        self.imageTensor_Length = imageTensor.shape[-1]
        self.imageTensor_Width = imageTensor.shape[-2]
        
        self.patch_height = patch_height
        self.patch_width = patch_width
        
        
    


    def get_scanpic(self, layernum):

        imageTensor = self.imageTensor # (3, 224, 224)
                                       # (3, 64, 64)

        self.saveOri(imageTensor)
        
        inputTensor = self.getFirstLayer(imageTensor) #(224,224,3,224,224)
                                                      #(64,64,3,64,64)
        
        if imageTensor.dim() == 2:
            imageTensor = torch.unsqueeze(torch.unsqueeze(imageTensor,0),0)
        if imageTensor.dim() == 3:
            imageTensor = torch.unsqueeze(imageTensor,0)
            
        imageTensor = imageTensor.to(self.device)#(1,3,224,224)
                                                 #(1,3,32,32)
        inputTensor = inputTensor.to(self.device)#(224,224,3,224,224)
                                                 #(32,32,3,32,32)

        pic_layer = 0

        layerIndex = 0

        ###scan the network
        '''
        PatchEmbed
            Conv2d
        ModuleList
            Block (x6) 
                Attention
                    Linear
                    Linear
                Mlp
                    Linear
                    ReLU
                    Linear
        Linear
        '''
        
        
        # 调试：打印模型结构
        children_list = list(self.network.children())
        print(f"  [DEBUG] 模型children数量: {len(children_list)}")
        for i, child in enumerate(children_list):
            print(f"  [DEBUG] Child {i}: {type(child).__name__}")
        
        if len(children_list) == 0:
            print(f"  [WARNING] 模型没有children！尝试使用named_children...")
            for name, child in self.network.named_children():
                print(f"  [DEBUG] Named child: {name} -> {type(child).__name__}")
        
        for layer in self.network.children():
            if layerIndex > layernum:
                break

            layerName = type(layer).__name__
            
            ###Sequential layer process
            if layerName == 'PatchEmbed':
                print(layerName)
                layerIndex += 1
                if layerIndex > layernum:
                    break
                
                for sublayer in layer.children():
                    sublayerName = type(sublayer).__name__
                    print(sublayerName)
                    
                    if sublayerName == 'Conv2d':
                        # x = self.patch_embed(x)
                        pic_layer += 1
                        # x = self.proj(x)
                        # (1,3,224,224), (224,224,3,224,224)
                        # (1,3,64,64), (64,64,3,64,64)
                        imageTensor, inputTensor = self.conv_pytorch(sublayer,imageTensor,inputTensor)
                        # (1,192,14,14), (14,14,192,224,224)
                        # (1,192,16,16), (16,16,192,64,64)           
                                    
                        # x = x.flatten(2).transpose(1, 2)
                        
                        # (1,192,14,14), (14,14,192,224,224)
                        imageTensor = imageTensor.flatten(2) #(1,192,196)
                                                             #(1,192,256)
                        imageTensor = imageTensor.transpose(1, 2) #(1,196,192)
                                                                  #(1,256,192)
                        
                        inputTensor = inputTensor.permute(3,4,2,0,1) #(224,224,192,14,14)
                                                                     #(64,64,192,16,16)
                        inputTensor = inputTensor.flatten(3) #(224,224,192,196)
                                                             #(64,64,192,256)
                        inputTensor = inputTensor.transpose(2, 3) #(224,224,196,192)
                                                                  #(64,64,256,192)
                        inputTensor = inputTensor.permute(2,3,0,1) #(196,192,224,224)
                                                                   #(256,192,64,64)
                        
                        # add the [CLS] token
                        cls_tokens = self.cls_token.expand(1, -1, -1) # cls_tokens:(1,1,192)
                        imageTensor = torch.cat((cls_tokens, imageTensor), dim=1) # imageTensor: (1,197,192)
                                                                                  # (1,257,192)
                        # add positional encoding to each token
                        # 处理pos_embed维度不匹配的情况（如降采样后的图像）
                        if imageTensor.shape[1] != self.pos_embed.shape[1]:
                            # 需要调整pos_embed的大小
                            num_patches = imageTensor.shape[1] - 1  # 减去CLS token
                            pos_embed_cls = self.pos_embed[:, 0:1, :]  # CLS token的位置编码
                            pos_embed_patches = self.pos_embed[:, 1:, :]  # patch的位置编码
                            
                            # 计算原始和目标的patch网格大小
                            orig_num_patches = pos_embed_patches.shape[1]
                            orig_size = int(orig_num_patches ** 0.5)
                            new_size = int(num_patches ** 0.5)
                            
                            # 使用双线性插值调整位置编码
                            pos_embed_patches = pos_embed_patches.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                            pos_embed_patches = torch.nn.functional.interpolate(
                                pos_embed_patches, size=(new_size, new_size), mode='bilinear', align_corners=False
                            )
                            pos_embed_patches = pos_embed_patches.permute(0, 2, 3, 1).reshape(1, num_patches, -1)
                            
                            # 合并CLS token和patch的位置编码
                            pos_embed_resized = torch.cat([pos_embed_cls, pos_embed_patches], dim=1)
                            imageTensor = imageTensor + pos_embed_resized
                        else:
                            imageTensor = imageTensor + self.pos_embed # imageTensor: (1,64,192)
                                                                       # (1,64,192)
                        
                        inputTensor = inputTensor.permute(2,3,0,1) #(32,32,64,192)
                        # Notice,这里的64,64是img_size                                         
                        input_cls_tokens = self.cls_token.expand(self.w, self.h, -1, -1) # cls_tokens:(224,224,1,192)
                                                                                 # (64,64,1,192)
                        inputTensor = torch.cat((input_cls_tokens, inputTensor), dim=2) #(224,224,197,192)
                                                                                        #(64,64,257,192)
                        
                        # 处理input_pos_embed维度不匹配的情况
                        if inputTensor.shape[2] != self.pos_embed.shape[1]:
                            # 需要调整pos_embed的大小以匹配inputTensor
                            num_patches = inputTensor.shape[2] - 1  # 减去CLS token
                            pos_embed_cls = self.pos_embed[:, 0:1, :]  # CLS token的位置编码
                            pos_embed_patches = self.pos_embed[:, 1:, :]  # patch的位置编码
                            
                            # 计算原始和目标的patch网格大小
                            orig_num_patches = pos_embed_patches.shape[1]
                            orig_size = int(orig_num_patches ** 0.5)
                            new_size = int(num_patches ** 0.5)
                            
                            # 使用双线性插值调整位置编码
                            pos_embed_patches = pos_embed_patches.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                            pos_embed_patches = torch.nn.functional.interpolate(
                                pos_embed_patches, size=(new_size, new_size), mode='bilinear', align_corners=False
                            )
                            pos_embed_patches = pos_embed_patches.permute(0, 2, 3, 1).reshape(1, num_patches, -1)
                            
                            # 合并CLS token和patch的位置编码
                            pos_embed_resized = torch.cat([pos_embed_cls, pos_embed_patches], dim=1)
                            input_pos_embed = pos_embed_resized.expand(self.w, self.h, -1, -1)
                        else:
                            input_pos_embed = self.pos_embed.expand(self.w, self.h, -1, -1) #(224,224,197,192)
                                                                                    #(64,64,257,192)
                        inputTensor = inputTensor + input_pos_embed #(32,32,64,192)
                        
                        inputTensor = inputTensor.permute(2,3,0,1) #(197,192,224,224)
                                                                   #(257,192,64,64)
                        
                        # self.saveDIR(imageTensor, inputTensor,pic_layer)
                        
                        
                    # if sublayerName == 'Linear':
                    #     pic_layer += 1
                    #     ###x = x.flatten(2) # (b, 49, 16)
                    #     ###x= x.transpose(1, 2) # (b, 16, 49)
                    #     # (1,49,4,4), (4,4,49,28,28)
                        
                        
                    #     #(1,16,49), (16,49,28,28)
                    #     imageTensor = torch.squeeze(imageTensor)
                    #     #(16,49), (16,49,28,28)
                    #     imageTensor, inputTensor = self.linear_pytorch(sublayer,imageTensor,inputTensor)
                    #     #(16,64), (16,64,28,28)
                    #     imageTensor = torch.unsqueeze(imageTensor,0)#(1,16,64)
                    #     #(1,16,64),(16,64,28,28)
                    #     self.saveDIR(imageTensor, inputTensor,pic_layer)
                    #     # self.Features_vision(imageTensor)
                    #     # self.Features_vision(inputTensor)
            
            ###Transformer layer process
            # 兼容timm模型：timm使用Sequential而不是ModuleList
            if layerName == 'ModuleList' or layerName == 'Sequential':
                layerIndex += 1
                if layerIndex > layernum:
                    break
                
                print(f"  [INFO] 处理 {layerName}，包含 {len(list(layer.children()))} 个子模块")
                
                for sublayer1 in layer.children():
                    sublayerName1 = type(sublayer1).__name__
                    print(sublayerName1)
                    if sublayerName1 == 'Block':
                        for sublayer2 in sublayer1.children():
                            sublayerName2 = type(sublayer2).__name__
                            print(sublayerName2)
                                    
                            if sublayerName2 == 'Attention':
                                pic_layer += 1
                                
                                originalImageTensor = imageTensor#residual
                                originalInputTensor = inputTensor  

                                #(1,65,192),(65,192,32,32)
                                imageTensor, image_attn, inputTensor, input_attn, input_v = self.new_attention_pytorch(sublayer2,imageTensor,inputTensor)
                                #(1,65,192),(1,3,65,65),(65,192,32,32),(32,32,3,65,65)
                                
                                ### Residual Add
                                inputTensor = inputTensor + originalInputTensor
                                imageTensor = imageTensor + originalImageTensor
                                #(1,257,192),(257,192,64,64)
                                #(1,65,192),(65,192,32,32)
                                
                                # self.saveDIR(imageTensor, inputTensor,pic_layer)
                                
                                name = str(pic_layer).rjust(2,'0')
                                
                                self.saveTen(image_attn, self.scan_path_attn+name+'/')
                                
                                # 先保存注意力tensor，等final_linear执行后再可视化
                                # 暂时先可视化，如果类别特征图可用会改进
                                self.visualize_attention(image_attn,input_attn,pic_layer)
                                  
                                # # (32,32,3,65,64)
                                self.visualize_v(input_v,pic_layer)
                                    
                            if sublayerName2 == 'Mlp':
                                
                                originalImageTensor = imageTensor#residual
                                originalInputTensor = inputTensor 
                                
                                for sublayer3 in sublayer2.children():
                                    sublayerName3 = type(sublayer3).__name__
                                    print(sublayerName3)
                                    
                                    
                                    
                                    if sublayerName3 == 'Linear':
                                        pic_layer += 1
                                        #(1,16,64),(16,64,28,28)
                                        #(1,257,192),(257,192,64,64)
                                        imageTensor = torch.squeeze(imageTensor)#(16,64)
                                        #(16,64),(16,64,28,28)
                                        imageTensor, inputTensor = self.linear_pytorch(sublayer3,imageTensor,inputTensor)
                                        #(16,128),(16,128,28,28)
                                        imageTensor = torch.unsqueeze(imageTensor,0)#(1,16,128)
                                        #(1,16,128),(16,128,28,28)
                                        
                                        ### (fc2): Linear(in_features=768, out_features=192, bias=False)
                                        if sublayer3.in_features > sublayer3.out_features:
                                                                                        
                                            ### Residual Add
                                            inputTensor = inputTensor + originalInputTensor
                                            imageTensor = imageTensor + originalImageTensor
                                            
                                        
                                        # self.saveDIR(imageTensor, inputTensor,pic_layer)
                                                    
                                    if sublayerName3 == 'ReLU':
                                        #(1,16,128),(16,128,28,28)
                                        #(1,257,192),(257,192,64,64)
                                        imageTensor = torch.squeeze(imageTensor)
                                        pic_layer += 1
                                        #(16,128),(16,128,28,28)
                                        # (257,192),(257,192,64,64)
                                        imageTensor, inputTensor = self.ReLU_pytorch(sublayer3,imageTensor,inputTensor)
                                        imageTensor = torch.unsqueeze(imageTensor,0)
                                        #(1,16,128),(16,128,28,28)
                                        # self.saveDIR(imageTensor, inputTensor,pic_layer)
                                        

            
            # ###Identity layer process
            # if layerName == 'Identity':
            #     layerIndex += 1
            #     if layerIndex > layernum:
            #         break
            #     #x = x.mean(dim = 1)
            #     #(1,16,64),(16,64,28,28)
            #     imageTensor = imageTensor.mean(dim = 1)
            #     inputTensor = inputTensor.mean(dim = 0)
                
            #     pic_layer += 1
                
            #     #(1,64),(64,28,28)
            #     self.saveDIR(imageTensor, inputTensor, pic_layer)
                
            
            
            
            ###linear layer process
            if layerName == 'Linear':
                layerIndex += 1
                if layerIndex > layernum:
                    break
                pic_layer += 1
                print(layerName)
                                    
                #(1,257,192),(257,192,64,64)
                
                # cls操作
                imageTensor = imageTensor[:,0]#(1,192)
                inputTensor = inputTensor[0,:]#(192,64,64)
                # (1,64),(64,28,28)
                # (1,192),(192,64,64)
                
                # #mean操作
                # imageTensor = imageTensor.mean(dim = 1)
                # inputTensor = inputTensor.mean(dim = 0)
                # #(1,192),(192,64,64)
                
                imageTensor = torch.squeeze(imageTensor)
                #(64),(64,28,28)
                #(192),(192,64,64)
                imageTensor,inputTensor=self.final_linear_pytorch(layer,imageTensor,inputTensor)
                #(10),(10,28,28)
                #(25),(25,64,64)
                imageTensor = torch.unsqueeze(imageTensor,0)
                #(1,10),(10,28,28)
                #(1,25),(25,64,64)
                self.saveDIR(imageTensor, inputTensor,pic_layer)
                
            



        return

    def visualize_v(self, input_v, pic_layer):
        """
        Enhanced fine-grained visualization of the 'value' component from attention mechanism.
        Creates detailed pixel-level visualizations similar to CNN feature maps.
        
        Parameters:
            input_v: Tensor of shape (H, W, heads, num_patches+1, head_dim)
            pic_layer: Layer number for naming output files
        """
        name = str(pic_layer).rjust(2, '0')
        v_path = self.scan_v + name + '/'
        if not os.path.exists(v_path):
            os.makedirs(v_path)
        
        w, h = self.w, self.h
        patch_size = self.patch_size
        w_featmap = max(1, w // patch_size)
        h_featmap = max(1, h // patch_size)
        
        # Process for each attention head
        for head_idx in range(self.heads):
            head_subpath = v_path + f'head{head_idx}/'
            if not os.path.exists(head_subpath):
                os.makedirs(head_subpath)
            
            # Extract value vectors for this head
            head_v = input_v[:, :, head_idx, :, :]  # (H, W, num_patches+1, head_dim)
            
            # 1. Get the value vector magnitude (L2 norm) for each position
            v_magnitudes = torch.norm(head_v, dim=3)  # (H, W, num_patches+1)
            
            # 2. CLS token value visualization
            cls_v_magnitudes = v_magnitudes[:, :, 0].to('cpu')  # (H, W)
            self.show_neural_feature_hotimg(cls_v_magnitudes, v_path + f'cls_v_magnitude_head{head_idx}.png')
            
            # 3. Average magnitude across all tokens
            mean_v_magnitude = torch.mean(v_magnitudes, dim=2).to('cpu')  # (H, W)
            self.show_neural_feature_hotimg(mean_v_magnitude, v_path + f'mean_v_magnitude_head{head_idx}.png')
            
            # 4. Fine-grained: Visualize value vectors for each patch (excluding CLS)
            # This shows how value information is distributed across patches
            if v_magnitudes.shape[2] > 1:
                patch_v_magnitudes = v_magnitudes[:, :, 1:].to('cpu')  # (H, W, num_patches)
                
                # Average value magnitude per patch across all spatial positions
                avg_patch_v = torch.mean(patch_v_magnitudes, dim=(0, 1))  # (num_patches,)
                
                if len(avg_patch_v) == w_featmap * h_featmap:
                    # Reshape to 2D and visualize
                    patch_v_2d = avg_patch_v.reshape(w_featmap, h_featmap)
                    
                    # Upscale to original image size
                    upscaled_patch_v = nn.functional.interpolate(
                        patch_v_2d.unsqueeze(0).unsqueeze(0),
                        size=(w, h),
                        mode="bilinear",
                        align_corners=False
                    )[0, 0]
                    
                    self.show_neural_feature_hotimg(
                        upscaled_patch_v,
                        head_subpath + f'patch_v_avg.png'
                    )
                
                # 5. Visualize value magnitude for top important patches
                if len(avg_patch_v) == w_featmap * h_featmap:
                    k = min(8, w_featmap * h_featmap)
                    top_patches = torch.topk(avg_patch_v.flatten(), k)
                    
                    for rank, patch_idx in enumerate(top_patches.indices):
                        patch_w = patch_idx.item() // h_featmap
                        patch_h = patch_idx.item() % h_featmap
                        patch_idx_full = patch_idx.item()
                        
                        if patch_idx_full < patch_v_magnitudes.shape[2]:
                            # Value magnitude map for this patch across all spatial positions
                            patch_v_map = patch_v_magnitudes[:, :, patch_idx_full]  # (H, W)
                            
                            self.show_neural_feature_hotimg(
                                patch_v_map,
                                head_subpath + f'top{rank}_patch_{patch_w}_{patch_h}_v_magnitude.png'
                            )
            
            # 6. Visualize individual value vector components (for detailed analysis)
            # This shows the contribution of different dimensions in the value vector
            if head_v.shape[3] > 0:  # head_dim > 0
                # Visualize first few dimensions of value vectors
                num_dims_to_vis = min(8, head_v.shape[3])
                for dim_idx in range(num_dims_to_vis):
                    # Get this dimension for CLS token across all spatial positions
                    cls_v_dim = head_v[:, :, 0, dim_idx].to('cpu')  # (H, W)
                    
                    self.show_neural_feature_hotimg(
                        cls_v_dim,
                        head_subpath + f'cls_v_dim{dim_idx}.png'
                    )
                    
                    # Also visualize average across all patches
                    mean_v_dim = torch.mean(head_v[:, :, 1:, dim_idx], dim=2).to('cpu')  # (H, W)
                    self.show_neural_feature_hotimg(
                        mean_v_dim,
                        head_subpath + f'mean_v_dim{dim_idx}.png'
                    )
        
        # Save the raw tensor if needed (disabled to save disk space)
        if saveTensor and False:  # Disabled to prevent disk full errors
            torch.save(input_v.to('cpu'), v_path + 'v_tensor.pt')
        
        return

    def visualize_attention(self, image_attn, input_attn, pic_layer):
        """
        Improved fine-grained attention visualization for ViT.
        Creates pixel-level attention maps similar to CNN feature visualizations.
        """
        name = str(pic_layer).rjust(2, '0')

        # Setup dimensions
        w, h = self.w, self.h  # Original image dimensions
        patch_size = self.patch_size
        w_featmap = max(1, w // patch_size)
        h_featmap = max(1, h // patch_size)
        
        # Create directories
        input_attn_path = self.scan_attn_map + name + '/input_attn/'
        if not os.path.exists(input_attn_path):
            os.makedirs(input_attn_path)

        # Save raw attention tensors
        if saveTensor:
            torch.save(image_attn.to('cpu'), self.scan_attn_map + name + '/image_attn_tensor.pt')
            torch.save(input_attn.to('cpu'), input_attn_path + 'input_attn_tensor.pt')

        # Fine-grained visualization for each attention head
        for h_idx in range(self.heads):
            head_path = input_attn_path + f'head{h_idx}/'
            if not os.path.exists(head_path):
                os.makedirs(head_path)
            
            try:
                # input_attn shape: (H, W, num_heads, num_patches+1, num_patches+1)
                # Extract CLS token attention to all patches (excluding CLS itself)
                cls_attn = input_attn[:, :, h_idx, 0, 1:].to('cpu')  # (H, W, num_patches)
                
                # Fine-grained visualization: Create attention map for each spatial position
                # This creates detailed pixel-level visualizations similar to CNN approach
                
                # 1. Average attention across all spatial positions (for overall view)
                # 改进：直接融合类别特征图，使其能看到鱼的形状
                avg_attn = cls_attn.mean(dim=(0,1))  # (num_patches,)
                
                # 尝试加载已保存的类别特征图（如果final_linear已经执行）
                # 或者使用注意力与类别特征的融合
                use_class_feature = False
                if self.class_feature_map is not None and self.predicted_class is not None:
                    # 获取预测类别的特征图 (H, W)
                    predicted_class_map = self.class_feature_map[self.predicted_class]  # (H, W)
                    
                    # 将类别特征图resize到与注意力相同的尺寸
                    if len(avg_attn) == w_featmap * h_featmap:
                        # 将类别特征图resize到patch级别
                        class_map_patch = nn.functional.interpolate(
                            predicted_class_map.unsqueeze(0).unsqueeze(0),
                            size=(w_featmap, h_featmap),
                            mode="bilinear",
                            align_corners=False
                        )[0, 0]  # (w_featmap, h_featmap)
                        
                        # 归一化类别特征图到[0, 1]
                        class_map_patch_norm = (class_map_patch - class_map_patch.min()) / (class_map_patch.max() - class_map_patch.min() + 1e-8)
                        
                        # 将类别特征图flatten并与注意力对齐
                        class_map_flat = class_map_patch_norm.flatten()  # (num_patches,)
                        
                        # 方法1：直接使用类别特征图（效果最好，能看到鱼的形状）
                        # 将类别特征图作为主要可视化，注意力作为辅助
                        alpha_class = 0.7  # 类别特征图的权重（主要）
                        alpha_attn = 0.3   # 注意力的权重（辅助）
                        
                        # 归一化注意力到[0, 1]
                        attn_norm = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)
                        
                        # 融合：类别特征图为主，注意力为辅
                        avg_attn_weighted = alpha_class * class_map_flat + alpha_attn * attn_norm
                        use_class_feature = True
                    else:
                        avg_attn_weighted = avg_attn
                elif self.class_weights is not None and len(self.class_weights) > 0:
                    # 如果类别特征图不可用，使用类别权重来加权
                    class_probs = torch.softmax(self.class_weights, dim=0)  # (10,)
                    predicted_weight = class_probs[self.predicted_class].item()
                    if predicted_weight > 0.3:
                        avg_attn_weighted = avg_attn * (1.0 + predicted_weight * 0.5)
                    else:
                        avg_attn_weighted = avg_attn
                else:
                    avg_attn_weighted = avg_attn
                
                # 对于深层，如果注意力分布太均匀，使用更激进的归一化来突出差异
                # 计算注意力值的标准差，如果太小说明分布太均匀
                attn_std = avg_attn_weighted.std().item()
                attn_mean = avg_attn_weighted.mean().item()
                
                if len(avg_attn_weighted) == w_featmap * h_featmap:
                    avg_attn_2d = avg_attn_weighted.reshape(w_featmap, h_featmap)  # (feat_w, feat_h)
                    
                    # 对于深层（注意力分布可能更均匀），使用更激进的归一化
                    # 计算变异系数（CV = std/mean），如果CV小说明分布均匀
                    cv = attn_std / (attn_mean + 1e-8)
                    
                    # 如果分布太均匀（CV < 0.1），使用百分位数归一化来突出差异
                    if cv < 0.1:
                        # 使用更激进的百分位数裁剪（1%-99%），突出主要差异
                        attn_np = avg_attn_2d.detach().cpu().numpy()
                        p_low = np.percentile(attn_np, 1)
                        p_high = np.percentile(attn_np, 99)
                        attn_clipped = np.clip(attn_np, p_low, p_high)
                        attn_range = p_high - p_low
                        if attn_range > 1e-8:
                            avg_attn_2d = torch.from_numpy((attn_clipped - p_low) / attn_range).float()
                        else:
                            avg_attn_2d = torch.from_numpy(attn_np).float()
                    
                    # 根据论文：使用max pooling和插值来提高精细度
                    # 1. 先对patch级别的注意力进行max pooling（增强重要区域）
                    # 2. 然后使用更好的插值方法上采样到原始图像尺寸
                    
                    # 方法1：使用max pooling增强重要区域（论文方法）
                    # 对patch级别的注意力图进行max pooling，突出重要区域
                    avg_attn_2d_pooled = avg_attn_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, w_featmap, h_featmap)
                    
                    # 使用adaptive max pooling来增强精细度（如果patch数量足够）
                    if w_featmap >= 4 and h_featmap >= 4:
                        # 使用小的kernel size进行max pooling，保持细节的同时增强重要区域
                        kernel_size = 2
                        avg_attn_2d_pooled = nn.functional.max_pool2d(
                            avg_attn_2d_pooled, 
                            kernel_size=kernel_size, 
                            stride=1, 
                            padding=kernel_size//2
                        )
                    
                    # 2. 使用更好的插值方法上采样到原始图像尺寸
                    # 使用bicubic插值（比bilinear更精细）来获得更好的细节
                    upscaled_avg_attn = nn.functional.interpolate(
                        avg_attn_2d_pooled, 
                        size=(w, h),
                        mode="bicubic",  # 使用bicubic插值，比bilinear更精细
                        align_corners=False
                    )[0, 0]  # (w, h)
                    
                    # 确保值在合理范围内
                    upscaled_avg_attn = torch.clamp(upscaled_avg_attn, 0, 1)
                    
                    self.show_neural_feature_hotimg(
                        upscaled_avg_attn,
                        head_path + f'avg_cls_attn_head{h_idx}.png'
                    )
                    
                    # 额外保存一个类别加权的注意力图（如果类别权重可用）
                    if self.class_weights is not None and len(self.class_weights) > 0:
                        # 创建一个类别引导的注意力图
                        # 使用类别特征图来增强注意力可视化
                        # 这里我们创建一个融合了类别信息的注意力图
                        class_probs = torch.softmax(self.class_weights, dim=0)
                        predicted_weight = class_probs[self.predicted_class].item()
                        
                        # 如果预测类别权重足够高，创建一个类别引导的注意力图
                        if predicted_weight > 0.3:
                            # 增强与类别相关的注意力区域
                            # 使用指数增强来突出重要区域
                            enhanced_attn_2d = avg_attn_2d ** (1.0 - predicted_weight * 0.3)  # 降低指数，增强高值区域
                            
                            upscaled_enhanced_attn = nn.functional.interpolate(
                                enhanced_attn_2d.unsqueeze(0).unsqueeze(0), 
                                size=(w, h),
                                mode="bilinear",
                                align_corners=False
                            )[0, 0]
                            
                            self.show_neural_feature_hotimg(
                                upscaled_enhanced_attn,
                                head_path + f'avg_cls_attn_head{h_idx}_class_guided.png'
                            )
                
                # 2. Fine-grained: Attention map for each spatial position (w, h)
                # This creates detailed visualizations showing how each pixel position attends to patches
                if cls_attn.shape[0] == w and cls_attn.shape[1] == h:
                    # For each spatial position, create a patch-level attention map
                    for spatial_idx in range(min(16, w * h)):  # Limit to avoid too many files
                        w_pos = spatial_idx // h
                        h_pos = spatial_idx % h
                        
                        pos_attn = cls_attn[w_pos, h_pos, :]  # (num_patches,)
                        
                        if len(pos_attn) == w_featmap * h_featmap:
                            pos_attn_2d = pos_attn.reshape(w_featmap, h_featmap)
                            
                            # Upscale to original size
                            upscaled_pos_attn = nn.functional.interpolate(
                                pos_attn_2d.unsqueeze(0).unsqueeze(0),
                                size=(w, h),
                                mode="bilinear",
                                align_corners=False
                            )[0, 0]
                            
                            self.show_neural_feature_hotimg(
                                upscaled_pos_attn,
                                head_path + f'pos_{w_pos}_{h_pos}_attn_head{h_idx}.png'
                            )
                
                # 3. Create per-patch attention visualization (how each patch is attended)
                # This shows the attention strength for each patch position
                if len(avg_attn) == w_featmap * h_featmap:
                    # For each patch, visualize how much attention it receives from all spatial positions
                    patch_attn_maps = torch.zeros(w_featmap, h_featmap, w, h)
                    for patch_w in range(w_featmap):
                        for patch_h in range(h_featmap):
                            patch_idx = patch_w * h_featmap + patch_h
                            if patch_idx < cls_attn.shape[2]:
                                # Get attention from all spatial positions to this patch
                                patch_attn = cls_attn[:, :, patch_idx]  # (W, H)
                                patch_attn_maps[patch_w, patch_h] = patch_attn
                    
                    # Visualize top patches that receive most attention
                    patch_importance = avg_attn_2d if len(avg_attn) == w_featmap * h_featmap else None
                    if patch_importance is not None:
                        # Get top-k most important patches
                        k = min(8, w_featmap * h_featmap)
                        top_patches = torch.topk(avg_attn.flatten(), k)
                        
                        for rank, patch_idx in enumerate(top_patches.indices):
                            patch_w = patch_idx.item() // h_featmap
                            patch_h = patch_idx.item() % h_featmap
                            
                            # Visualize attention map for this important patch
                            patch_attn_map = patch_attn_maps[patch_w, patch_h]  # (W, H)
                            
                            self.show_neural_feature_hotimg(
                                patch_attn_map,
                                head_path + f'top{rank}_patch_{patch_w}_{patch_h}_attn_head{h_idx}.png'
                            )
                
                # 4. Create attention flow visualization (how attention changes across patches)
                # Reshape cls_attn to show attention from each spatial position
                if cls_attn.shape[2] == w_featmap * h_featmap:
                    # Create a grid showing attention from each patch position
                    # This helps understand spatial attention patterns
                    for patch_idx in range(min(9, w_featmap * h_featmap)):  # Visualize first 9 patches
                        patch_w = patch_idx // h_featmap
                        patch_h = patch_idx % h_featmap
                        
                        # Attention from all spatial positions to this patch
                        patch_attn = cls_attn[:, :, patch_idx]  # (W, H)
                        
                        self.show_neural_feature_hotimg(
                            patch_attn,
                            head_path + f'patch_{patch_w}_{patch_h}_attn_head{h_idx}.png'
                        )
                
            except Exception as e:
                print(f"处理头{h_idx}时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        return 



    def saveOri(self,imageTensor):
        picdir = self.rootdir+'/tem0/'
        if not os.path.exists(picdir):
            os.makedirs(picdir)
        toPIL = transforms.ToPILImage()
        picI =  toPIL(imageTensor)
        picI.save(picdir+'0.png')
        torch.save(imageTensor,picdir+'0.pt')



    def saveDIR(self,imageTensor, inputTensor,layerIndex):
        name = str(layerIndex).rjust(2,'0')
        if savePicure:
            self.savePic(inputTensor,self.scan_path_pic+name+'/')
        if saveFeature:
            self.saveFea(imageTensor,self.scan_path_fea+name+'/')
        if saveTensor:
            self.saveTen(inputTensor,self.scan_path_ten+name+'/')
        if saveSumTen:
            self.channelPic(imageTensor,inputTensor,self.scan_path_sumten+name+'/')
    



    def savePic(self,t,filepath):
        if pic_RGB==True:
            self.savePic_RGB(t,filepath)
        else:
            self.savePic_gray(t,filepath)
                
            


    # ###save tensor
    def saveFea(self,t,filepath):
        t = t.to('cpu')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        torch.save(t,filepath+'tensor.pt')



    ###save tensor
    def saveTen(self,t,filepath):
        t = t.to('cpu')
        # t = t - t[0,0]
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            torch.save(t,filepath+'tensor.pt')
        



    ###save pic from tensor
    def savePic_gray(self,t,filepath):
        t = t.to('cpu')
        # t = t - t[0,0]
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        if t.dim() == 2:
            toPIL = transforms.ToPILImage()
            pic =  toPIL(t)
            name = str(0).rjust(4,'0')
            pic.save(filepath+'/'+name+'.png')
        if t.dim() == 3:
            for i  in range(t.shape[0]):
                toPIL = transforms.ToPILImage()
                pic =  toPIL(t[i]-t[i][0][0])
                name = str(i).rjust(4,'0')
                pic.save(filepath+'/'+name+'.png')
        if t.dim() == 4:
            for i  in range(t.shape[0]):
                path = filepath+'/num'+str(i)
                if not os.path.exists(path):
                    os.makedirs(path)
                for j in range(t.shape[1]):
                    toPIL = transforms.ToPILImage()
                    pic =  toPIL(t[i][j]-t[i][j][0][0])
                    name = '/channel'+str(j).rjust(2,'0')
                    pic.save(path+name+'.png')
        if t.dim() == 5:
            toPIL = transforms.ToPILImage()
            for c  in range(t.shape[2]):
                chan = '/channel'+str(c).rjust(2,'0')
                path1 = filepath+chan
                if not os.path.exists(path1):
                    os.makedirs(path1)
                size = t.shape[0]
                for n in range(size*size):
                    w = n // size
                    h = n % size                    
                    pic =  toPIL(t[w][h][c]-t[w][h][c][0][0])
                    name = str(n).rjust(4,'0')
                    pic.save(path1+'/'+name+'.png')
                



    ###save pic from tensor
    def savePic_RGB(self,t,filepath):
        t = t.to('cpu')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        if t.dim() == 2:
            image = t
            img_path = filepath+'.png'
            self.show_neural_feature_hotimg(image,img_path)
        if t.dim() == 3:
            # 如果是类别特征图（10个类别），使用全局归一化使颜色一致
            # 先计算所有类别的全局最小值和最大值
            global_min = t.min().item()
            global_max = t.max().item()
            global_range = global_max - global_min
            
            for i  in range(t.shape[0]):
                image = t[i]
                img_path = filepath+'/'+str(i)+'.png'
                
                # 使用全局范围归一化，使所有类别特征图颜色一致
                if global_range > 1e-8:
                    # 归一化到[0, 1]使用全局范围
                    image_normalized = (image - global_min) / global_range
                    # 调用show_neural_feature_hotimg，但传入已归一化的图像
                    # 需要修改show_neural_feature_hotimg来接受已归一化的图像，或者直接在这里处理
                    # 为了保持兼容性，我们创建一个包装函数
                    self.show_neural_feature_hotimg_uniform(image_normalized, img_path, global_min, global_max)
                else:
                    # 如果范围太小，使用原来的方法
                    self.show_neural_feature_hotimg(image,img_path)
        if t.dim() == 4:
            for i in range(t.shape[0]):                
                path = filepath+'/num'+str(i)
                if not os.path.exists(path):
                    os.makedirs(path)
                for j in range(t.shape[1]):
                    image = t[i][j]
                    img_path = path+'/channel'+str(j)+'.png'
                    self.show_neural_feature_hotimg(image,img_path)
        if t.dim() == 5:
            toPIL = transforms.ToPILImage()
            for c  in range(t.shape[2]):
                path1 = filepath+'/channel'+str(c)
                if not os.path.exists(path1):
                    os.makedirs(path1)
                for w in range(t.shape[0]):
                    path2 = path1 + '/width'+str(w)
                    if not os.path.exists(path2):
                        os.makedirs(path2)
                    for h in range(t.shape[1]):
                        image = t[w][h][c]
                        img_path = path2+'/height'+str(h)+'.png'
                        self.show_neural_feature_hotimg(image,img_path)


    # cv2
    def show_neural_feature_hotimg(self, image, path, title='', grayscale=False, ax=None, enhance_contrast=True):
        """
        Improved visualization function with enhanced contrast and better normalization.
        Creates fine-grained visualizations with clear distinction between attended and non-attended regions.
        根据论文方法：使用max pooling和更好的插值来提高精细度。
        """
        import cv2
        import numpy as np
        
        # 检查张量维度并正确处理
        if hasattr(image, 'detach'):
            if image.dim() == 1:
                image = image.unsqueeze(0)
            image_np = image.detach().cpu().numpy()
        else:
            image_np = np.array(image)
            if image_np.ndim == 1:
                image_np = image_np.unsqueeze(0)
        
        # 使用简单的min-max归一化，保留完整的动态范围
        # 不使用百分位数裁剪，避免过度压缩导致正方形块
        img_min = np.min(image_np)
        img_max = np.max(image_np)
        img_range = img_max - img_min
        
        if img_range > 1e-8:
            # 简单的线性归一化到[0, 1]
            image_np = (image_np - img_min) / img_range
        else:
            # 如果所有值相同
            if np.abs(img_min) < 1e-8:
                image_np = np.zeros_like(image_np)  # 全零
            else:
                image_np = np.ones_like(image_np) * 0.5  # 常数值
        
        # 确保值在[0, 1]范围内
        image_np = np.clip(image_np, 0, 1)
        
        # 对V向量数据进行强平滑处理，彻底消除网格效果
        if 'v_magnitude' in path or 'v_dim' in path:
            # 使用更强的高斯滤波 + 双边滤波组合
            from scipy import ndimage
            # 第一步：高斯平滑
            image_np = ndimage.gaussian_filter(image_np, sigma=1.0)
            # 第二步：中值滤波去除块状伪影
            image_np = ndimage.median_filter(image_np, size=3)
            # 第三步：再次轻微高斯平滑
            image_np = ndimage.gaussian_filter(image_np, sigma=0.3)
        
        # 转换为uint8
        image_uint8 = (image_np * 255).astype(np.uint8)
        
        # 如果图像太小，先放大（使用更好的插值方法）
        if image_uint8.shape[0] < 32 or image_uint8.shape[1] < 32:
            target_h = max(32, image_uint8.shape[0] * 4)
            target_w = max(32, image_uint8.shape[1] * 4)
            if 'v_magnitude' in path or 'v_dim' in path:
                # V向量使用双线性插值
                image_uint8 = cv2.resize(image_uint8, (target_w, target_h), 
                                         interpolation=cv2.INTER_LINEAR)
            else:
                # 注意力图使用LANCZOS4插值
                image_uint8 = cv2.resize(image_uint8, (target_w, target_h), 
                                         interpolation=cv2.INTER_LANCZOS4)
        
        if grayscale:
            heat_img = image_uint8
        else:
            # 根据用户需求选择颜色映射
            if 'v_magnitude' in path or 'v_dim' in path:
                # V向量使用蓝白配色（更清爽）
                heat_img = cv2.applyColorMap(image_uint8, cv2.COLORMAP_WINTER)
            else:
                # 注意力图也使用蓝白配色
                heat_img = cv2.applyColorMap(image_uint8, cv2.COLORMAP_WINTER)
        
        # 最终尺寸 - 对V向量使用更平滑的插值
        target_size = (512, 512)
        if heat_img.shape[:2] != target_size:
            if 'v_magnitude' in path or 'v_dim' in path:
                # V向量使用双线性插值，更平滑
                heat_img = cv2.resize(heat_img, target_size, interpolation=cv2.INTER_LINEAR)
            else:
                # 注意力图使用LANCZOS4插值，保持细节
                heat_img = cv2.resize(heat_img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        cv2.imwrite(path, heat_img)
    
    def show_neural_feature_hotimg_uniform(self, image_normalized, path, global_min, global_max):
        """
        使用已归一化的图像（使用全局范围）来生成热力图，确保颜色一致
        image_normalized: 已经使用全局范围归一化到[0, 1]的图像
        """
        import cv2
        import numpy as np
        
        # image_normalized已经是[0, 1]范围
        image_np = image_normalized.detach().cpu().numpy() if hasattr(image_normalized, 'detach') else image_normalized
        
        # 确保值在[0, 1]范围内
        image_np = np.clip(image_np, 0, 1)
        
        # 转换为uint8
        image = (image_np * 255).astype(np.uint8)
        
        # 如果图像太小，先放大
        if image.shape[0] < 32 or image.shape[1] < 32:
            image = cv2.resize(image, (max(32, image.shape[1] * 4), max(32, image.shape[0] * 4)), 
                             interpolation=cv2.INTER_LINEAR)
        
        # 使用JET colormap（红-黄-绿-蓝）
        heat_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        
        # 最终尺寸
        target_size = (512, 512)
        heat_img = cv2.resize(heat_img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        cv2.imwrite(path, heat_img)
                   
         
    def channelPic(self,imagetensor,inputtensor,filepath):
        """
        Enhanced channelPic function for ViT.
        Creates fine-grained feature aggregations similar to CNN approach.
        This function aggregates learned weights with scanned input tensors
        to show pixel-level contributions.
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Handle different tensor dimensions for different layers
        if inputtensor.dim() == 3:
            # Case: (C, H, W) - e.g., final linear layer output
            # imagetensor: (1, C) or (C,)
            # inputtensor: (C, H, W)
            w_tensor = torch.squeeze(imagetensor)  # (C,)
            if w_tensor.dim() == 0:
                w_tensor = w_tensor.unsqueeze(0)
            
            i_tensor = inputtensor.permute(1,2,0)  # (H, W, C)
            img = w_tensor * i_tensor  # (H, W, C)
            img = img.permute(2,0,1)  # (C, H, W)
            img = img.sum(dim=0)  # (H, W)
            torch.save(img, filepath+'tensor.pt')
            
            # Also save visualization
            if savePicure:
                self.show_neural_feature_hotimg(img, filepath+'aggregated.png')

        elif inputtensor.dim() == 5:
            # Case: (H, W, C, H_out, W_out) - e.g., after attention layers
            # imagetensor: (1, num_tokens, embed_dim) or similar
            # inputtensor: (num_tokens, embed_dim, H, W)
            
            # Flatten spatial dimensions if needed
            if imagetensor.dim() == 3:  # (1, num_tokens, embed_dim)
                w_tensor = imagetensor.squeeze(0)  # (num_tokens, embed_dim)
            else:
                w_tensor = imagetensor.squeeze()
            
            # Reshape for processing
            if w_tensor.dim() == 2:
                # (num_tokens, embed_dim)
                num_tokens, embed_dim = w_tensor.shape
                
                # inputtensor: (num_tokens, embed_dim, H, W)
                if inputtensor.shape[0] == num_tokens and inputtensor.shape[1] == embed_dim:
                    # For each token, aggregate its contribution
                    saveimg = torch.zeros(embed_dim, inputtensor.shape[2], inputtensor.shape[3])
                    
                    for token_idx in range(num_tokens):
                        # Weight for this token
                        token_weight = w_tensor[token_idx]  # (embed_dim,)
                        # Input tensor for this token
                        token_input = inputtensor[token_idx]  # (embed_dim, H, W)
                        
                        # Weighted contribution
                        weighted_contrib = token_weight.unsqueeze(1).unsqueeze(2) * token_input  # (embed_dim, H, W)
                        saveimg += weighted_contrib
                    
                    torch.save(saveimg, filepath+'tensor.pt')
                    
                    # Also save per-dimension visualizations for important dimensions
                    if savePicure:
                        # Save aggregated visualization
                        aggregated = saveimg.sum(dim=0)  # (H, W)
                        self.show_neural_feature_hotimg(aggregated, filepath+'aggregated.png')
                        
                        # Save top-k most important dimensions
                        dim_importance = saveimg.abs().sum(dim=(1,2))  # (embed_dim,)
                        k = min(8, embed_dim)
                        top_dims = torch.topk(dim_importance, k)
                        
                        for rank, dim_idx in enumerate(top_dims.indices):
                            dim_img = saveimg[dim_idx.item()]  # (H, W)
                            self.show_neural_feature_hotimg(
                                dim_img,
                                filepath+f'top{rank}_dim{dim_idx.item()}.png'
                            )
            
            else:
                # Fallback to original implementation
                w_tensor = imagetensor.squeeze().flatten(1,2) if imagetensor.dim() > 2 else imagetensor.squeeze()
                i_tensor = inputtensor.permute(2,0,1,3,4).flatten(1,2)
                saveimg = torch.zeros(w_tensor.shape[0], i_tensor.shape[2], i_tensor.shape[3])
                
                for i in range(w_tensor.shape[0]):
                    img = w_tensor[i] * i_tensor[i].permute(1,2,0)
                    img = img.permute(2,0,1)
                    img = img.sum(dim=0)
                    saveimg[i] = img
                
                torch.save(saveimg, filepath+'tensor.pt')
                
                if savePicure:
                    # Save aggregated and per-channel visualizations
                    aggregated = saveimg.sum(dim=0)
                    self.show_neural_feature_hotimg(aggregated, filepath+'aggregated.png')
                    
                    for ch_idx in range(min(8, saveimg.shape[0])):
                        self.show_neural_feature_hotimg(
                            saveimg[ch_idx],
                            filepath+f'channel{ch_idx}.png'
                        )
                

    ### padding tensor
    def padTensor(self,t,num):
        if num == 0 :
            return t
        new = torch.zeros(t.shape[0]+num*2,t.shape[1]+num*2,t.shape[2],t.shape[3],t.shape[4]).to(self.device) #t.dim() == 5
        for i in range(new.shape[0]-num):
            for j in range(new.shape[1]-num):
                if i!=0 and j !=0 :
                    new[i][j] = t[i-num][j-num]
        return new

    ### padding image tensor
    def padImageTensor(self,t,num):
        if num == 0 :
            return t
        new = torch.zeros(t.shape[0],t.shape[1],t.shape[2]+num*2,t.shape[3]+num*2).to(self.device) #t.dim() == 4
        for x in range(new.shape[0]):
            for y in range(new.shape[1]):
                for i in range(new.shape[2]-num):
                    for j in range(new.shape[3]-num):
                        if i!=0 and j!=0 :
                            new[x][y][i][j] = t[x][y][i-num][j-num]
        return new
    
    
        
    # @torch.no_grad() 
    # def ReLU_pytorch(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
    #     #(16,128),(16,128,28,28)
    #     mask = imagetensor<0
    #     imagetensor = layer(imagetensor) #(16,128)
    #     inputtensor = inputtensor.permute(2,3,0,1) #(28,28,16,128)
    #     size = inputtensor.size()#(28,28,16,128)
    #     for n1 in range(size[0]):
    #         for n2 in range(size[1]):
    #             inputtensor[n1][n2] = inputtensor[n1][n2]*mask
    #     inputtensor = inputtensor.permute(2,3,0,1) #(16,128,28,28)
    #     return imagetensor, inputtensor
        
    @torch.no_grad() 
    def ReLU_pytorch(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
        imagetensor = layer(imagetensor) #(1,64,32,32)-->(1,64,32,32)  #ReLU(inplace=True)

        if imagetensor.dim()==2:
            for i in range(imagetensor.shape[0]):
                if imagetensor[0][i] == 0:
                    inputtensor[i].data *= 0
        else :
            shape = imagetensor.shape
            for n in range(shape[0]):
                for c in range(shape[1]):
                    for w in range(shape[2]):
                        for h in range(shape[3]):
                            if imagetensor[n][c][w][h] == 0:
                                inputtensor[w][h][c]*=0 #(32,32,64,32,32)
        
        return imagetensor,inputtensor #(1,64,32,32),(32,32,64,32,32)
   
        
    @torch.no_grad() 
    def linear_pytorch(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
        #(16,49), (16,49,28,28)
        imagetensor = layer(imagetensor)#(16,49)->(16,64)
        inputtensor = inputtensor.permute(2,3,0,1)
        size = inputtensor.size()#(28,28,16,49)
        newtensor = torch.zeros((size[0],size[1],size[2],imagetensor.shape[-1])).to(self.device)#(28,28,16,64)
        for n1 in range(size[0]):
            for n2 in range(size[1]):
                newtensor[n1][n2] = layer(inputtensor[n1][n2].to(self.device))#(16,49)->(16,64)
        
        newtensor = newtensor.permute(2,3,0,1)
        return imagetensor,newtensor#(16,64),(16,64,28,28)
    
    
    @torch.no_grad() 
    def final_linear_pytorch(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
        #(64),(64,28,28)
        imagetensor = layer(imagetensor)#(64)->(10)
        inputtensor = inputtensor.permute(1,2,0) #(28,28,64)
        size = inputtensor.size()
        newtensor = torch.zeros((size[0],size[1],imagetensor.shape[-1])).to(self.device)#(28,28,10)
        for n1 in range(size[0]):
            for n2 in range(size[1]):
                newtensor[n1][n2] = layer(inputtensor[n1][n2].to(self.device))#(64)->(10)
        #(28,28,10)
        newtensor = newtensor.permute(2,0,1)
        #(10,28,28)
        
        # 保存类别权重和预测类别，用于改进注意力可视化
        self.class_weights = imagetensor.detach().cpu()  # (10,) 类别logits
        self.predicted_class = torch.argmax(imagetensor).item()  # 预测的类别索引
        self.class_feature_map = newtensor.detach().cpu()  # (10, H, W) 类别特征图
        
        # 现在类别特征图已可用，重新可视化所有注意力层（使用类别特征图改进）
        # 重新处理所有已保存的注意力tensor
        if hasattr(self, 'scan_path_attn') and os.path.exists(self.scan_path_attn):
            # 查找所有已保存的注意力层
            for layer_dir in os.listdir(self.scan_path_attn):
                layer_path = os.path.join(self.scan_path_attn, layer_dir)
                if os.path.isdir(layer_path) and layer_dir.isdigit():
                    try:
                        pic_layer = int(layer_dir)
                        # 尝试加载已保存的注意力tensor
                        attn_tensor_path = os.path.join(self.scan_path_attn, layer_dir, 'tensor.pt')
                        input_attn_path = os.path.join(self.scan_attn_map, layer_dir, 'input_attn_tensor.pt')
                        
                        if os.path.exists(attn_tensor_path) and os.path.exists(input_attn_path):
                            image_attn = torch.load(attn_tensor_path)
                            input_attn = torch.load(input_attn_path)
                            # 重新可视化，这次使用类别特征图
                            self.visualize_attention(image_attn, input_attn, pic_layer)
                    except Exception as e:
                        print(f"  [WARN] 重新可视化层 {layer_dir} 时出错: {e}")
                        continue
        
        return imagetensor,newtensor#(10),(10,28,28)
        
        
    #使用学习图像的q,k计算学习图像的权重
    @torch.no_grad() 
    def attention_pytorch(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
        #(1,16,64),(16,64,28,28)
        
        layer_attend,layer_toqkv,layer_toout = layer.children()
        
        inputtensorx = inputtensor.permute(2,3,0,1) #(28,28,16,64)
        size = inputtensorx.size()#(28,28,16,64)
        
        #(16,192),(16,192,28,28)
        image_qkv,input_qkv = self.linear_pytorch(layer_toqkv,imagetensor.squeeze(),inputtensor)
        image_qkv = torch.unsqueeze(image_qkv, 0)#(1,16,192)
        image_qkv = image_qkv.chunk(3, dim = -1)
        input_qkv = input_qkv.permute(2,3,0,1) #(28,28,16,192)
        input_qkv = input_qkv.chunk(3, dim = -1)
        
        ###imagetensor
        #(1,1,16,64),(1,1,16,64),(1,1,16,64)
        image_q,image_v,image_k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), image_qkv)
        #image_q:[1,1,16,64]
        #image_k.transpose(-1, -2):[1,1,64,16]
        #image_dots:[1,1,16,16]
        image_dots = torch.matmul(image_q, image_k.transpose(-1, -2)) * self.scale
        image_attn = layer_attend(image_dots)
        # print(image_attn)
        image_out = torch.matmul(image_attn, image_v)
        image_out = rearrange(image_out, 'b h n d -> b n (h d)')
        newimagetensor = layer_toout(image_out)#(1,16,64)
        
        ###inputtensor
        #(28,28,1,16,64),(28,28,1,16,64),(28,28,1,16,64)
        input_q,input_k,input_v = map(lambda t: rearrange(t, 'n1 n2 n (h d) -> n1 n2 h n d', h = self.heads), input_qkv)
        #input_q:[28,28,1,16,64]
        #input_k.transpose(-1, -2):[28,28,1,64,16]
        #input_dots:[28,28,1,16,16]
        input_dots = torch.matmul(input_q, input_k.transpose(-1, -2)) * self.scale
        input_attn = layer_attend(input_dots)
        # print(input_attn)
        input_out = torch.matmul(input_attn, input_v)#(28,28,1,16,64)
        input_out = rearrange(input_out, 'n1 n2 h n d -> n1 n2 n (h d)')
        newinputtensor = layer_toout(input_out)#(28,28,16,64)  
             
        #(1,16,64),(28,28,16,64) 
        newinputtensor = newinputtensor.permute(2,3,0,1)
        #(1,16,64),(16,64,28,28) 
        return newimagetensor,newinputtensor
    
    
    #使用原始图像的q,k计算学习图像的权重
    @torch.no_grad() 
    def attention_pytorch_1(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
        #(1,16,64),(16,64,28,28)
        
        layer_attend,layer_toqkv,layer_toout = layer.children()
        
        inputtensorx = inputtensor.permute(2,3,0,1) #(28,28,16,64)
        size = inputtensorx.size()#(28,28,16,64)
        
        #(16,192),(16,192,28,28)
        image_qkv,input_qkv = self.linear_pytorch(layer_toqkv,imagetensor.squeeze(),inputtensor)
        image_qkv = torch.unsqueeze(image_qkv, 0)#(1,16,192)
        image_qkv = image_qkv.chunk(3, dim = -1)
        input_qkv = input_qkv.permute(2,3,0,1) #(28,28,16,192)
        input_qkv = input_qkv.chunk(3, dim = -1)
        
        ###imagetensor
        #(1,1,16,64),(1,1,16,64),(1,1,16,64)
        image_q,image_v,image_k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), image_qkv)
        #image_q:[1,1,16,64]
        #image_k.transpose(-1, -2):[1,1,64,16]
        #image_dots:[1,1,16,16]
        image_dots = torch.matmul(image_q, image_k.transpose(-1, -2)) * self.scale
        image_attn = layer_attend(image_dots)
        # print(image_attn)
        image_out = torch.matmul(image_attn, image_v)
        image_out = rearrange(image_out, 'b h n d -> b n (h d)')
        newimagetensor = layer_toout(image_out)#(1,16,64)
        
        ###inputtensor
        #(28,28,1,16,64),(28,28,1,16,64),(28,28,1,16,64)
        input_q,input_k,input_v = map(lambda t: rearrange(t, 'n1 n2 n (h d) -> n1 n2 h n d', h = self.heads), input_qkv)
        #image_attn:[1,1,16,16]
        #直接用image_attn作为权重
        input_out = torch.matmul(image_attn, input_v)#(28,28,1,16,64)
        input_out = rearrange(input_out, 'n1 n2 h n d -> n1 n2 n (h d)')
        newinputtensor = layer_toout(input_out)#(28,28,16,64)  
             
        #(1,16,64),(28,28,16,64) 
        newinputtensor = newinputtensor.permute(2,3,0,1)
        #(1,16,64),(16,64,28,28) 
        return newimagetensor,newinputtensor
    

    ### new attention by pytorch 
    ### return: x, attn
    @torch.no_grad() 
    def new_attention_pytorch(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
        
        #(1,257,192),(257,192,64,64)
        '''
        Attention(
        (qkv): Linear(in_features=192, out_features=576, bias=False)
        (proj): Linear(in_features=192, out_features=192, bias=False)
        )
        '''
        # timm模型使用named_children来访问qkv和proj
        # 尝试通过named_children获取（timm方式）
        named_dict = dict(layer.named_children())
        if 'qkv' in named_dict and 'proj' in named_dict:
            layer_qkv = named_dict['qkv']
            layer_proj = named_dict['proj']
            print(f"  [INFO] 使用timm方式获取qkv和proj")
        else:
            # 如果不是timm，尝试从children获取（自定义ViT方式）
            children_list = list(layer.children())
            if len(children_list) == 2:
                layer_qkv, layer_proj = children_list
                print(f"  [INFO] 使用自定义ViT方式获取qkv和proj")
            else:
                # 作为最后手段，查找Linear层
                layer_qkv = None
                layer_proj = None
                for child in children_list:
                    if type(child).__name__ == 'Linear':
                        if layer_qkv is None:
                            layer_qkv = child
                        else:
                            layer_proj = child
                            break
                
                if layer_qkv is None or layer_proj is None:
                    raise ValueError(f"无法从Attention层找到qkv和proj层。Children: {[type(c).__name__ for c in children_list]}, Named: {list(named_dict.keys())}")
        
        
        size_image = imagetensor.size()#(1,65,192)
        size_input = inputtensor.size()#(65,192,32,32)
        
        '''
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        '''
        image_qkv,input_qkv = self.linear_pytorch(layer_qkv,imagetensor.squeeze(),inputtensor)
        # qkv: (3,1,1,257,192) #(3,batch_size,num_heads,257,head_dim)
        # q or k or v: (1,1,257,192)
        image_qkv = torch.unsqueeze(image_qkv, 0)#(1,257,768)
        image_qkv = image_qkv.chunk(3, dim = -1)
        input_qkv = input_qkv.permute(2,3,0,1) #(64,64,257,768)
        input_qkv = input_qkv.chunk(3, dim = -1)
        
        # image_qkv: (1,65,192)
        # input_qkv: (32,32,65,192)
        
        #(1,3,65,64),(1,3,65,64),(1,3,65,64) 
        #(batch_size, num_heads, num_patches+1, head_dim)
        image_q,image_v,image_k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), image_qkv)
        #(32,32,3,65,64),(32,32,3,65,64),(32,32,3,65,64)
        #(w, h, num_heads, num_patches+1, head_dim)
        input_q,input_k,input_v = map(lambda t: rearrange(t, 'n1 n2 n (h d) -> n1 n2 h n d', h = self.heads), input_qkv)
        
        '''
        # q: (1,1,257,192)
        # k.transpose(-2, -1): (1,1,192,257)
        attn = (q @ k.transpose(-2, -1)) * self.scale # attn: (1,1,257,257)
        attn = attn.softmax(dim=-1) # attn: (1,1,257,257)
        '''
        image_attn = (image_q @ image_k.transpose(-2, -1)) * self.scale
        image_attn = image_attn.softmax(dim=-1) # (1,3,65,65) # (batch_size, num_heads, num_patches+1, num_patches+1)
        input_attn = torch.matmul(input_q, input_k.transpose(-1, -2)) * self.scale
        input_attn = input_attn.softmax(dim=-1) # (32,32,3,65,65) # (w, h, batch_size, num_patches+1, num_patches+1)
        
        '''
        # attn @ v: (1,1,257,192)
        # (attn @ v).transpose(1, 2): (1,257,1,192)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # x: (1,257,192)
        x = self.proj(x) # x: (1,257,192)
        return x, attn # x: (1,257,192), attn: (1,1,257,257)
        '''
        # Notice: 这里是用image_attn作为权重还是input_attn作为权重？
        image_out = torch.matmul(image_attn, image_v)
        image_out = rearrange(image_out, 'b h n d -> b n (h d)')
        newimagetensor = layer_proj(image_out)#(1,65,192)
        input_out = torch.matmul(input_attn, input_v)#(64,64,1,257,192)
        input_out = rearrange(input_out, 'n1 n2 h n d -> n1 n2 n (h d)')
        newinputtensor = layer_proj(input_out)#(65,192,32,32)  
        
        #(1,257,192),(64,64,257,192)  
        newinputtensor = newinputtensor.permute(2,3,0,1)
        #(1,257,192),(257,192,64,64)  
        
        
        return newimagetensor,image_attn,newinputtensor,input_attn,input_v
    
    
    

    ### conv by pytorch
    @torch.no_grad() 
    def conv_pytorch(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
        # layer: Conv2d(1, 49, kernel_size=(7, 7), stride=(7, 7))
        # CNN中设stride为(1, 1), 但ViT的stride为(7, 7)，与其kernel_size相同
        # 不知道是否有影响
        # 好像没影响
        
        # Conv2d(3, 192, kernel_size=(16, 16), stride=(16, 16))
        
        imagetensor = layer(imagetensor)#(1,3,224,224)-->(1,192,14,14)

        size = inputtensor.size() #(224,224,3,224,224)
        #翻转维度，并将扫描图压缩至一维，移入batch维度
        inputtensor = inputtensor.permute(3,4,2,0,1) #(224,224,3,224,224)
                                                                      
        inputtensor = inputtensor.reshape(-1,size[2],size[0],size[1])#(50176,3,224,224)

        inputtensor1 = layer(inputtensor)#(50176,3,224,224)-->(50176,192,14,14)

        #还原
        inputtensor1 = inputtensor1.reshape(size[3],size[4],-1,imagetensor.size(2),imagetensor.size(3)) #(224,224,192,14,14)
        inputtensor1 = inputtensor1.permute(3,4,2,0,1) #(14,14,192,224,224)


        return imagetensor,inputtensor1 #(1,192,14,14), (14,14,192,224,224)


    def getFirstLayer(self,imageTensor):
        
        if  imageTensor.dim() == 2:
            newtensor = torch.Tensor(imageTensor.shape[0],imageTensor.shape[1],1,imageTensor.shape[0],imageTensor.shape[1])
            for i in range(imageTensor.shape[0]):
                for j in range(imageTensor.shape[1]):
                    newtensor[i][j][0] = 0
                    newtensor[i][j][0][i][j] = imageTensor[i][j]
        if imageTensor.dim() == 3:
            newtensor = torch.Tensor(imageTensor.shape[1],imageTensor.shape[2],imageTensor.shape[0],imageTensor.shape[1],imageTensor.shape[2])
            for i in range(imageTensor.shape[0]):
                for j in range(imageTensor.shape[1]):
                    for k in range(imageTensor.shape[2]):
                        newtensor[j][k][i] = 0
                        newtensor[j][k][i][j][k] = imageTensor[i][j][k]
        return newtensor  



