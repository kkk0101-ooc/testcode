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

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)



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
        # self.cls_token = params['cls_token']
        self.pos_embed = params['pos_embed']
        self.embed_dim = self.pos_embed.shape[-1]
        self.scale = self.embed_dim ** -0.5
        
        
        self.rootdir = '/nas/scan_vit_cnn/'+str(model_type)+'_'+str(dataset)+'/'+str(model_name)+'/'+ str(id) + '_'+str(img_name)+'/'

        if pic_RGB:
            self.scan_path_pic = self.rootdir+'scan_RGB/'
        else:
            self.scan_path_pic = self.rootdir+'scan_gray/'
        self.scan_path_ten = self.rootdir+'scan_ten/'
        self.scan_path_fea = self.rootdir+'scan_fea/'
        self.scan_path_sumten = self.rootdir+'scan_sumten/'
        self.scan_path_ori_pic = self.rootdir+'scan_ori/'
        self.scan_attn_map = self.rootdir+'scan_attn_map/'
        

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
                        
                        # # add the [CLS] token
                        # cls_tokens = self.cls_token.expand(1, -1, -1) # cls_tokens:(1,1,192)
                        # imageTensor = torch.cat((cls_tokens, imageTensor), dim=1) # imageTensor: (1,197,192)
                        #                                                           # (1,257,192)
                        # add positional encoding to each token
                        imageTensor = imageTensor + self.pos_embed # imageTensor: (1,64,192)
                                                                   # (1,64,192)
                        
                        inputTensor = inputTensor.permute(2,3,0,1) #(32,32,64,192)
                        # # Notice,这里的64,64是img_size                                         
                        # input_cls_tokens = self.cls_token.expand(self.w, self.h, -1, -1) # cls_tokens:(224,224,1,192)
                        #                                                          # (64,64,1,192)
                        # inputTensor = torch.cat((input_cls_tokens, inputTensor), dim=2) #(224,224,197,192)
                        #                                                                 #(64,64,257,192)
                        
                        input_pos_embed = self.pos_embed.expand(self.w, self.h, -1, -1) #(32,32,64,192)
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
            if layerName == 'ModuleList':
                layerIndex += 1
                if layerIndex > layernum:
                    break
                
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
                                
                                
                                #(1,16,64),(16,64,28,28)
                                #(1,257,192),(257,192,64,64)
                                imageTensor, image_attn, inputTensor, input_attn = self.new_attention_pytorch(sublayer2,imageTensor,inputTensor)
                                #(1,16,64),(16,64,28,28)
                                #(1,257,192),(1,1,257,257),(257,192,64,64),(64,64,1,257,257)
                                #(1,65,192),(1,1,65,65),(65,192,32,32),(32,32,1,65,65)
                                
                                ### Residual Add
                                inputTensor = inputTensor + originalInputTensor
                                imageTensor = imageTensor + originalImageTensor
                                #(1,257,192),(257,192,64,64)
                                #(1,65,192),(65,192,32,32)
                                
                                # self.saveDIR(imageTensor, inputTensor,pic_layer)
                                
                                self.visualize_attention(image_attn,input_attn,pic_layer)
                                    
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

                                    
                #(1,257,192),(257,192,64,64)
                
                #cls操作
                # imageTensor = imageTensor[:,0]#(1,192)
                # inputTensor = inputTensor[0,:]#(192,64,64)
                #(1,64),(64,28,28)
                #(1,192),(192,64,64)
                
                #mean操作
                imageTensor = imageTensor.mean(dim = 1)
                inputTensor = inputTensor.mean(dim = 0)
                #(1,192),(192,64,64)
                
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


    # image_attn, input_attn
    # (1,1,257,257),(64,64,1,257,257)
    # (1,1,65,65),(32,32,1,65,65)
    def visualize_attention(self, image_attn, input_attn, pic_layer):
        name = str(pic_layer).rjust(2,'0')
        
        # make the image divisible by the patch size
        w, h = self.w, self.h # 32,32
        patch_size = self.patch_size # 4
        device = self.device

        # img = img[:, :w, :h].unsqueeze(0)
        # print('img.shape:',img.shape) # (1,3,640,960)

        w_featmap = w // patch_size # 8
        h_featmap = h // patch_size # 8

        # print('w_featmap:',w_featmap) # 80
        # print('h_featmap:',h_featmap) # 120

        # attentions = model.get_last_selfattention(img.to(device))  # (batch_size, num_heads, num_patches+1, num_patches+1)
        #                                 # (1, 6, 9601, 9601)
        #                                 # 9601 = 9600 + 1, 9600 = 80 * 120
        image_attentions = image_attn#(1,1,65,65)
        
        input_attentions = input_attn#(32,32,1,65,65)
        
        # print('attentions.shape:',attentions.shape) # (1, 6, 9601, 9601)
        
        
        img_attn_before_path = self.scan_attn_map + name +'/' 
        if not os.path.exists(img_attn_before_path):
            os.makedirs(img_attn_before_path)
        image_attentions_before = image_attentions.to('cpu')
        image_attentions_before = image_attentions_before[0, 0, :, :]
        self.show_neural_feature_hotimg(image_attentions_before, img_attn_before_path+ 'image_attn_before.png')
        

        nh = image_attentions.shape[1]  # number of head # 1
        '''
        # print('nh:',nh) # 6

        # keep only the output patch attention
        # 只保留 CLS token 对所有 patch 的注意力（去掉 CLS 对 CLS 的关注）
        # print('attentions[0, :, 0, 1:].shape:', attentions[0, :, 0, 1:].shape) # (6, 9600)
        # attentions = attentions[0, :, 0, 1:].reshape(nh, -1) # (6, 9600)
        # print('attentions.shape:',attentions.shape)

        image_attentions = image_attentions[0, :, 0, 1:].reshape(nh, -1) # (1, 64)
        input_attentions = input_attentions[:, :, 0, 0, 1:] # (32,32,64)

        # attentions = attentions.reshape(nh, w_featmap, h_featmap) # (6, 80, 120)
        # # 插值放大到原始图像大小
        # attentions = nn.functional.interpolate(attentions.unsqueeze(
        #     0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

        # print('attentions.shape:',attentions.shape) # (6, 640, 960)
        
        image_attentions = image_attentions.reshape(nh, w_featmap, h_featmap) # (1, 8, 8)
        image_attentions = nn.functional.interpolate(image_attentions.unsqueeze(
            0), scale_factor=patch_size, mode="nearest")[0]#(1,32,32)
     
        
        
        img_attn_path = self.scan_attn_map + name +'/' 
        if not os.path.exists(img_attn_path):
            os.makedirs(img_attn_path)
        image_attentions = image_attentions.to('cpu')
        self.show_neural_feature_hotimg(image_attentions[0], img_attn_path+ 'image_attn_after.png')
        
        input_attentions = input_attentions.permute(2,0,1) #(32,32,64)-->(64,32,32)
        input_attentions = input_attentions.to('cpu')
        
        input_attn_path = self.scan_attn_map + name +'/input_attn/'
        if not os.path.exists(input_attn_path):
            os.makedirs(input_attn_path)
        for i  in range(input_attentions.shape[0]):
            image = input_attentions[i]
            img_path = input_attn_path+str(i)+'.png'
            self.show_neural_feature_hotimg(image, img_path)
        '''

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
            for i  in range(t.shape[0]):
                image = t[i]
                img_path = filepath+'/'+str(i)+'.png'
                self.show_neural_feature_hotimg(image,img_path)
        if t.dim() == 4:
            # Notice: 这里-50是为了减小GPU占用
            # 不能减小GPU占用，只能减少运行时间
            # for i in range(t.shape[0] - 50):  
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
    def show_neural_feature_hotimg(self,image, path,title='', grayscale = False, ax=None):
        image = image - image[0,0]  #minus mix
        image = image.detach().numpy()
        image = image*255
        image = image.astype(np.uint8)
        if grayscale:
            heat_img = image
        if not grayscale:
            heat_img = cv2.applyColorMap(image,cv2.COLORMAP_JET)

        heat_img=cv2.resize(heat_img,(280,280),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(path,heat_img)
        # print('********************')
                   
         
    def channelPic(self,imagetensor,inputtensor,filepath):

        if not os.path.exists(filepath):
                os.makedirs(filepath)

        if inputtensor.dim() == 3:#(4096,32,32)
            w_tensor = torch.squeeze(imagetensor)#(4096,)
            i_tensor = inputtensor.permute(1,2,0)#(32,32,4096)
            img = w_tensor * i_tensor#(32,32,4096)
            img = img.permute(2,0,1)#(4096,32,32)
            img = img.sum(dim=0)#(32,32)
            torch.save(img,filepath+'tensor.pt')

        if inputtensor.dim() == 5:
            # saveimg = []
            w_tensor = imagetensor.squeeze().flatten(1,2)#(1,64,32,32)-->(64,1024)
            i_tensor = inputtensor.permute(2,0,1,3,4).flatten(1,2)#(32,32,64,32,32)-->(64,1024,32,32)
            saveimg = torch.zeros(w_tensor.shape[0],i_tensor.shape[2],i_tensor.shape[3])#(64,32,32)
            for i in range(w_tensor.shape[0]):
                img = w_tensor[i] * i_tensor[i].permute(1,2,0)#(32,32,1024)
                img = img.permute(2,0,1)#(1024,32,32)
                img = img.sum(dim=0)#(32,32)
                # saveimg.append(img) 
                saveimg[i] = img
            
            torch.save(saveimg,filepath+'tensor.pt')#(64,32,32)
                

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
    
    
        
    @torch.no_grad() 
    def ReLU_pytorch(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
        #(16,128),(16,128,28,28)
        mask = imagetensor<0
        imagetensor = layer(imagetensor) #(16,128)
        inputtensor = inputtensor.permute(2,3,0,1) #(28,28,16,128)
        size = inputtensor.size()#(28,28,16,128)
        for n1 in range(size[0]):
            for n2 in range(size[1]):
                inputtensor[n1][n2] = inputtensor[n1][n2]*mask
        inputtensor = inputtensor.permute(2,3,0,1) #(16,128,28,28)
        return imagetensor, inputtensor
        
    # @torch.no_grad() 
    # def ReLU_pytorch(self,layer,imagetensor:torch.FloatTensor,inputtensor:torch.FloatTensor) -> Tuple[torch.FloatTensor,torch.FloatTensor]:
    #     imagetensor = layer(imagetensor) #(1,64,32,32)-->(1,64,32,32)  #ReLU(inplace=True)

    #     if imagetensor.dim()==2:
    #         for i in range(imagetensor.shape[0]):
    #             if imagetensor[0][i] == 0:
    #                 inputtensor[i].data *= 0
    #     else :
    #         shape = imagetensor.shape
    #         for n in range(shape[0]):
    #             for c in range(shape[1]):
    #                 for w in range(shape[2]):
    #                     for h in range(shape[3]):
    #                         if imagetensor[n][c][w][h] == 0:
    #                             inputtensor[w][h][c]*=0 #(32,32,64,32,32)
        
    #     return imagetensor,inputtensor #(1,64,32,32),(32,32,64,32,32)
   
        
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
        
        layer_qkv,layer_proj = layer.children() #Linear(192, 576), Linear(192,192)
        
        
        size_image = imagetensor.size()#(1,257,192)
        size_input = inputtensor.size()#(257,192,64,64)
        
        '''
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        '''
        # (257,768),(257,768,64,64)
        image_qkv,input_qkv = self.linear_pytorch(layer_qkv,imagetensor.squeeze(),inputtensor)
        # qkv: (3,1,1,257,192) #(3,batch_size,num_heads,257,head_dim)
        # q or k or v: (1,1,257,192)
        image_qkv = torch.unsqueeze(image_qkv, 0)#(1,257,768)
        image_qkv = image_qkv.chunk(3, dim = -1)
        input_qkv = input_qkv.permute(2,3,0,1) #(64,64,257,768)
        input_qkv = input_qkv.chunk(3, dim = -1)
        #(1,1,257,192),(1,1,257,192),(1,1,257,192)
        image_q,image_v,image_k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), image_qkv)
        #(64,64,1,257,192),(64,64,1,257,192),(64,64,1,257,192)
        input_q,input_k,input_v = map(lambda t: rearrange(t, 'n1 n2 n (h d) -> n1 n2 h n d', h = self.heads), input_qkv)
        
        '''
        # q: (1,1,257,192)
        # k.transpose(-2, -1): (1,1,192,257)
        attn = (q @ k.transpose(-2, -1)) * self.scale # attn: (1,1,257,257)
        attn = attn.softmax(dim=-1) # attn: (1,1,257,257)
        '''
        image_attn = (image_q @ image_k.transpose(-2, -1)) * self.scale
        image_attn = image_attn.softmax(dim=-1) # (1,1,257,257)
        input_attn = torch.matmul(input_q, input_k.transpose(-1, -2)) * self.scale
        input_attn = input_attn.softmax(dim=-1) # (64,64,1,257,257)
        
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
        newimagetensor = layer_proj(image_out)#(1,257,192)
        input_out = torch.matmul(input_attn, input_v)#(64,64,1,257,192)
        input_out = rearrange(input_out, 'n1 n2 h n d -> n1 n2 n (h d)')
        newinputtensor = layer_proj(input_out)#(64,64,257,192)  
        
        #(1,257,192),(64,64,257,192)  
        newinputtensor = newinputtensor.permute(2,3,0,1)
        #(1,257,192),(257,192,64,64)  
        
        
        return newimagetensor,image_attn,newinputtensor,input_attn
    
    
    

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



