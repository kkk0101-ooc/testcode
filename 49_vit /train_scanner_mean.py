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
from model.vit_mean_final import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from NNS.Scannerpic1107_mean_nonet import NNS_NONET
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
                    help='ILSVRC32 tiny200ILSVRC CUB VOC MNIST')
#parser.add_argument('--mark', type=str, default='epoch') #test test1
#parser.add_argument('--accmark', type=str, default='00') #test test1
parser.add_argument('--classes', type=int, default=25) #test 10,25
parser.add_argument('--use_all_class', default=True, type=bool)
args = parser.parse_args()

device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
print('device:'+args.device+','+str(torch.cuda.is_available()))
print(args.model+'_'+args.dataset)





def train_scanenr():

    classes=args.classes
    COUNT=args.count
    if args.model == 'VisionTransformer':
        if args.dataset=="tiny200ILSVRC":
            modeldir = '/nas/NNS_model_and_dataset/models/vit_mean_3channel/tiny200ILSVRC_VisionTransformer_32imgsize_25classes_30threshold_1heads_6depth_4patch_size_/'
            traindir = '/nas/NNS_model_and_dataset/dataset/tiny-imagenet-5/scanner'
            model = VisionTransformer(img_size=[32],patch_size=args.patch_size, num_classes=args.classes, embed_dim=192, depth=6, num_heads=1, mlp_ratio=4,
                                            qkv_bias=False)
        if args.dataset == 'MNIST':       
            # modeldir = '/nas/NNS_model_and_dataset/models/vit_3channel/MNIST_VisionTransformer_28imgsize_10classes_80threshold_1heads_3depth_4patch_size_'
            modeldir = '/nas/NNS_model_and_dataset/models/vit_mean_3channel/MNIST_VisionTransformer_28imgsize_10classes_80threshold_1heads_3depth_4patch_size_'
            traindir = '/nas/NNS_model_and_dataset/dataset/mnist_scanner'
            model = VisionTransformer(img_size=[28],patch_size=args.patch_size, in_chans=1, num_classes=args.classes, embed_dim=192, depth=3, num_heads=1, mlp_ratio=4,
                                            qkv_bias=False)
            
            
            
    child = list(model.children())
    LN = len(child)


    if args.dataset=="tiny200ILSVRC":
        if args.use_all_class:
            dataloader = torch.utils.data.DataLoader(ScanDataSet(traindir, ''), batch_size=1, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(ScanDataSet_num(traindir, args.classes), batch_size=1, shuffle=False)
    elif args.dataset == 'MNIST': 
        dataloader = torch.utils.data.DataLoader(MnistDataset(traindir),batch_size=1, shuffle=True)
    
    
    for m in os.listdir(modeldir):
        model_path = os.path.join(modeldir, m)
        model_name = ".".join(m.split('.')[:-1])
        params = load_checkpoint_and_return_params(model_path, model)
        model = model.to(device)
        print(model)


        pbar = tqdm(total=COUNT)
        ii = 0
        real = 0
        for i, (id, img_name, image, target) in enumerate(dataloader):

            if ii>=COUNT:
                break

            image = image.to(device)
            target = target.to(device)
            output = model(image)
            out = torch.argsort(output)[0,-1:]

            if out != target:
                continue


            name = '.'.join(img_name[0].split('.')[:-1]).split('/')[-1]
            print(img_name[0]+'*'+ name)

            nns = NNS_NONET(model,image[0],name,ii,device,model_name,args.model,args.dataset,args.patch_size,args.num_heads,args.depth,params)
            #不重复跑
            if not os.path.exists(nns.rootdir):
                nns.get_scanpic(LN)

            ii = ii + 1
            torch.cuda.empty_cache()
            pbar.update(1)

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
