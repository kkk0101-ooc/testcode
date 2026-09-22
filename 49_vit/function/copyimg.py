import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random

####每个类选出 N 个样本进行扫描

N = 0
DATA = 'tinyimagenet_2'


if DATA == 'imagenet_2': #imagenet_2
    N = 50
    traindir =  '/data/dataset/imagenet_2/test1'
    outputdir = '/data/dataset/imagenet_2/scanimg'

elif DATA == 'tinyimagenet_2':#tinyimagenet_2
    N = 50
    traindir =  '/data/dataset/tiny-imagenet-5/train'
    outputdir = '/data/dataset/tiny-imagenet-5/scanimg'

img_class = os.listdir(traindir)
img_class.sort()

for i in range(len(img_class)):
    sub_root = traindir + '/' + img_class[i] + '/'
    images = os.listdir(sub_root)
    images.sort()
    count = 0
    for im in images:
        if count > N:
            break
        d = img_class[i] + '/' 
        output_folder = outputdir+ '/' +d
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image = Image.open(traindir+ '/' +d+ im)
        image.save(output_folder + im)

        count = count +1
