import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random


class SigmaDataSet(Dataset):
    def __init__(self, root, model):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transformation = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size


class AugmentDataSet(Dataset):
    def __init__(self, root, model):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size
    
class AugmentDataSet_v2(Dataset):
    def __init__(self, root, model):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size

class AugmentDataSet_num(Dataset):
    def __init__(self, root,num):
        self.root = root
        self.origin_img_class = os.listdir(self.root)
        self.origin_img_class.sort()
        self.img_class=self.origin_img_class[:num]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size
class SigmaDataSet_num(Dataset):
    def __init__(self, root, num):
        self.root = root
        self.origin_img_class = os.listdir(self.root)
        self.origin_img_class.sort()
        self.img_class = self.origin_img_class[:num]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transformation = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]).convert('RGB'))
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size
    
class ScanDataSet(Dataset):  #CenterCrop
    def __init__(self, root, model):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transformation = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        im = Image.open(self.root+'/'+self.name_list[idx]).convert('RGB')
        image = self.transformation(im)
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size

class ScanDataSet_num(Dataset):  #CenterCrop
    def __init__(self, root, num):
        self.root = root
        self.origin_img_class = os.listdir(self.root)
        self.origin_img_class.sort()
        self.img_class = self.origin_img_class[:num]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transformation = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), normalize])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        im = Image.open(self.root+'/'+self.name_list[idx]).convert('RGB')
        image = self.transformation(im)
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size


class SimpleDataSet(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        self.transformation = transforms.Compose([transforms.ToTensor()])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(str(i)+ '_' + im)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        # image = self.transformation(Image.open(self.root+'/'+self.name_list[idx]))

        image = torch.load(self.root+'/'+self.name_list[idx])
        image = torch.unsqueeze(image,0)

        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size
    
    
class MnistDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_class = os.listdir(self.root)
        self.img_class.sort()
        self.transform_mnist = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])
        self.name_list = []
        self.label_list = []
        self.size = 0
        for i in range(len(self.img_class)):
            sub_root = self.root + '/' + self.img_class[i] + '/'
            images = os.listdir(sub_root)
            images.sort()
            for im in images:
                self.name_list.append(self.img_class[i] + '/' + im)
                self.label_list.append(i)
                self.size += 1

    def __getitem__(self, idx):
        label = self.label_list[idx]
        im = Image.open(self.root+'/'+self.name_list[idx])
        image = self.transform_mnist(im)
        image_name = self.name_list[idx]
        return idx, image_name, image, label

    def __len__(self):
        return self.size
    
