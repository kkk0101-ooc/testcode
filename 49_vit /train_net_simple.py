
import os
import warnings

from model.vit_final import *
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import shutil
import datetime as dt
import time
import argparse
from torch.utils.data import DataLoader
from function.dataset32 import *
from tqdm import tqdm
import os



parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, default='1', help='cuda number')
parser.add_argument("--model", type=str, default="VisionTransformer") #alexnet32，vgg1132, vgg1632,new_vgg1632,new_vgg1132,resnet
parser.add_argument('--dataset', default='tiny200ILSVRC', type=str) #ILSVRC32, tiny200ILSVRC, CUB
parser.add_argument('--classes', default=10, type=int)
parser.add_argument('--patch_size', default=4, type=int)
parser.add_argument('--num_heads', default=3, type=int)
parser.add_argument('--depth', default=6, type=int)
parser.add_argument('--img_size', default=32, type=int)


parser.add_argument("--mode", type=str, default="_without_pretrain", help="please add '_' before the mode")
parser.add_argument("--epochs", type=int, default=5000, help="number of training iterations")

parser.add_argument("--date", type=str, default="", help="please add '_' before the date")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate 1e-3 0.0005")
parser.add_argument("--lr_down", type=int, default=1, help="learning rate")
parser.add_argument('--logspace', default=True, type=bool)
parser.add_argument("--batch_size", type=int, default=100, help="batch size 256 500")
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--print-freq', '-p', default=3, type=int, metavar='N')
parser.add_argument('--save_per_epoch', default=True, type=bool)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--mark', type=str, default='10_17')
parser.add_argument('--threshold', default=60, type=int)
parser.add_argument('--use_all_class', default=False, type=bool)
args = parser.parse_args()

device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# 自动检测路径（Windows/Linux兼容）
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if os.path.exists(os.path.join(project_root, 'NNS_model_and_dataset')):
    base_path = os.path.join(project_root, 'NNS_model_and_dataset')
elif os.path.exists('/nas/NNS_model_and_dataset'):
    base_path = '/nas/NNS_model_and_dataset'
else:
    base_path = os.path.join(project_root, 'NNS_model_and_dataset')

PATH = os.path.join(base_path, 'models', 'vit_3channel_mutlihead', 
                    f'{args.dataset}_{args.model}_{args.img_size}imgsize_{args.classes}classes_{args.threshold}threshold_{args.num_heads}heads_{args.depth}depth_{args.patch_size}patch_size_')
if not os.path.exists(PATH):
    os.makedirs(PATH, exist_ok=True)


def train_network(PATH, args):
    if args.seed is not None:
        set_random(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    if args.dataset == 'CUB':
        traindir, valdir = '/nas/share/dataset/CUB_200_2011/train', '/nas/share/dataset/CUB_200_2011/val'
    elif args.dataset == 'VOC':
        traindir, valdir = '/nas/share/dataset/VOC/train', '/nas/share/dataset/VOC/test'
    elif args.dataset == 'ILSVRC' or args.dataset == 'ILSVRC32':
        traindir, valdir = '/data/dataset/imagenet_2/tiny_train', '/data/dataset/imagenet_2/tiny_val'
    elif args.dataset == 'tiny200ILSVRC':
        # 自动检测路径
        possible_paths = [
            (os.path.join(base_path, 'dataset', 'tiny-imagenet-5', 'train'), os.path.join(base_path, 'dataset', 'tiny-imagenet-5', 'val')),
            ('/nas/share/dataset/tiny-imagenet-5/train', '/nas/share/dataset/tiny-imagenet-5/val'),
        ]
        traindir, valdir = None, None
        for train_p, val_p in possible_paths:
            if os.path.exists(train_p) and os.path.exists(val_p):
                traindir, valdir = train_p, val_p
                break
        if traindir is None:
            sys.exit(f"Dataset error: 找不到tiny-imagenet-5数据集，检查路径: {possible_paths}")
    else:
        sys.exit("dataset error")
    if args.use_all_class:
        if args.dataset == 'CUB':
            train_loader = torch.utils.data.DataLoader(AugmentDataSet_v2(traindir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
        elif args.dataset == 'VOC' or args.dataset == 'ILSVRC'or args.dataset == 'tiny200ILSVRC':
            train_loader = torch.utils.data.DataLoader(AugmentDataSet(traindir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
        elif args.dataset == 'ILSVRC32'or args.dataset == 'tiny200ILSVRC32':
            train_loader = torch.utils.data.DataLoader(AugmentDataSet(traindir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
        else:
            sys.exit("Dataset Error")
        val_loader = torch.utils.data.DataLoader(SigmaDataSet(valdir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
    else:
        if args.dataset == 'CUB':
            train_loader = torch.utils.data.DataLoader(AugmentDataSet_num(traindir, num=args.classes), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
        elif args.dataset == 'VOC' or args.dataset == 'ILSVRC'or args.dataset == 'tiny200ILSVRC':
            train_loader = torch.utils.data.DataLoader(AugmentDataSet_num(traindir, num=args.classes), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
        elif args.dataset == 'ILSVRC32'or args.dataset == 'tiny200ILSVRC32':
            train_loader = torch.utils.data.DataLoader(AugmentDataSet_num(traindir, num=args.classes), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
        else:
            sys.exit("Dataset Error")
        val_loader = torch.utils.data.DataLoader(SigmaDataSet_num(valdir,num=args.classes), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)


    if args.model == 'VisionTransformer':
        model = VisionTransformer(img_size=[32],patch_size=args.patch_size, num_classes=args.classes, embed_dim=192, depth=args.depth, num_heads=args.num_heads, mlp_ratio=4,
                                            qkv_bias=False)
        
    # if args.model == 'alexnet32':
    #     model = AlexNet32(args.classes)
    # elif args.model == 'vgg1632':
    #     model = Vgg1632(args.classes)
    # elif args.model == 'vgg1132':
    #     model = Vgg1132(args.classes)
    # elif args.model == 'new_vgg1632':
    #     model = DELVgg1632(args.classes)
    # elif args.model == 'new_vgg1132':
    #     model = DELVgg1132(args.classes)
    # elif args.model =="resnet":
    #     model = ResNet18(args.classes)
    #     if args.dataset == 'CUB':
    #         pretrained_dict=torch.load('/home/trained_model/CUB_resnet_10classes_30threshold_10_17/resnet_101.pth')["state_dict"]
    #         model_dict = model.state_dict()
    #         model_dict.update(pretrained_dict)
    #         model.load_state_dict(model_dict)
    else:
        sys.exit('Model Name Error')

    # if args.model == 'alexnet32':
    #     #model.load_state_dict(torch.load('/nas/trained_model/ILSVRC32_alexnet32_pretrain/AlexNet_cifar10.pth'))
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load('/nas/trained_model/AlexNet_cifar10.pth')
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)

    model = model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.epochs)):
        prec1=0
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # if epoch % 10 == 0:
        # # evaluate on validation set
        acc, loss = validate(val_loader, model, criterion, args.print_freq)
        print(f"epoch:{epoch} acc:{acc},loss:{loss}")
        # 确保acc是数值而不是tensor或AverageMeter
        if hasattr(acc, 'item'):
            acc_value = acc.item()
        elif hasattr(acc, 'avg'):
            acc_value = acc.avg
        elif isinstance(acc, (int, float)):
            acc_value = float(acc)
        else:
            acc_value = float(acc)
        print(f"  [DEBUG] acc_value={acc_value}, threshold={args.threshold}, will_save={acc_value > args.threshold}")
        if acc_value > args.threshold:
            save_checkpoint({
           'epoch': epoch + 1,
             'model': args.model,
             'state_dict': model.state_dict()
         }, PATH + args.model+ '_' + str(epoch+1) +'_' + str(round(acc_value,2))+ '.pth')

        # # if epoch % 20 == 0:
        # if epoch == args.epochs-1:
        #     save_checkpoint({
        #     'epoch': epoch + 1,
        #     'model': args.model,
        #     'state_dict': model.state_dict()
        # }, PATH+args.model + '.pth')
            

        # # remember the best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
    """save_checkpoint({
            'model': args.model,
            'state_dict': model.state_dict()
        }, PATH+args.model + '.pth')"""

    # 训练结束后保存最终模型
    final_model_path = PATH + args.model + '_' + str(args.epochs) + '_final.pth'
    print(f"\n保存最终模型到: {final_model_path}")
    save_checkpoint({
        'epoch': args.epochs,
        'model': args.model,
        'state_dict': model.state_dict()
    }, final_model_path)
    print(f"✓ 最终模型已保存")
    # prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)

    

def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    weight_distance_conv, weight_distance_fc1, weight_distance_fc2, weight_distance_fc3 = 0, 0, 0, 0
    end = time.time()
    for i, (id, img_name, input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # w_origin_conv, w_origin_fc1, w_origin_fc2, w_origin_fc3 = get_weight(model)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    acc1, acc5 = accuracy(output, target, topk=(1, 1))
    losses.update(loss.item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))
    print('Train * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    

@torch.no_grad()        # deactivate autograd engine to reduce memory consumption and speed up computations
def validate(val_loader, model, criterion, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (id, img_name, input, target) in enumerate(val_loader):
        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 1))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
    print('Test * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    # log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    # set_tensorboard(log_dict, epoch, logger)
    return top1.avg, losses.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# set for tensorboard
def set_tensorboard(log_dict, epoch, logger):

    info = log_dict
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.epoch_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
          lr += [param_group['lr']]
    return lr


def _init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


def set_random(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def get_weight(model):

    if args.model[:3] == 'vgg':
        w_origin_fc1 = model.net.classifier[0].weight.clone().detach()
        w_origin_fc2 = model.net.classifier[3].weight.clone().detach()
        w_origin_fc3 = model.net.classifier[6].weight.clone().detach()
        if args.model == 'vgg11':
            w_origin_conv = model.net.features[18].weight.clone().detach()
        elif args.model == 'vgg16':
            w_origin_conv = model.net.features[28].weight.clone().detach()
        elif args.model == 'vgg19':
            w_origin_conv = model.net.features[34].weight.clone().detach()
        else:
            sys.exit('Model Name Error')
    elif args.model == 'alexnet':
        w_origin_conv = model.net.features[10].weight.clone().detach()
        w_origin_fc1 = model.net.classifier[1].weight.clone().detach()
        w_origin_fc2 = model.net.classifier[4].weight.clone().detach()
        w_origin_fc3 = model.net.classifier[6].weight.clone().detach()
    elif args.model[:6] == 'resnet':
        w_origin_conv = model.features[3].weight.clone().detach()
        w_origin_fc1 = model.net.fc[0].weight.clone().detach()
        w_origin_fc2 = model.net.fc[3].weight.clone().detach()
        w_origin_fc3 = model.net.fc[6].weight.clone().detach()
    else:
        sys.exit('Model Name Error')

    return w_origin_conv, w_origin_fc1, w_origin_fc2, w_origin_fc3


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    train_network(PATH, args)