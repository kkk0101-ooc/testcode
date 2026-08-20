import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from model.vit_final import *

# 设置设备
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义超参数
batch_size = 128
num_epochs = 100
learning_rate = 3e-4
weight_decay = 0.05
patch_size = 4  # CIFAR10为32x32，使用4x4的patch
img_size = 32
embed_dim = 384  # 嵌入维度
depth = 7        # Transformer块数量
num_heads = 6    # 注意力头数量
num_classes = 10 # CIFAR10有10个类别

# 数据增强和预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# 加载CIFAR10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=2)

# CIFAR10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 实例化模型
model = VisionTransformer(
    img_size=[img_size],
    patch_size=patch_size,
    in_chans=3,
    num_classes=num_classes,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=4.,
    qkv_bias=True,
    norm_layer=nn.LayerNorm
)

# 初始化权重
def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

model.apply(_init_vit_weights)

# 将模型移至GPU
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f'Epoch: {epoch+1}/{num_epochs}')
    start_time = time.time()
    
    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (i+1) % 100 == 0:
            print(f'Batch: {i+1}/{len(trainloader)}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.3f}%')
            running_loss = 0.0
    
    end_time = time.time()
    print(f'Epoch {epoch+1} completed in {(end_time-start_time)/60:.2f} minutes')
    return 100. * correct / total

# 测试函数
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    print(f'Test Loss: {test_loss/len(testloader):.3f}, Test Acc: {acc:.3f}%')
    return acc

# 保存检查点
def save_checkpoint(model, epoch, acc, best_acc):
    state = {
        'model': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, f'./checkpoint/vit_cifar10_epoch_{epoch}.pth')
    if acc > best_acc:
        torch.save(state, './checkpoint/vit_cifar10_best.pth')
        print(f'Best model saved with accuracy: {acc:.3f}%')
        return acc
    return best_acc

# 可视化训练过程
def plot_metrics(train_acc_history, test_acc_history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 1, 1)
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(test_acc_history, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Testing Accuracy')
    plt.savefig('vit_cifar10_accuracy.png')
    plt.show()

# 主训练循环
train_acc_history = []
test_acc_history = []
best_acc = 0

for epoch in range(num_epochs):
    train_acc = train(epoch)
    test_acc = test(epoch)
    scheduler.step()
    
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    
    best_acc = save_checkpoint(model, epoch, test_acc, best_acc)
    
    print(f'Current LR: {scheduler.get_last_lr()[0]}')
    print('-------------------------')

# 绘制训练和测试准确率
plot_metrics(train_acc_history, test_acc_history)
print(f'Best accuracy: {best_acc:.3f}%')

# 加载最佳模型并在测试集上评估
print("Loading best model for final evaluation...")
checkpoint = torch.load('./checkpoint/vit_cifar10_best.pth')
model.load_state_dict(checkpoint['model'])
best_epoch = checkpoint['epoch']
print(f"Best model was from epoch {best_epoch+1}")

test_acc = test(best_epoch)
print(f'Final test accuracy: {test_acc:.3f}%')

# 可视化一些预测结果
def visualize_predictions():
    model.eval()
    # 获取一批测试数据
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # 预测结果
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # 将部分图像移回CPU以可视化
    images = images.cpu().numpy()
    
    # 绘制图像和预测结果
    fig = plt.figure(figsize=(12, 6))
    for i in range(12):
        ax = fig.add_subplot(3, 4, i+1)
        img = np.transpose(images[i], (1, 2, 0))
        # 反归一化
        img = img * np.array((0.2470, 0.2435, 0.2616)) + np.array((0.4914, 0.4822, 0.4465))
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f'Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('vit_cifar10_predictions.png')
    plt.show()

visualize_predictions()
