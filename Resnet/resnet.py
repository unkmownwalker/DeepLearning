import torch ,torch.nn as nn
import os
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#加载 CIFAR-10 原始数据
def load_cifar10_raw(data_dir='./data'):
    # 下载数据集，transform=None 表示返回原始 PIL 图像
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=None
    )
    
    # 转换为 NumPy 数组 (N,C, H, W) uint8
    train_images = np.stack([np.array(img) for img, _ in train_dataset]).transpose(0, 3, 1, 2)   
    test_images = np.stack([np.array(img) for img, _ in test_dataset]).transpose(0, 3, 1, 2)
    train_labels = np.array([label for _, label in train_dataset])
    test_labels = np.array([label for _, label in test_dataset])
    return train_images, train_labels, test_images, test_labels

class basicbolck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
         super().__init__()
         self.features=nn.Sequential(
         nn.BatchNorm2d(in_channels),
         nn.ReLU(inplace=True),
         nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=True),
         nn.BatchNorm2d(out_channels),
         nn.ReLU(inplace=True),
         nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=True),
         )
         self.shortcut = nn.Sequential()
         if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        identity=x
        x = self.features(x)
        out=x+self.shortcut(identity)
        return out
    
class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
         super().__init__()
         self.features=nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, 
                               stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, 
                               stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, out_channels, kernel_size=1, 
                               stride=1, padding=0, bias=True),
         )
         self.shortcut = nn.Sequential()
         if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self,x):
        identity=x
        x = self.features(x)
        out=x+self.shortcut(identity)
        return out
    
class resnet(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        # stage 1
        self.stage1=nn.Sequential(
         nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, 
                               stride=1, padding=1, bias=True),
         nn.BatchNorm2d(32),
         nn.ReLU(inplace=True),
         )

        #stage 2
        self.stage2=nn.Sequential(
             basicbolck(32, 32, stride=1),
             basicbolck(32, 64, stride=2),   # 16x16
             basicbolck(64, 64, stride=1),
             basicbolck(64, 128, stride=2),  # 8x8
             basicbolck(128, 128, stride=1),
             basicbolck(128, 256, stride=2), # 4x4
             basicbolck(256, 256, stride=1),
            
        )

        #stage 3
        self.stage3=nn.Sequential(
             nn.AdaptiveAvgPool2d((1, 1)),
             nn.Flatten(),
             nn.Dropout(p=0.6),
             nn.Linear(256,num_classes),
        )
    
    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        out = self.stage3(x)
        return out

#data prepare

def data_prepare(batch_size=128, data_dir='./data', augment=True):
    # CIFAR-10 图像的均值和标准差（预先计算好的）
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # 训练集的预处理：数据增强 + 归一化
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # 随机裁剪，四周填充4像素
            transforms.RandomHorizontalFlip(),      # 随机水平翻转
            transforms.ToTensor(),                  # 转为张量并缩放到 [0,1]
            transforms.Normalize(mean, std)         # 标准化
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # 测试集：仅归一化，不增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 下载/加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader

#train
def one_epoch_train(model,device,loader,optimizer,criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images,labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

#test
def test(model,device,loader,criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  #不构建计算图
        for images,labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #.item()：将结果（标量张量）转换为 Python 整数

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":

    #gpu setup
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #settings
    epochs=50
    train_loader, test_loader = data_prepare()
    model = resnet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)#平滑标签微弱提高泛化
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)
    for epoch in range(epochs):
        train_loss, train_acc = one_epoch_train(model,device,train_loader,optimizer,criterion)
        scheduler.step()
        print(f"{epoch+1}: Loss:{train_loss}, Acc:{train_acc}")

    test_loss, test_acc =test(model,device,test_loader,criterion)
    print(f"test_Loss:{test_loss}, text_Acc:{test_acc}")
    print("训练完成！")
    
    #保存模型权重
    torch.save(model.state_dict(), "resnet_cifar10.pth")
    print("模型已保存为 resnet_cifar10.pth")




    
