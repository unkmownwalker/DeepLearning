import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torchvision

#model definition
class alexnet(nn.Module):
    def __init__(self, num_classes=10):
          super().__init__()
          self.features=nn.Sequential(
               nn.Conv2d(in_channels=1,kernel_size=5, stride=1,padding=2,out_channels=16,bias=False),
               nn.BatchNorm2d(num_features=16),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=2,stride=2),

               nn.Conv2d(in_channels=16,kernel_size=3, stride=1,padding=1,out_channels=32,bias=False),
               nn.BatchNorm2d(num_features=32),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=2,stride=2),

               nn.Conv2d(in_channels=32,kernel_size=3, stride=1,padding=1,out_channels=64,bias=False),
               nn.BatchNorm2d(num_features=64),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=2,stride=1,padding=0),

               #shape:6,6,64
          )
          self.classifier=nn.Sequential(
               nn.Linear(64*6*6,1024),
               nn.ReLU(inplace=True),
               nn.Dropout(p=0.4),

               nn.Linear(1024,128),
               nn.ReLU(inplace=True),
               nn.Dropout(p=0.4),

               nn.Linear(128,num_classes),
          )

    def  forward(self,x):
         x = self.features(x)
         x = torch.flatten(x, 1)
         x = self.classifier(x)
         return x

#data set
def download_mnist_raw(data_dir='./data'):
    """下载MNIST原始数据"""
    # 下载训练集（不应用任何 transform）
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=None
    )
    # 直接获取原始图像数据和标签
    train_data = train_dataset.data.float()       # 已经是 torch.uint8 类型
    train_labels = train_dataset.targets

    # 下载测试集
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=None
    )
    test_data = test_dataset.data.float()
    test_labels = test_dataset.targets

    return train_data, train_labels, test_data, test_labels

#data prepare
def data_prepare(batch_size=100):
    train_data, train_labels, test_data, test_labels = download_mnist_raw()
    train_data = train_data.reshape(-1,1,28,28)/255
    test_data =test_data .reshape(-1,1,28,28)/255

    # 将张量包装为 TensorDataset，便于 DataLoader 使用
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
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
    epochs=20
    train_loader, test_loader = data_prepare()
    model = alexnet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        train_loss, train_acc = one_epoch_train(model,device,train_loader,optimizer,criterion)
        print(f"{epoch+1}: Loss:{train_loss}, Acc:{train_acc}")

    test_loss, test_acc =test(model,device,test_loader,criterion)
    print(f"test_Loss:{test_loss}, text_Acc:{test_acc}")
    print("训练完成！")
    
    #保存模型权重
    torch.save(model.state_dict(), "alexnet_mnist.pth")
    print("模型已保存为 alexnet_mnist.pth")
    


