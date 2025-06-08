import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

# 1. 自定义数据加载
def load_cifar10(data_root):
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data
    
    train_data = []
    train_labels = []
    # 加载训练集（5个batch）
    for i in range(1, 6):
        batch_path = os.path.join(data_root, f'data_batch_{i}')
        data = unpickle(batch_path)
        train_data.append(data[b'data'])
        train_labels.extend(data[b'labels'])
    
    # 加载测试集
    test_path = os.path.join(data_root, 'test_batch')
    test_data = unpickle(test_path)
    
    # 合并并转换数据格式
    X_train = np.concatenate(train_data, axis=0).reshape(-1, 3, 32, 32).astype(np.float32)
    y_train = np.array(train_labels)
    X_test = test_data[b'data'].reshape(-1, 3, 32, 32).astype(np.float32)
    y_test = np.array(test_data[b'labels'])
    
    # 归一化到 [0, 1]
    X_train /= 255.0
    X_test /= 255.0
    
    return (X_train, y_train), (X_test, y_test)

# 2. 数据集封装
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        
        if self.transform:
            # 自定义变换处理NHWC -> NCHW
            img = img.permute(1, 2, 0).numpy()  # 转为HWC格式供transform处理
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.labels)

# 3. 模型定义（修正后的版本）
class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 4. 训练与验证函数
def train_model(model, train_loader, test_loader, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss = test_loss/len(test_loader)
        test_acc = 100.*correct/total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1:02d} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        scheduler.step()
    
    return history

# 5. 主程序
if __name__ == '__main__':
    # 配置文件路径
    DATA_ROOT = r'C:\Users\lenovo\Desktop\nndl\PJ2\data\CIFAR10\cifar-10-batches-py'
    
    # 数据增强配置
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载数据
    print("正在加载数据...")
    (X_train, y_train), (X_test, y_test) = load_cifar10(DATA_ROOT)
    train_dataset = CIFAR10Dataset(X_train, y_train, transform_train)
    test_dataset = CIFAR10Dataset(X_test, y_test, transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    # 初始化模型
    print("初始化模型...")
    model = BasicCNN()
    
    # 训练模型
    print("开始训练...")
    history = train_model(model, train_loader, test_loader, epochs=30)
    
    # 可视化结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(DATA_ROOT, '..', 'training_results.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"训练结果已保存到: {save_path}")
    plt.show()
    
    # 保存模型
    model_save_path = os.path.join(DATA_ROOT, '..', 'basic_cnn_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"模型参数已保存到: {model_save_path}")
