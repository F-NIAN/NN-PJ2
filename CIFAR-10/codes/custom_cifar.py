import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. 配置类
class Config:
    def __init__(self):
        # 优化器配置 
        self.optimizer_type = 'adamw'  
        self.learning_rate = 0.001     
        self.momentum = 0.9
        self.weight_decay = 5e-4       
        
        # 网络结构配置 
        self.conv1_filters = 128       
        self.conv2_filters = 256
        self.conv3_filters = 512       
        self.fc1_units = 1024
        
        # 损失函数配置
        self.loss_type = 'cross_entropy'  
        self.reg_type = 'l2'            
        self.reg_lambda = 1e-4          
        
        # 激活函数配置 
        self.activation = 'leaky_relu'  
        self.leaky_relu_slope = 0.1
        
        # 训练配置
        self.batch_size = 128
        self.epochs = 30               
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_scheduler = True       # 学习率调度标志

# 2. 自定义数据加载
def load_cifar10(data_root):
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data
    
    train_data = []
    train_label = []
    for i in range(1, 6):
        batch_path = os.path.join(data_root, f'data_batch_{i}')
        data = unpickle(batch_path)
        train_data.append(data[b'data'])
        train_label.extend(data[b'labels'])
    
    test_path = os.path.join(data_root, 'test_batch')
    test_data = unpickle(test_path)
    
    X_train = np.concatenate(train_data, axis=0).reshape(-1, 3, 32, 32).astype(np.float32)
    y_train = np.array(train_label)
    X_test = test_data[b'data'].reshape(-1, 3, 32, 32).astype(np.float32)
    y_test = np.array(test_data[b'labels'])
    
    X_train /= 255.0
    X_test /= 255.0
    
    return (X_train, y_train), (X_test, y_test)

# 3. 数据集封装
class CIFAR10Dataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label).long()
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        
        if self.transform:
            img = img.permute(1, 2, 0).numpy()
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.label)

# 4. 模型定义
class CustomCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 激活函数选择
        if config.activation == 'relu':
            self.activation = nn.ReLU()
        elif config.activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=config.leaky_relu_slope)
        elif config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"未知激活函数: {config.activation}")
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, config.conv1_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.conv1_filters)
        self.conv2 = nn.Conv2d(config.conv1_filters, config.conv2_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(config.conv2_filters)
        self.conv3 = nn.Conv2d(config.conv2_filters, config.conv3_filters, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(config.conv3_filters)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(config.conv3_filters * 4 * 4, config.fc1_units)  # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(config.fc1_units, 10)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(-1, self.config.conv3_filters * 4 * 4)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x

# 5. 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        
        # 根据损失类型选择基础损失函数
        if config.loss_type == 'cross_entropy':
            self.base_loss = nn.CrossEntropyLoss()
        elif config.loss_type == 'nll':
            self.base_loss = nn.NLLLoss()
        elif config.loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        else:
            raise ValueError(f"未知损失函数: {config.loss_type}")

    def forward(self, output, target):
        # 对于NLL和CrossEntropy，直接使用输出和目标
        if self.config.loss_type in ['nll', 'cross_entropy']:
            base_loss = self.base_loss(output, target)
        # 对于MSE，需要one-hot编码
        elif self.config.loss_type == 'mse':
            target_onehot = torch.zeros_like(output)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            base_loss = self.base_loss(output, target_onehot)
        
        # 正则化项（仅权重参数）
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'bn' not in name:  # 仅权重，排除BatchNorm
                if self.config.reg_type == 'l1':
                    reg_loss += torch.norm(param, 1)
                elif self.config.reg_type == 'l2':
                    reg_loss += torch.norm(param, 2)
        
        return base_loss + self.config.reg_lambda * reg_loss

# 6. 训练函数
def train_model(model, train_loader, test_loader, config):
    # 优化器选择
    if config.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"未知优化器: {config.optimizer_type}")
    
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # 损失函数
    criterion = CustomLoss(config, model)
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        if scheduler:
            scheduler.step(test_acc)  # 更新学习率
        
        print(f'Epoch {epoch+1}/{config.epochs} | '
              f'LR: {current_lr:.6f} | '  # 显示实际使用的学习率
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}%')
    
    return history

# 7. 主程序
if __name__ == '__main__':
    # 配置参数
    config = Config()
    
    # 数据加载
    DATA_ROOT = r'C:\Users\lenovo\Desktop\nndl\PJ2\data\CIFAR10\cifar-10-batches-py'
    (X_train, y_train), (X_test, y_test) = load_cifar10(DATA_ROOT)
    
    # 数据增强
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
    
    train_dataset = CIFAR10Dataset(X_train, y_train, transform_train)
    test_dataset = CIFAR10Dataset(X_test, y_test, transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # 模型初始化
    model = CustomCNN(config).to(config.device)
    
    # 训练
    history = train_model(model, train_loader, test_loader, config)
    
    # 可视化
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
    save_path = os.path.join(DATA_ROOT, '..', 'custom_training_results_2.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"训练结果已保存到: {save_path}")
    plt.show()
    
    # 保存模型
    model_save_path = os.path.join(DATA_ROOT, '..', 'costom_cnn_model_2.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"模型参数已保存到: {model_save_path}")