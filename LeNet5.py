import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import CustomMNISTDataset

# 定义 LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class UpLeNet5(nn.Module):
    def __init__(self):
        super(UpLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x1 = torch.mean(x, dim=1, keepdim=True)
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # 使用双线性插值上采样将输出大小调整为输入大小
        output = nn.functional.interpolate(x1, size=(28, 28), mode='bilinear', align_corners=True)
        return x, output

class DeConLeNet5(nn.Module):
    def __init__(self):
        super(DeConLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 添加反卷积层
        self.deconv1 = nn.ConvTranspose2d(16, 6, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(6, 3, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(3, 1, kernel_size=2, stride=2)
        self.finalconv = nn.Conv2d(1, 1, kernel_size=5)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x1 = x
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # 使用反卷积进行上采样
        x_up = self.deconv1(x1)
        x_up = self.deconv2(x_up)
        x_up = self.deconv3(x_up)
        output = self.finalconv(x_up)
        return x, output

class DeConLeNet5Large(nn.Module):
    def __init__(self, num_classes=1):
        super(DeConLeNet5Large, self).__init__()

        # 编码器部分
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码器部分
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU()

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.relu6 = nn.ReLU()

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu7 = nn.ReLU()

        self.fc1 = nn.Linear(512 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # 输出层

    def forward(self, x):
        # 编码器部分
        x1 = self.relu1(self.conv1(x))
        x2 = self.pool1(x1)

        x3 = self.relu2(self.conv2(x2))
        x4 = self.pool2(x3)

        x5 = self.relu3(self.conv3(x4))
        x6 = self.pool3(x5)

        x7 = self.relu4(self.conv4(x6))
        x8 = self.pool4(x7)

        xs = x8.view(-1, 512 * 1 * 1)
        xs = torch.relu(self.fc1(xs))
        xs = torch.relu(self.fc2(xs))
        xs = torch.relu(self.fc3(xs))
        xs = self.fc4(xs)
        # 解码器部分
        x9 = self.relu5(self.upconv1(x8))
        x10 = self.relu6(self.upconv2(x9))
        x11 = self.relu7(self.upconv3(x10))

        # 使用双线性插值上采样将输出大小调整为输入大小
        output = nn.functional.interpolate(self.final_conv(x11), size=(28, 28), mode='bilinear', align_corners=True)

        return xs, output



#LENET测试分类代码
if __name__ == '__main__':
    train_dataset = torch.load('train_dataset.pt')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 加载测试数据集
    test_dataset = torch.load('test_dataset.pt')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (images, _ , labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

    # 测试模型
    with torch.no_grad():
        correct = 0
        total = 0
        for images, _, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total}%')
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
    torch.save(model.state_dict(), 'saved_model/lenet5_model.pth')