import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import CustomMNISTDataset
from LeNet5 import LeNet5,UpLeNet5,DeConLeNet5,DeConLeNet5Large

model_list = ['UpLeNet5', 'DeConLeNet5', 'DeConLeNet5Large'] #保存要训练的模型

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, gt, _ in train_loader:
            images, gt = images.to(device), gt.to(device)  # 将数据放到 GPU
            optimizer.zero_grad()
            xs , outputs = model(images)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_dataset = torch.load('train_dataset.pt')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for model_name in model_list:
        if model_name == 'UpLeNet5':
            model = UpLeNet5().to(device)  # 将模型放到 GPU
        elif model_name == 'DeConLeNet5':
            model = DeConLeNet5().to(device)  # 将模型放到 GPU
        elif model_name == 'DeConLeNet5Large':
            model = DeConLeNet5Large().to(device)  # 将模型放到 GPU

        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        if model_name == 'DeConLeNet5Large':
            num_epochs = 20
        else:
            num_epochs = 10
        trained_model = train(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
        torch.save(trained_model.state_dict(), f'saved_model/{model_name}_model.pth')
        print(f"模型已保存为 'saved_model/{model_name}_model.pth'.")