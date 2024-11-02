import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from data import CustomMNISTDataset  # Assuming your dataset class is in custom_dataset.py
from FCN import FCN
from LeNet5 import ModifiedLeNet5

model_name = 'ModifiedLeNet5'
model_name = 'FCN'

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, gt, _ in train_loader:
            images, gt = images.to(device), gt.to(device)  # 将数据放到 GPU
            optimizer.zero_grad()
            outputs = model(images)
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    if model_name == 'FCN':
        model = FCN().to(device)  # 将模型放到 GPU
    elif model_name == 'LeNet5':
        model = ModifiedLeNet5().to(device)  # 将模型放到 GPU
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train(model, train_loader, criterion, optimizer, num_epochs=10)
    torch.save(trained_model.state_dict(), 'fcn_model.pth')
    print("模型已保存为 'fcn_model.pth'.")