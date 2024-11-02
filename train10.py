import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import CustomMNISTDataset  # 假设自定义数据集的实现
from FCN import FCN10  # 假设你的模型在此文件中

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_dataset = torch.load('train_dataset.pt')  # 加载处理后的训练数据集
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = FCN10(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()  # 对于多类分割
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # 根据需要调整
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, gt, labels in train_loader:
            images, gt, labels = images.to(device), gt.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            gt_labels = gt.clone()
            for i, label in enumerate(labels):
                gt_labels[i][gt_labels[i] == 1] = label.float()
            gt_labels = gt_labels.squeeze(1).long()
            loss = criterion(outputs, gt_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'fcn10_model.pth')
    print("模型已保存为 fcn10_model.pth")

if __name__ == "__main__":
    train()
