import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data import CustomMNISTDataset



class ModifiedLeNet5(nn.Module):
    def __init__(self):
        super(ModifiedLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 1, kernel_size=5)
        self.upsample = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.upsample(x)  # Upsample to (28, 28)
        return x



def train_model(train_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, gt, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    # Load the datasets
    train_dataset = torch.load('train_dataset.pt')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define model, loss function, and optimizer
    model = ModifiedLeNet5()
    criterion = nn.BCEWithLogitsLoss()  # Suitable for binary segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(train_loader, model, criterion, optimizer, num_epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), 'modified_lenet5_segmentation.pth')
    print("模型已保存为 modified_lenet5_segmentation.pth")
