import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from data import CustomMNISTDataset  # Assuming your dataset class is in custom_dataset.py



class FCN(nn.Module):
    def __init__(self, num_classes=1):
        super(FCN, self).__init__()

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

        # 解码器部分
        x9 = self.relu5(self.upconv1(x8))
        x10 = self.relu6(self.upconv2(x9))
        x11 = self.relu7(self.upconv3(x10))

        # 使用双线性插值上采样将输出大小调整为输入大小
        output = nn.functional.interpolate(self.final_conv(x11), size=(28, 28), mode='bilinear', align_corners=True)

        return output





class FCN10(nn.Module):
    def __init__(self, num_classes=10):  # 修改为10个类别
        super(FCN10, self).__init__()

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

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # 输出层更改为num_classes

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

        # 解码器部分
        x9 = self.relu5(self.upconv1(x8))
        x10 = self.relu6(self.upconv2(x9))
        x11 = self.relu7(self.upconv3(x10))

        # 上采样至输入大小
        output = nn.functional.interpolate(self.final_conv(x11), size=(28, 28), mode='bilinear', align_corners=True)

        return output
