import os
import random
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from PIL import Image

save_dir = './processed_images'
os.makedirs(save_dir, exist_ok=True)

class CustomMNISTDataset(Dataset):
    def __init__(self, mnist_data, background_images_dir, transform=None):
        self.mnist_data = mnist_data
        self.background_images_dir = background_images_dir
        self.transform = transform
        self.background_images = self.load_background_images()

    def load_background_images(self):
        backgrounds = []
        for filename in os.listdir(self.background_images_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(self.background_images_dir, filename)
                img = Image.open(img_path).convert('RGB')
                backgrounds.append(img)
        return backgrounds

    def random_crop(self, background, size=(28, 28)):
        w, h = background.size
        x = random.randint(0, w - size[0])
        y = random.randint(0, h - size[1])
        return background.crop((x, y, x + size[0], y + size[1]))

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        # 获取手写数字图像和标签
        mnist_image, label = self.mnist_data[idx]

        # 转换 mnist_image 为 NumPy 数组
        mnist_array = np.array(mnist_image)

        # 随机选择背景图像并切割
        background = random.choice(self.background_images)
        background_patch = self.random_crop(background)

        # 拼接手写数字和背景图像
        combined_image = Image.new('RGB', (28, 28))
        combined_image.paste(background_patch, (0, 0))
        combined_image.paste(mnist_image.convert('RGB'), (0, 0), mnist_image.convert('L'))

        if self.transform:
            combined_image = self.transform(combined_image)

        # 创建分割 GT
        segmentation_gt = torch.zeros((1, 28, 28), dtype=torch.float32)  # 确保数据类型为 float32
        segmentation_gt[0, mnist_array > 0] = 1  # 前景为1，背景为0

        # 保存分割 GT 图像
        gt_filename = os.path.join(save_dir, f'gt_{idx}.png')
        cv2.imwrite(gt_filename, (segmentation_gt[0].numpy() * 255).astype(np.uint8))

        return combined_image, segmentation_gt


# 数据预处理与加载
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

mnist_data = MNIST(root='./data', train=True, download=True)
background_images_dir = './background_images'  # 背景图像目录

custom_dataset = CustomMNISTDataset(mnist_data, background_images_dir, transform=transform)
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# 示例：遍历数据加载器
for images, gt in data_loader:
    print(images.shape, gt.shape)
    break  # 只显示一个批次
