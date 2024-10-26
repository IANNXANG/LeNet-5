import os
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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
        mnist_image, label = self.mnist_data[idx]
        mnist_array = np.array(mnist_image)

        background = random.choice(self.background_images)
        background_patch = self.random_crop(background)

        combined_image = Image.new('RGB', (28, 28))
        combined_image.paste(background_patch, (0, 0))
        combined_image.paste(mnist_image.convert('RGB'), (0, 0), mnist_image.convert('L'))

        if self.transform:
            combined_image = self.transform(combined_image)

        segmentation_gt = torch.zeros((1, 28, 28), dtype=torch.float32)
        segmentation_gt[0, mnist_array > 0] = 1

        return combined_image, segmentation_gt, label

# 数据预处理与加载
background_images_dir = './background_images'  # 背景图像目录
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# 创建输出目录
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# 创建子文件夹用于保存训练集和测试集的数字和GT
for dataset_type in ['train', 'test']:
    for digit in range(10):
        os.makedirs(os.path.join(output_dir, dataset_type, str(digit)), exist_ok=True)

# 处理训练集
train_data = MNIST(root='./data', train=True, download=True)
train_dataset = CustomMNISTDataset(train_data, background_images_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 处理测试集
test_data = MNIST(root='./data', train=False, download=True)
test_dataset = CustomMNISTDataset(test_data, background_images_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 保存训练集合成图片和GT
for idx, (images, gt, labels) in enumerate(train_loader):
    for i in range(images.size(0)):
        label = labels[i].item()
        img_path = os.path.join(output_dir, 'train', str(label), f'train_image_{idx * 32 + i}.png')
        gt_path = os.path.join(output_dir, 'train', str(label), f'train_gt_{idx * 32 + i}.png')
        Image.fromarray((images[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(img_path)
        Image.fromarray((gt[i].numpy()[0] * 255).astype(np.uint8)).save(gt_path)

# 保存测试集合成图片和GT
for idx, (images, gt, labels) in enumerate(test_loader):
    for i in range(images.size(0)):
        label = labels[i].item()
        img_path = os.path.join(output_dir, 'test', str(label), f'test_image_{idx * 32 + i}.png')
        gt_path = os.path.join(output_dir, 'test', str(label), f'test_gt_{idx * 32 + i}.png')
        Image.fromarray((images[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(img_path)
        Image.fromarray((gt[i].numpy()[0] * 255).astype(np.uint8)).save(gt_path)

print("所有图像和GT已分类保存到:", output_dir)
