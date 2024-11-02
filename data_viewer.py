import torch
from data import CustomMNISTDataset



# 加载训练集
train_dataset = torch.load('train_dataset.pt')

# 加载测试集
test_dataset = torch.load('test_dataset.pt')

# 可以通过索引访问数据集中的元素
example_image, example_segmentation_gt, example_label = train_dataset[0]

print(f"Example label: {example_label}")