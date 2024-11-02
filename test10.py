import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import CustomMNISTDataset  # 假设自定义数据集的实现
from FCN import FCN10  # 假设你的模型在此文件中
import numpy as np

def calculate_iou(preds, targets, num_classes):
    iou = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum().item()
        union = ((preds == cls) | (targets == cls)).sum().item()
        iou.append(intersection / union if union != 0 else 0)
    return sum(iou) / num_classes  # 返回平均 IoU

def calculate_mean_accuracy(preds, targets, num_classes):
    accuracy = []
    for cls in range(num_classes):
        correct = ((preds == cls) & (targets == cls)).sum().item()
        total = (targets == cls).sum().item()
        accuracy.append(correct / total if total != 0 else 0)
    return sum(accuracy) / num_classes  # 返回平均准确率

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据集
    test_dataset = torch.load('test_dataset.pt')  # 加载处理后的测试数据集
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = FCN10(num_classes=10).to(device)
    model.load_state_dict(torch.load('fcn10_model.pth'))  # 加载训练好的模型
    model.eval()  # 设置模型为评估模式

    all_preds = []
    all_gt = []

    with torch.no_grad():
        for images, gt, labels in test_loader:
            images = images.to(device)
            gt = gt.to(device)
            outputs = model(images)

            # 取得每个像素点的预测类别
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())
            all_gt.append(gt.cpu().numpy())

    # 将所有预测结果和真实标签合并
    all_preds = np.concatenate(all_preds)
    all_gt = np.concatenate(all_gt)

    # 计算IoU和平均准确率
    mean_iou = calculate_iou(all_preds, all_gt, num_classes=10)
    mean_accuracy = calculate_mean_accuracy(all_preds, all_gt, num_classes=10)

    print(f'Mean IoU: {mean_iou:.4f}')
    print(f'Mean Accuracy: {mean_accuracy:.4f}')

if __name__ == "__main__":
    test()
