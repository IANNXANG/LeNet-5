import torch
from torch.utils.data import DataLoader
from FCN import FCN  # 确保引入你的 FCN 模型
from data import CustomMNISTDataset  # 确保引入自定义数据集类
import numpy as np


def compute_iou(predictions, targets, num_classes=10):
    iou = []
    for cls in range(num_classes):
        pred_cls = (predictions == cls).astype(np.uint8)
        target_cls = (targets == cls).astype(np.uint8)

        intersection = np.sum(pred_cls & target_cls)
        union = np.sum(pred_cls | target_cls)

        iou.append(intersection / (union + 1e-6))  # 防止除以零
    return np.mean(iou)


def compute_accuracy(predictions, targets):
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total


def test(model, test_loader):
    model.eval()
    total_iou = 0
    total_accuracy = 0
    num_batches = 0

    with torch.no_grad():
        for images, gt, _ in test_loader:
            images, gt = images.to(device), gt.to(device)  # 将数据放到 GPU
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)  # 获取类别索引

            # 计算 IoU 和准确率
            iou = compute_iou(predicted.cpu().numpy(), gt.argmax(dim=1).cpu().numpy())
            accuracy = compute_accuracy(predicted, gt.argmax(dim=1))

            total_iou += iou
            total_accuracy += accuracy
            num_batches += 1

    mean_iou = total_iou / num_batches
    mean_accuracy = total_accuracy / num_batches

    print(f'测试结果: 平均 IoU: {mean_iou:.4f}, 平均准确率: {mean_accuracy:.4f}')


if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据集
    test_dataset = torch.load('test_dataset.pt')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = FCN(num_classes=10).to(device)  # 确保模型是10个类别
    model.load_state_dict(torch.load('fcn_model.pth'))
    test(model, test_loader)
    print("测试完成.")
