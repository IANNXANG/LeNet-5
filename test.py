import torch
from torch.utils.data import DataLoader
from FCN import FCN  # 确保引入你的 FCN 模型
from data import CustomMNISTDataset  # 确保引入自定义数据集类
import numpy as np




def compute_iou(predictions, targets):
    predictions = predictions.cpu().numpy().astype(np.uint8)
    targets = targets.cpu().numpy().astype(np.uint8)

    intersection = np.sum(predictions & targets)
    union = np.sum(predictions | targets)

    return intersection / (union + 1e-6)


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
            predicted = torch.sigmoid(outputs) > 0.5

            iou = compute_iou(predicted, gt > 0.5)
            accuracy = compute_accuracy(predicted, gt > 0.5)

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

    model = FCN().to(device)  # 将模型放到 GPU
    model.load_state_dict(torch.load('fcn_model.pth'))
    test(model, test_loader)
    print("测试完成.")
