import torch
from torch.utils.data import DataLoader
from LeNet5 import UpLeNet5, DeConLeNet5, DeConLeNet5Large
from data import CustomMNISTDataset  # 确保引入自定义数据集类
import numpy as np
import matplotlib.pyplot as plt  # 导入可视化库

model_list = ['UpLeNet5', 'DeConLeNet5', 'DeConLeNet5Large']  # 保存要测试的模型
#model_list = ['DeConLeNet5']

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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


def visualize(images, predictions, targets, num_images=5):
    """可视化原图、预测结果和真实结果"""
    plt.figure(figsize=(15, 5))

    for i in range(min(num_images, images.shape[0])):  # 确保不会超出数组范围
        plt.subplot(3, num_images, i + 1)
        # 对于输入图像，需要选择一个通道（如第一个通道）来显示
        plt.imshow(images[i].cpu().numpy().transpose(1, 2, 0), cmap='gray')  # 转换为(H, W, C)
        plt.title("输入图片")
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(predictions[i].cpu().numpy().squeeze(0), cmap='gray')  # 移除通道维度
        plt.title("分割结果")
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(targets[i].cpu().numpy().squeeze(0), cmap='gray')  # 移除通道维度
        plt.title("Ground Truth")
        plt.axis('off')

    plt.show()


def test(model, test_loader):
    model.eval()
    total_iou = 0
    total_accuracy = 0
    num_batches = 0

    with torch.no_grad():
        correct = 0
        total = 0
        for images, gt, labels in test_loader:
            images, gt, labels = images.to(device), gt.to(device), labels.to(device)  # 将数据放到 GPU
            x, outputs = model(images)
            _, predicted_cls = torch.max(x.data, 1)
            total += labels.size(0)
            correct += (predicted_cls == labels).sum().item()
            predicted = torch.sigmoid(outputs) > 0.5

            iou = compute_iou(predicted, gt > 0.5)
            accuracy = compute_accuracy(predicted, gt > 0.5)

            total_iou += iou
            total_accuracy += accuracy
            num_batches += 1

            # 可视化部分
            if num_batches == 100:  # 只可视化一个批次
                visualize(images, predicted, gt)

    class_accuracy = correct / total
    mean_iou = total_iou / num_batches
    mean_accuracy = total_accuracy / num_batches

    print(f'测试结果: 平均 IoU: {mean_iou:.4f}, 平均准确率: {mean_accuracy:.4f}, 分类准确率: {class_accuracy:.4f}')


if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据集
    test_dataset = torch.load('test_dataset.pt')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for model_name in model_list:
        if model_name == 'UpLeNet5':
            model = UpLeNet5().to(device)  # 将模型放到 GPU
        elif model_name == 'DeConLeNet5':
            model = DeConLeNet5().to(device)  # 将模型放到 GPU
        elif model_name == 'DeConLeNet5Large':
            model = DeConLeNet5Large().to(device)  # 将模型放到 GPU

        model.load_state_dict(torch.load(f'saved_model/{model_name}_10_model.pth'))
        test(model, test_loader)
        print(f"{model_name}测试完成.")
