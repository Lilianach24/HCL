import argparse
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from data_loader.data_loader import CustomDatasetWithProbs
from model.loss import c_loss
from model.model import CLIPLinearModel
from utils.utils import set_seed, get_num_classes


# 主函数
def main(opt):
    # 固定随机种子
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置参数
    root_dir = opt.root_dir  # 数据集根目录
    dataset_name = opt.dataset_name  # 数据集名称
    num_classes = get_num_classes(dataset_name, root_dir)  # 类别数
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    learning_rate = float(opt.learning_rate)
    input_size = 224
    # 准备日志文件夹
    # logs = set_log('./Saved', dataset_name)
    # output_path = logs['logs']
    output_path = os.path.join("manual_result", dataset_name)
    os.makedirs(output_path, exist_ok=True)
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[
                            logging.FileHandler('./{}/result_{}_{}.log'.
                                                format(output_path, opt.dataset_name, t)),
                            logging.StreamHandler()
                        ])
    logging.info(opt.__dict__)

    # 创建训练集
    train_dataset = CustomDatasetWithProbs(
        root_dir=root_dir,
        dataset_name=dataset_name,
        pattern='train',
        input_size=input_size,
        conflict_only = True
    )

    # 创建测试集（使用 BaseDatasetHandler 或类似类）
    test_dataset = CustomDatasetWithProbs(
        root_dir=root_dir,
        dataset_name=dataset_name,
        pattern='val',  # 假设 DatasetLoader 支持 'val' 模式
        input_size=input_size,
        conflict_only = True
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 创建模型
    model = CLIPLinearModel(num_classes=num_classes).to(device)
    # 定义优化器
    optimizer = torch.optim.AdamW(model.linear.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练模型
    print("开始训练模型...")
    train_s1_only(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dataset_name=dataset_name,
        output_path=output_path,
        num_epochs=opt.num_epochs
    )


def validate(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device).float()
            labels = labels.to(device)

            _, outputs = model(images)
            loss = c_loss(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    return test_loss / len(test_loader), test_acc

def train_s1_only(model, train_loader, test_loader, optimizer, scheduler, device, dataset_name, output_path,
                  num_epochs=30):
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels, s_values, label_probs) in enumerate(train_loader):
            images = images.to(device).float()
            labels = labels.to(device)
            s_values = s_values.to(device)
            label_probs = label_probs.to(device).float()

            # 只保留 s=1 的冲突样本
            mask = (s_values == 1)
            if mask.sum() == 0:
                continue  # 本 batch 无 s=1 样本，跳过

            images = images[mask]
            labels = labels[mask]
            label_probs = label_probs[mask]  # 如果需要，可用于损失函数

            optimizer.zero_grad()
            _, outputs = model(images)
            loss = c_loss(outputs, labels)  # 使用标准交叉熵即可
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                logging.info(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                    f'Train Loss: {train_loss / (batch_idx + 1):.4f}, '
                    f'Train Acc: {100. * train_correct / train_total:.2f}%')

        if scheduler:
            scheduler.step()

        test_loss, test_acc = validate(model, test_loader, device)

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # save_model(model, output_path, dataset_name, test_acc)

        logging.info(
            f'Epoch [{epoch + 1}/{num_epochs}] Completed - '
            f'Train Loss: {train_loss / len(train_loader):.4f}, '
            f'Train Acc: {100. * train_correct / train_total:.2f}%, '
            f'Test Loss: {test_loss:.4f}, '
            f'Test Acc: {test_acc:.2f}%')

    # 保存最终训练得到的模型
    # save_model(model, output_path, dataset_name, test_acc)
    print(f'完成训练，Best Test Accuracy: {best_test_acc:.2f}%')

def save_model(model, output_path, dataset_name, test_acc):
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_path = f"{output_path}/checkpoint_{dataset_name}_{t}_{test_acc:.2f}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")


if __name__ == "__main__":
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, dest='root_dir', default='./datasets', help='dataset root dir')
    # 指定预训练模型的文件路径。如果该参数不为空，程序会加载这个路径下的模型权重，以便继续训练整个模型
    # 当你想要在已有模型的基础上继续训练时，可以使用这个参数指定预训练模型的位置。
    parser.add_argument('--pretrained_model', type=str, default='',
                        dest='pretrained_model', required=False, help='continue to train model')
    # 指定训练数据集
    parser.add_argument('--dataset_name', type=str, default='CIFAR100',
                        dest='dataset_name', required=False, help='dataset name')
    # 指定批次大小
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=64, help='batch size')
    # 指定训练轮数
    parser.add_argument('--num_epochs', type=int, default=30, dest='num_epochs', help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, dest='learning_rate', help='learning rate')
    opt = parser.parse_args()
    main(opt)