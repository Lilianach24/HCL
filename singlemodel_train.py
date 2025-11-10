import argparse
import csv
import logging
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_loader.data_loader import DatasetLoader
from model.loss import c_loss
from model.model import CLIPLinearModel
from utils.utils import set_seed, get_num_classes


class SingleModelDatasetWithProbs(Dataset):
    def __init__(self, root_dir, dataset_name, model='clip', pattern='train', input_size=224):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.model = model
        self.pattern = pattern
        self.input_size = input_size

        # 实例化 DatasetLoader 并调用对应数据加载方法
        self.loader = self._create_dataset_loader()
        # 根据模式加载不同数据
        if pattern == 'train':
            self.images, self.labels = self._load_train_data()
        else:  # test/val 模式
            self.images, self.labels = self._load_test_data()
            # self.s_values = None  # 测试集不需要 s_values
            # self.label_probs = None  # 测试集不需要 label_probs

        # 定义数据预处理
        self.transform = self._get_transform()

    def _create_dataset_loader(self):
        """创建 DatasetLoader 实例"""
        return DatasetLoader(
            root=self.root_dir,
            model=None,
            dataset=self.dataset_name,
            model_name=self.model,
            pattern=self.pattern,
            input_size=self.input_size,
            processor=None
        )

    def _load_train_data(self):
        """加载图像、标签、s_values、label_probs"""
        num_classes = 100
        if self.dataset_name == "CIFAR100":
            # 调用 DatasetLoader 的 CIFAR100 加载逻辑
            train_imgs, _, _, _ = self.loader.read_data_cifar_100()
        elif self.dataset_name == "tiny-imagenet-200":
            train_imgs, _, _, _, num_classes = self.loader.read_data_tiny_imagenet_200()
        elif self.dataset_name == "stanford_cars":
            train_imgs, _, _, _, num_classes = self.loader.read_data_stanford_cars()
        elif self.dataset_name == "caltech-101":
            train_imgs, _, _, _, num_classes = self.loader.read_data_caltech_101()
        elif self.dataset_name == "food-101":
            train_imgs, _, _, _, num_classes = self.loader.read_data_food_101()
        elif self.dataset_name == 'EuroSAT':
            train_imgs, _, _, _, num_classes = self.loader.read_data_eruosat()
        elif self.dataset_name == 'cifar10':
            train_imgs, _, _, _, num_classes = self.loader.read_data_cifar10()
        elif self.dataset_name == 'fmnist':
            train_imgs, _, _, _, num_classes = self.loader.read_data_fmnist()
        elif self.dataset_name == 'dtd':
            train_imgs, _, _, _, num_classes = self.loader.read_data_dtd()
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

        if self.model == 'clip':
            pred_labels_path = os.path.join(self.root_dir, self.dataset_name, 'CLIP-L14', 'train_label_pre.txt')
        elif self.model == 'qwen':
            pred_labels_path = os.path.join(self.root_dir, self.dataset_name, 'Qwen_VL_7B_label', 'train_label_pre.txt')

        with open(pred_labels_path, 'r') as f:
            pred_labels = [int(line.strip()) for line in f]

        pred_labels = [label if 0 <= label < num_classes else 0 for label in pred_labels]

        return train_imgs, pred_labels

    def _load_test_data(self):
        """加载测试数据（仅图像和真实标签）"""
        if self.dataset_name == "CIFAR100":
            _, _, test_data, test_label = self.loader.read_data_cifar_100()
            images, labels = test_data, test_label
        elif self.dataset_name == "tiny-imagenet-200":
            _, _, test_imgs, test_labels, _ = self.loader.read_data_tiny_imagenet_200()
            images, labels = test_imgs, test_labels
        elif self.dataset_name == "stanford_cars":
            _, _, test_imgs, test_labels, _ = self.loader.read_data_stanford_cars()
            images, labels = test_imgs, test_labels
        elif self.dataset_name == "caltech-101":
            _, _, test_imgs, test_labels, _ = self.loader.read_data_caltech_101()
            images, labels = test_imgs, test_labels
        elif self.dataset_name == "food-101":
            _, _, test_imgs, test_labels, _ = self.loader.read_data_food_101()
            images, labels = test_imgs, test_labels
        elif self.dataset_name == 'EuroSAT':
            _, _, test_imgs, test_labels, _ = self.loader.read_data_eruosat()
            images, labels = test_imgs, test_labels
        elif self.dataset_name == 'cifar10':
            _, _, test_imgs, test_labels, _ = self.loader.read_data_cifar10()
            images, labels = test_imgs, test_labels
        elif self.dataset_name == 'fmnist':
            _, _, test_imgs, test_labels, _ = self.loader.read_data_fmnist()
            images, labels = test_imgs, test_labels
        elif self.dataset_name == 'dtd':
            _, _, test_imgs, test_labels, _ = self.loader.read_data_dtd()
            images, labels = test_imgs, test_labels
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

        return images, labels

    def _get_transform(self):
        """定义数据预处理（适配 PIL 图像和 numpy 数组）"""
        return transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),  # 自动处理 PIL 图像或 numpy 数组
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """根据模式返回不同数据"""
        img_data = self.images[idx]
        label = self.labels[idx]

        # 处理图像数据
        if isinstance(img_data, np.ndarray):
            image = Image.fromarray(img_data.transpose(1, 2, 0))  # numpy → PIL
        else:
            image = Image.open(img_data).convert("RGB")  # 路径 → PIL

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        return image, label  # 测试集只返回图像和真实标签

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
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)
    output_path = os.path.join(opt.output_path, dataset_name)
    os.makedirs(output_path, exist_ok=True)
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logs = f"{output_path}/{dataset_name}_{t}.csv"
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
    train_dataset = SingleModelDatasetWithProbs(
        root_dir=root_dir,
        dataset_name=dataset_name,
        model=opt.model_name,
        pattern='train',
        input_size=input_size,
    )

    # 创建测试集（使用 BaseDatasetHandler 或类似类）
    test_dataset = SingleModelDatasetWithProbs(
        root_dir=root_dir,
        dataset_name=dataset_name,
        model=opt.model_name,
        pattern='val',
        input_size=input_size,
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 创建模型
    model = CLIPLinearModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.linear.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


    # 训练模型
    print("开始训练模型...")
    train(opt, model, train_loader, test_loader, optimizer, scheduler, device, logging, logs)

def valid(model, test_loader, device):
    # 测试阶段
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device).float()
            labels = labels.to(device)

            _, outputs = model(images)
            loss = c_loss(outputs, labels)  # 测试时使用简单损失函数

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    # 计算测试集准确率
    test_acc = 100. * test_correct / test_total

    return test_loss, test_acc

def train(opt, model, train_loader, test_loader, optimizer, scheduler, device, logging, logs):
    best_test_acc = 0.0

    for epoch in range(opt.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            _, outputs = model(images)
            loss = c_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                logging.info(f'Epoch: {epoch + 1}/{opt.num_epochs}, Batch: {batch_idx + 1}/{len(train_loader)}, '
                      f'Train Loss: {train_loss / (batch_idx + 1):.4f}, Train Acc: {100. * train_correct / train_total:.2f}%')

        # 学习率调整
        if scheduler:
            scheduler.step()

        # 测试阶段
        test_loss, test_acc = valid(model, test_loader, device)

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # self._save_model(epoch)

        # 打印本轮训练和测试结果
        logging.info(f'Epoch: {epoch + 1}/{opt.num_epochs}, '
              f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {100. * train_correct / train_total:.2f}%, '
              f'Test Loss: {test_loss / len(test_loader):.4f}, Test Acc: {test_acc:.2f}%')
        # 保存每一轮的准确率
        with open(logs, "a") as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(["Epoch", "TrainAcc", "TestAcc", "TrainLoss", "TestLoss"])
            writer.writerow([epoch + 1, 100. * train_correct / train_total, test_acc, train_loss / len(train_loader), test_loss / len(test_loader)])
    #保存最终训练得到的模型
    # self._save_model()
    print(f'Best Test Accuracy: {best_test_acc:.2f}%')


if __name__ == "__main__":
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, dest='root_dir', default='./datasets', help='dataset root dir')
    # 指定训练数据集
    parser.add_argument('--dataset_name', type=str, default='CIFAR100',
                        dest='dataset_name', required=False, help='dataset name')
    # 指定批次大小
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=64, help='batch size')
    # 指定训练轮数
    parser.add_argument('--num_epochs', type=int, default=30, dest='num_epochs', help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, dest='learning_rate', help='learning rate')
    parser.add_argument('--output_path', type=str, default='./result', help='output path')
    parser.add_argument('--model_name', type=str, default='clip', choices=['clip', 'qwen'],
                        help='model used for prediction labels')
    opt = parser.parse_args()
    main(opt)