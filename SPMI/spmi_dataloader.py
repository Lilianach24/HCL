import os

import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from PaPi.utils import Cutout
from PaPi.autoaugment import CIFAR10Policy, ImageNetPolicy
from data_loader.data_loader import DatasetLoader


def convert_to_rgb(image):
    return image.convert("RGB")


class CIFAR100Partialize(Dataset):
    def __init__(self, X, Y, num_classes, data_path):
        self.X = X  # 图像数据
        self.Y = Y  # 真实标签
        self.num_classes = num_classes
        N = len(Y)
        self.given_partial_label_matrix = torch.zeros(N, num_classes)
        self.pre = False  # 控制是否使用伪标签

        # EMA平滑伪标签初始化
        self.pseudo_cand_labels = self.given_partial_label_matrix.clone()
        torch.manual_seed(1)
        np.random.seed(1)

        clip_preds = np.loadtxt(f"{data_path}/CLIP-L14/train_label_pre.txt", dtype=int)
        qwen_preds = np.loadtxt(f"{data_path}/Qwen_VL_7B_label/train_label_pre.txt", dtype=int)

        for i in range(N):
            clip_pred = int(clip_preds[i])
            qwen_pred = int(qwen_preds[i])
            true_label = int(Y[i])

            # 超出范围的默认改为0
            if not (0 <= clip_pred < num_classes):
                clip_pred = 0
            if not (0 <= qwen_pred < num_classes):
                qwen_pred = 0
            if not (0 <= true_label < num_classes):
                true_label = 0

            if clip_pred == qwen_pred:
                # 一致：只标记 clip_pred
                self.given_partial_label_matrix[i][clip_pred] = 1.0
            else:
                # 不一致：标记三者
                self.given_partial_label_matrix[i][clip_pred] = 1.0
                self.given_partial_label_matrix[i][qwen_pred] = 1.0
                self.given_partial_label_matrix[i][true_label] = 1.0

        self.transform1 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # self.transform2 = transforms.Compose([
        #     convert_to_rgb,
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, 4, padding_mode='reflect'),
        #     CIFAR10Policy(),
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     Cutout(n_holes=1, length=16),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        each_image = self.transform1(x)
        # each_image2 = self.transform2(x)
        each_label = self.given_partial_label_matrix[index]
        each_true_label = torch.Tensor(self.Y)[index]
        # each_true_label = self.Y[index]

        return each_image, each_label, each_label, each_true_label.float(), index

    def set_pseudo_cand_labels(self, cand_label_batch, index_batch, ema_theta=0.999):
        with torch.no_grad():
            index_batch_cpu = index_batch.cpu()
            cand_label_batch = cand_label_batch.cpu()
            
            # 更新伪标签
            self.pseudo_cand_labels[index_batch_cpu] = \
                ema_theta * self.pseudo_cand_labels[index_batch_cpu] + \
                (1 - ema_theta) * cand_label_batch


class DatasetPartialize(Dataset):
    def __init__(self, X, Y, num_classes, data_path):
        self.X = X
        self.Y = Y
        N = len(Y)
        self.given_partial_label_matrix = torch.zeros(N, num_classes)
        self.pre = False
        self.pseudo_cand_labels = self.given_partial_label_matrix.clone()
        torch.manual_seed(1)
        np.random.seed(1)

        clip_preds = np.loadtxt(f"{data_path}/CLIP-L14/train_label_pre.txt", dtype=int)
        qwen_preds = np.loadtxt(f"{data_path}/Qwen_VL_7B_label/train_label_pre.txt", dtype=int)

        for i in range(N):
            clip_pred = int(clip_preds[i])
            qwen_pred = int(qwen_preds[i])
            true_label = int(Y[i])

            # 超出范围的默认改为0
            if not (0 <= clip_pred < num_classes):
                clip_pred = 0
            if not (0 <= qwen_pred < num_classes):
                qwen_pred = 0
            if not (0 <= true_label < num_classes):
                true_label = 0

            if clip_pred == qwen_pred:
                # 一致：只标记 clip_pred
                self.given_partial_label_matrix[i][clip_pred] = 1.0
            else:
                # 不一致：标记三者
                self.given_partial_label_matrix[i][clip_pred] = 1.0
                self.given_partial_label_matrix[i][qwen_pred] = 1.0
                self.given_partial_label_matrix[i][true_label] = 1.0

        self.transform1 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # self.transform2 = transforms.Compose([
        #     convert_to_rgb,
        #     transforms.RandomResizedCrop(64),
        #     transforms.RandomHorizontalFlip(),
        #     ImageNetPolicy(),
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     Cutout(n_holes=1, length=32),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        each_image1 = self.transform1(x)
        # each_image2 = self.transform2(x)
        each_label = self.given_partial_label_matrix[index]
        each_true_label = torch.Tensor(self.Y)[index]

        return each_image1, each_label, each_label, each_true_label.float(), index

    def set_pseudo_cand_labels(self, cand_label_batch, index_batch, ema_theta=0.999):
        with torch.no_grad():
            index_batch_cpu = index_batch.cpu()
            self.pseudo_cand_labels[index_batch_cpu] = \
                ema_theta * self.pseudo_cand_labels[index_batch_cpu] + \
                (1 - ema_theta) * cand_label_batch.cpu()


class UnlabeledTrainingCIFAR100(Dataset):
    def __init__(self, data, true_labels, input_size, num_classes, device):
        """
        data: numpy [N, C, H, W]
        true_labels: list or numpy [N]
        """
        self.data = data
        self.true_labels = torch.tensor(true_labels, dtype=torch.long)
        self.num_classes = num_classes
        self.device = device

        self.transform = transforms.Compose([
            lambda x: Image.fromarray(np.transpose(x, (1, 2, 0))),  # BGR->RGB PIL
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # 后验概率，始终维护类别级别的后验
        self.posterior = torch.ones(num_classes, dtype=torch.float32) / num_classes
        
        # 样本级别的后验概率（仅用于存储，不直接使用）
        self.sample_posterior = None
        
        self.pseudo_complementary_labels = torch.zeros(len(data), num_classes, dtype=torch.float32)
        self.pre = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.transform(self.data[idx])
        
        # 始终使用类别级别的后验概率
        label = self.pseudo_complementary_labels[idx] if self.pre else self.posterior
        
        true_label = self.true_labels[idx]
        return img, label, true_label, idx

    def set_posterior(self, posterior_tensor):
        """设置后验概率，确保始终维护类别级别的后验概率"""
        if posterior_tensor.ndim == 1:
            # 直接设置类别后验概率
            self.posterior = posterior_tensor.clone()
            # 清空样本后验概率（如果存在）
            self.sample_posterior = None
        else:
            # 从样本后验概率计算类别后验概率
            self.sample_posterior = posterior_tensor.clone()
            self.posterior = torch.mean(posterior_tensor, dim=0)
            # print(f"[INFO] 从样本后验概率计算得到类别后验概率，形状: {self.posterior.shape}")

    def set_pseudo_complementary_labels(self, labels_tensor, indices_tensor, init=False):
        with torch.no_grad():
            # # 计算有效索引范围
            # if self.pseudo_complementary_labels is not None:
            #     max_valid_idx = len(self.pseudo_complementary_labels) - 1
            # else:
            #     max_valid_idx = len(self) - 1  # 使用数据集大小作为上限

            # # 检查是否存在无效索引
            # invalid_mask = (indices_tensor < 0) | (indices_tensor > max_valid_idx)
            # if torch.any(invalid_mask):
            #     invalid_count = torch.sum(invalid_mask).item()
            #     print(f"[WARNING] Found {invalid_count} invalid indices in {len(indices_tensor)}")
            #     print(
            #         f"[WARNING] Index range: min={torch.min(indices_tensor)}, max={torch.max(indices_tensor)}, valid_max={max_valid_idx}")

            #     # 过滤无效索引
            #     valid_mask = ~invalid_mask
            #     indices_tensor = indices_tensor[valid_mask]
            #     labels_tensor = labels_tensor[valid_mask]

            #     if len(indices_tensor) == 0:
            #         print("[WARNING] No valid indices left after filtering. Skipping pseudo label update.")
            #         return

            # 确保伪标签张量已初始化
            if init or (self.pseudo_complementary_labels is None):
                self.pseudo_complementary_labels = torch.zeros(len(self), self.num_classes)

            # 将张量移至CPU进行索引操作
            indices_cpu = indices_tensor.cpu()
            labels_cpu = labels_tensor.cpu()

            # 更新伪标签
            self.pseudo_complementary_labels[indices_cpu] = labels_cpu


class UnlabeledTrainingDatasetGeneric(Dataset):
    def __init__(self, data_paths, true_labels, input_size, num_classes, device):
        """
        data_paths: list of str (image file paths)
        true_labels: list or numpy [N]
        """
        self.data_paths = data_paths
        self.true_labels = torch.tensor(true_labels, dtype=torch.long)
        self.num_classes = num_classes
        self.device = device

        self.transform = transforms.Compose([
            lambda x: x.convert("RGB"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(64),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # 后验概率，始终维护类别级别的后验
        self.posterior = torch.ones(num_classes, dtype=torch.float32) / num_classes
        
        # 样本级别的后验概率（仅用于存储，不直接使用）
        self.sample_posterior = None
        
        self.pseudo_complementary_labels = torch.zeros(len(data_paths), num_classes, dtype=torch.float32)
        self.pre = False

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img = Image.open(self.data_paths[idx])
        img = self.transform(img)
        
        # 始终使用类别级别的后验概率
        label = self.pseudo_complementary_labels[idx] if self.pre else self.posterior
        
        true_label = self.true_labels[idx]
        return img, label, true_label, idx

    def set_posterior(self, posterior_tensor):
        """设置后验概率，确保始终维护类别级别的后验概率"""
        if posterior_tensor.ndim == 1:
            # 直接设置类别后验概率
            self.posterior = posterior_tensor.clone()
            # 清空样本后验概率（如果存在）
            self.sample_posterior = None
        else:
            # 从样本后验概率计算类别后验概率
            self.sample_posterior = posterior_tensor.clone()
            self.posterior = torch.mean(posterior_tensor, dim=0)
            # print(f"[INFO] 从样本后验概率计算得到类别后验概率，形状: {self.posterior.shape}")

    def set_pseudo_complementary_labels(self, labels_tensor, indices_tensor, init=False):
        with torch.no_grad():
            indices_tensor_cpu = indices_tensor.cpu()
            labels_tensor_cpu = labels_tensor.cpu()
            if init or self.pseudo_complementary_labels is None:
                self.pseudo_complementary_labels = torch.zeros(len(self), self.num_classes)
            self.pseudo_complementary_labels[indices_tensor_cpu] = labels_tensor_cpu


def get_data_handler(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = args.dataset
    data_path = os.path.join(args.data_dir, dataset)
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = DatasetLoader(processor=None, model=None, dataset=dataset).read_data_cifar_100()
        datahandler1 = CIFAR100Partialize(train_data, train_label, num_classes=100, data_path=data_path)
        datahandler2 = UnlabeledTrainingCIFAR100(train_data, train_label, input_size=224, num_classes=100, device=device)
    else:
        if dataset == 'tiny-imagenet-200':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(processor=None, model=None, dataset=dataset).read_data_tiny_imagenet_200()
        elif dataset == 'stanford_cars':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(processor=None, model=None, dataset=dataset).read_data_stanford_cars()
        elif dataset == 'caltech-101':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(processor=None, model=None, dataset=dataset).read_data_caltech_101()
        elif dataset == 'food-101':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(processor=None, model=None, dataset=dataset).read_data_food_101()
        elif dataset == 'EuroSAT':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(
                processor=None, model=None, dataset=dataset).read_data_eruosat()

        datahandler1 = DatasetPartialize(train_data, train_label, num_classes=num_classes, data_path=data_path)
        datahandler2 = UnlabeledTrainingDatasetGeneric(train_data, train_label, 224, num_classes, device)
    return datahandler1, datahandler2


def load_data(args):
    test_loader, _ = DatasetLoader(
        processor=None,
        model=None,
        dataset=args.dataset,
        model_name='clip',
        pattern="val",
        input_size=224,
        batch_size=args.batch_size * 4,
        num_workers=args.num_workers).load_datasets()

    partial_training_dataset, unlabeled_dataset = get_data_handler(args)
    partialY_matrix = partial_training_dataset.given_partial_label_matrix

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # 创建无标签训练数据加载器
    unlabeled_training_dataloader = torch.utils.data.DataLoader(
        dataset=unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    return partial_training_dataloader, partialY_matrix, unlabeled_training_dataloader, test_loader
