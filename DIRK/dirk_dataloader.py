import os.path
import numpy as np
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
        self.X = X
        self.Y = Y
        self.num_classes = num_classes
        N = len(Y)
        self.given_partial_label_matrix = torch.zeros(N, num_classes)
        torch.manual_seed(1)
        np.random.seed(1)

        clip_preds = np.loadtxt(f"{data_path}/CLIP-L14/train_label_pre.txt", dtype=int)
        qwen_preds = np.loadtxt(f"{data_path}/Qwen_VL_7B_label/train_label_pre.txt", dtype=int)

        for i in range(N):
            clip_pred = int(clip_preds[i])
            qwen_pred = int(qwen_preds[i])
            true_label = int(Y[i])

            if not (0 <= clip_pred < num_classes):
                clip_pred = 0
            if not (0 <= qwen_pred < num_classes):
                qwen_pred = 0
            if not (0 <= true_label < num_classes):
                true_label = 0

            if clip_pred == qwen_pred:
                self.given_partial_label_matrix[i][clip_pred] = 1.0
            else:
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

        self.transform2 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform3 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomRotation(15),
            transforms.RandomCrop(32, 6, padding_mode='reflect'),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=24),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        each_image1 = self.transform1(x)  # img_w
        each_image2 = self.transform2(x)  # img_s
        each_image3 = self.transform3(x)  # img_distill
        each_label = self.given_partial_label_matrix[index]
        each_true_label = torch.Tensor(self.Y)[index]

        return each_image1, each_image2, each_image3, each_label, each_true_label.float(), index


class DatasetPartialize(Dataset):
    def __init__(self, X, Y, num_classes, data_path):
        self.X = X
        self.Y = Y
        N = len(Y)
        self.given_partial_label_matrix = torch.zeros(N, num_classes)
        torch.manual_seed(1)
        np.random.seed(1)

        clip_preds = np.loadtxt(f"{data_path}/CLIP-L14/train_label_pre.txt", dtype=int)
        qwen_preds = np.loadtxt(f"{data_path}/Qwen_VL_7B_label/train_label_pre.txt", dtype=int)

        for i in range(N):
            clip_pred = int(clip_preds[i])
            qwen_pred = int(qwen_preds[i])
            true_label = int(Y[i])

            if not (0 <= clip_pred < num_classes):
                clip_pred = 0
            if not (0 <= qwen_pred < num_classes):
                qwen_pred = 0
            if not (0 <= true_label < num_classes):
                true_label = 0

            if clip_pred == qwen_pred:
                self.given_partial_label_matrix[i][clip_pred] = 1.0
            else:
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

        self.transform2 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=32),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform3 = transforms.Compose([
            convert_to_rgb,
            transforms.RandomResizedCrop(64, scale=(0.3, 0.9)),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=24),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        each_image1 = self.transform1(x)  # img_w
        each_image2 = self.transform2(x)  # img_s
        each_image3 = self.transform3(x)  # img_distill
        each_label = self.given_partial_label_matrix[index]
        each_true_label = torch.Tensor(self.Y)[index]

        return each_image1, each_image2, each_image3, each_label, each_true_label.float(), index


def get_data_handler(args):
    dataset = args.dataset
    data_path = os.path.join(args.data_dir, dataset)
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = DatasetLoader(processor=None, model=None,
                                                                       dataset=dataset).read_data_cifar_100()
        datahandler = CIFAR100Partialize(train_data, train_label, num_classes=100, data_path=data_path)
    else:
        if dataset == 'tiny-imagenet-200':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(
                processor=None, model=None, dataset=dataset).read_data_tiny_imagenet_200()
        elif dataset == 'caltech-101':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(
                processor=None, model=None, dataset=dataset).read_data_caltech_101()
        elif dataset == 'food-101':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(
                processor=None, model=None, dataset=dataset).read_data_food_101()
        elif dataset == 'EuroSAT':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(
                processor=None, model=None, dataset=dataset).read_data_eruosat()
        elif dataset == 'dtd':
            train_data, train_label, test_data, test_label, num_classes = DatasetLoader(
                processor=None, model=None, dataset=dataset).read_data_dtd()

        datahandler = DatasetPartialize(train_data, train_label, num_classes=num_classes, data_path=data_path)
    return datahandler


def load_data(args):
    test_loader, _ = DatasetLoader(
        processor=None,
        model=None,
        dataset=args.dataset,
        model_name='clip',
        pattern="val",
        input_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers).load_datasets()

    partial_training_dataset = get_data_handler(args)
    partialY_matrix = partial_training_dataset.given_partial_label_matrix

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )


    return partial_training_dataloader, partialY_matrix, test_loader
