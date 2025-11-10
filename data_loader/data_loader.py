import os
import struct

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pickle
import random
import glob
import scipy.io
from typing import Tuple, Dict
import numpy as np
from torchvision import transforms

from labeling.clip_labeling_classes import CIFAR100_generate_label_clip, Dataset_gengerate_label_clip
from labeling.deepseek_labeling_classes import CIFAR100_handler_train_DP, DatasetHandlerTrainDP
from labeling.qwen_labeling_classes import CIFAR100_handler_train_TF_saved, DatasetHandlerTrainClip
from utils.utils import get_transform

class CustomDatasetWithProbs(Dataset):
    def __init__(self, root_dir, dataset_name, model='clip', pattern='train', input_size=224, conflict_only=False):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.model = model
        self.pattern = pattern
        self.input_size = input_size

        # 实例化 DatasetLoader 并调用对应数据加载方法
        self.loader = self._create_dataset_loader()
        # 根据模式加载不同数据
        if pattern == 'train':
            self.images, self.labels, self.s_values, self.label_prob = self._load_train_data()

            # 仅使用 s=1 冲突样本
            if conflict_only:
                conflict_indices = (self.s_values == 1).nonzero(as_tuple=True)[0]
                self.images = [self.images[i] for i in conflict_indices]
                self.labels = [self.labels[i] for i in conflict_indices]
                self.s_values = self.s_values[conflict_indices]
                self.label_prob = self.label_prob[conflict_indices]
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
        if self.dataset_name == "CIFAR100":
            # 调用 DatasetLoader 的 CIFAR100 加载逻辑
            train_data, train_label, _, _ = self.loader.read_data_cifar_100()
            images, labels = train_data, train_label
        elif self.dataset_name == "tiny-imagenet-200":
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_tiny_imagenet_200()
            images, labels = train_imgs, train_labels
        elif self.dataset_name == "stanford_cars":
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_stanford_cars()
            images, labels = train_imgs, train_labels
        elif self.dataset_name == "caltech-101":
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_caltech_101()
            images, labels = train_imgs, train_labels
        elif self.dataset_name == "food-101":
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_food_101()
            images, labels = train_imgs, train_labels
        elif self.dataset_name == 'EuroSAT':
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_eruosat()
            images, labels = train_imgs, train_labels
        elif self.dataset_name == 'cifar10':
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_cifar10()
            images, labels = train_imgs, train_labels
        elif self.dataset_name == 'fmnist':
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_fmnist()
            images, labels = train_imgs, train_labels
        elif self.dataset_name == 'dtd':
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_dtd()
            images, labels = train_imgs, train_labels
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

        # 加载 s_values（标注一致性标记）
        s_values = self._load_s_values()

        # 加载 label_probs.npy
        label_probs_path = os.path.join(self.root_dir, self.dataset_name, 'CLIP-L14/label_prob.npy')
        label_probs = np.load(label_probs_path)

        return images, labels, s_values, label_probs

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

    def _load_s_values(self):
        """从 corrected_mask.txt 加载 s 值"""
        mask_file = os.path.join(self.root_dir, self.dataset_name, "corrected_mask.txt")
        if not os.path.exists(mask_file):
            print("文件不存在")

        with open(mask_file, "r") as f:
            s_values = [int(line.strip()) for line in f]
        return torch.tensor(s_values, dtype=torch.int64)
        # else:
        #     return torch.zeros(len(self.images), dtype=torch.int64)  # 默认全为0（标注一致）

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

        # 根据模式返回不同数据
        if self.pattern == 'train':
            return image, label, self.s_values[idx], torch.from_numpy(self.label_prob[idx]).float()
        else:
            return image, label  # 测试集只返回图像和真实标签

class BaseDatasetHandler(Dataset):
    """基础的数据集处理类，定义了通用的 __init__、__getitem__ 和 __len__ 方法"""
    def __init__(self, X, Y, input_size):
        self.X = X
        self.Y = Y
        # self.input_size = input_size
        self.transform = get_transform(input_size)

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class CIFAR100DatasetHandler(BaseDatasetHandler):
    """继承自 BaseDatasetHandler，专门处理 CIFAR-100 数据集。"""
    def __init__(self, X, Y, input_size):
        super().__init__(X, Y, input_size)

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        return x, y


class DatasetLoader:
    """包含所有的数据加载和处理静态方法，如读取不同数据集、获取数据处理类、生成数据加载器等"""
    def __init__(self, processor, model, root='datasets', dataset='CIFAR100', model_name='clip', pattern='train', input_size=224, batch_size=64, num_workers=0, ):
        self.root = root
        self.dataset = dataset
        self.data_path = os.path.join(self.root, self.dataset)
        self.model = model
        self.model_name = model_name
        self.pattern = pattern
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processor = processor

    def load_cifar100(self):
        with open(os.path.join(self.data_path, "train"), 'rb') as f:
            data_train = pickle.load(f, encoding='latin1')
        with open(os.path.join(self.data_path, "test"), 'rb') as f:
            data_test = pickle.load(f, encoding='latin1')
        with open(os.path.join(self.data_path, "meta"), 'rb') as f:
            data_meta = pickle.load(f, encoding='latin1')
        return data_train, data_test, data_meta

    def read_data_cifar_100(self):
        # random.seed(1)
        data_train, data_test, data_meta = self.load_cifar100()
        train_data = data_train['data'].reshape((data_train['data'].shape[0], 3, 32, 32))  # .transpose((0,1,3,2))
        test_data = data_test['data'].reshape((data_test['data'].shape[0], 3, 32, 32))  # .transpose((0,1,3,2))
        train_label = data_train["fine_labels"]
        test_label = data_test["fine_labels"]

        return train_data, train_label, test_data, test_label

    def read_data_food_101(self):
        id_dict = {}
        for i, line in enumerate(open(f'{self.data_path}/food-101_ram_taglist.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        num_classes = len(id_dict)
        train_imgs = []
        train_labels = []
        test_imgs = []
        test_labels = []
        with open(f'{self.data_path}/meta/train.txt', 'r') as f:
            for line in f:
                image = line.replace('\n', '') + '.jpg'
                label = line.split('/')[0]
                train_imgs.append(os.path.join(self.data_path, 'images', image).replace('\\', '/'))
                train_labels.append(id_dict[label])
        with open(f'{self.data_path}/meta/test.txt', 'r') as f:
            for line in f:
                image = line.replace('\n', '') + '.jpg'
                label = line.split('/')[0]
                test_imgs.append(os.path.join(f'{self.data_path}/images', image).replace('\\', '/'))
                test_labels.append(id_dict[label])

        return train_imgs, train_labels, test_imgs, test_labels, num_classes

    def read_data_stanford_cars(self):
        id_dict = {}
        for i, line in enumerate(open(f'{self.data_path}/stanford_cars_ram_taglist.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        num_classes = len(id_dict)
        data = scipy.io.loadmat(f'{self.data_path}/cars_annos.mat')
        annotations = data['annotations']
        train_imgs = []
        train_labels = []
        test_imgs = []
        test_labels = []
        for i in range(annotations.shape[1]):
            name = str(annotations[0, i][0])[2:-2]
            img_path = os.path.join(self.data_path, name).replace('\\', '/')
            clas = int(annotations[0, i][5])
            test = int(annotations[0, i][6])
            if test == 0:
                train_imgs.append(img_path)
                train_labels.append(clas - 1)
            elif test == 1:
                test_imgs.append(img_path)
                test_labels.append(clas - 1)

        return train_imgs, train_labels, test_imgs, test_labels, num_classes

    def read_data_eruosat(self):
        id_dict = {}
        for i, line in enumerate(open(f'{self.data_path}/EuroSAT_ram_taglist.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        num_classes = len(id_dict)
        EuroSAT_imgs = glob.glob(f"{self.data_path}/raw/2750/*/*.jpg")
        EuroSAT_imgs = [img_path.replace('\\', '/') for img_path in EuroSAT_imgs]
        EuroSAT_labels = [id_dict[img_path.split('/')[5]] for img_path in EuroSAT_imgs]
        EuroSAT_dataset = list(zip(EuroSAT_imgs, EuroSAT_labels))
        random.seed(0)
        random.shuffle(EuroSAT_dataset)
        train_size = int(0.7 * len(EuroSAT_dataset))
        train_set, test_set = EuroSAT_dataset[:train_size], EuroSAT_dataset[train_size:]
        train_imgs, train_labels = zip(*train_set)
        test_imgs, test_labels = zip(*test_set)

        return list(train_imgs), list(train_labels), list(test_imgs), list(test_labels), num_classes

    def read_data_tiny_imagenet_200(self):
        id_dict = {}
        for i, line in enumerate(open(os.path.join(self.data_path, 'wnids.txt'), 'r')):
            id_dict[line.replace('\n', '')] = i
        num_classes = len(id_dict)
        cls_dic = {}
        for i, line in enumerate(open(os.path.join(self.data_path, 'val/val_annotations.txt'), 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            cls_dic[img] = id_dict[cls_id]

        train_imgs = glob.glob(f"{self.data_path}/train/*/*/*.JPEG")
        test_imgs = glob.glob(f"{self.data_path}/val/images/*.JPEG")
        train_imgs = [img_path.replace('\\', '/') for img_path in train_imgs]
        test_imgs = [img_path.replace('\\', '/') for img_path in test_imgs]

        train_labels = [id_dict[train_img.split('/')[4]] for train_img in train_imgs]
        test_labels = [cls_dic[os.path.basename(test_img)] for test_img in test_imgs]

        return train_imgs, train_labels, test_imgs, test_labels, num_classes

    def read_data_caltech_101(self):
        id_dict = {}
        for i, line in enumerate(open(f'{self.data_path}/caltech-101_ram_taglist.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        num_classes = len(id_dict)
        caltech_101_imgs = glob.glob(f"{self.data_path}/101_ObjectCategories/*/*.jpg")
        caltech_101_imgs = [img_path.replace('\\', '/') for img_path in caltech_101_imgs]
        caltech_101_labels = [id_dict[img_path.split('/')[4]] for img_path in caltech_101_imgs]
        caltech_101_dataset = list(zip(caltech_101_imgs, caltech_101_labels))
        random.seed(0)
        random.shuffle(caltech_101_dataset)
        train_size = int(0.7 * len(caltech_101_dataset))
        train_set, test_set = caltech_101_dataset[:train_size], caltech_101_dataset[train_size:]
        train_imgs, train_labels = zip(*train_set)
        test_imgs, test_labels = zip(*test_set)
        return list(train_imgs), list(train_labels), list(test_imgs), list(test_labels), num_classes

    def read_data_fmnist(self):
        def read_idx_images(filename):
            with open(filename, 'rb') as f:
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
            return images

        def read_idx_labels(filename):
            with open(filename, 'rb') as f:
                magic, num = struct.unpack(">II", f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

        tmp_save_dir = os.path.join(self.data_path, 'tmp_fmnist_jpg')
        os.makedirs(tmp_save_dir, exist_ok=True)

        # 读取训练集
        train_images = read_idx_images(os.path.join(self.data_path, 'train-images-idx3-ubyte'))
        train_labels = read_idx_labels(os.path.join(self.data_path, 'train-labels-idx1-ubyte'))

        train_imgs_paths, train_labels_list = [], []
        for idx in range(len(train_labels)):
            img_pil = Image.fromarray(train_images[idx], mode='L').resize((32, 32), Image.Resampling.BICUBIC)
            img_path = os.path.join(tmp_save_dir, f"train_{idx}.jpg")
            img_pil.save(img_path, quality=70)
            train_imgs_paths.append(img_path.replace('\\', '/'))
            train_labels_list.append(int(train_labels[idx]))

        # 读取测试集
        test_images = read_idx_images(os.path.join(self.data_path, 't10k-images-idx3-ubyte'))
        test_labels = read_idx_labels(os.path.join(self.data_path, 't10k-labels-idx1-ubyte'))

        test_imgs_paths, test_labels_list = [], []
        for idx in range(len(test_labels)):
            img_pil = Image.fromarray(test_images[idx], mode='L').resize((32, 32), Image.Resampling.BICUBIC)
            img_path = os.path.join(tmp_save_dir, f"test_{idx}.jpg")
            img_pil.save(img_path, quality=70)
            test_imgs_paths.append(img_path.replace('\\', '/'))
            test_labels_list.append(int(test_labels[idx]))

        num_classes = 10

        return list(train_imgs_paths), list(train_labels_list), list(test_imgs_paths), list(
            test_labels_list), num_classes

    def read_data_cifar10(self):
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        tmp_save_dir = os.path.join(self.data_path, 'tmp_cifar10_jpg')
        os.makedirs(tmp_save_dir, exist_ok=True)

        train_imgs, train_labels = [], []
        for i in range(1, 6):
            batch = unpickle(os.path.join(self.data_path, f"data_batch_{i}"))
            data = batch[b'data']
            labels = batch[b'labels']
            for idx in range(len(labels)):
                img_array = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)  # CHW -> HWC
                img_pil = Image.fromarray(img_array)
                img_path = os.path.join(tmp_save_dir, f"train_{i}_{idx}.jpg")
                img_pil.save(img_path, quality=70)
                train_imgs.append(img_path.replace('\\', '/'))
                train_labels.append(labels[idx])

        test_imgs, test_labels = [], []
        batch = unpickle(os.path.join(self.data_path, "test_batch"))
        data = batch[b'data']
        labels = batch[b'labels']
        for idx in range(len(labels)):
            img_array = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
            img_pil = Image.fromarray(img_array)
            img_path = os.path.join(tmp_save_dir, f"test_{idx}.jpg")
            img_pil.save(img_path, quality=70)
            test_imgs.append(img_path.replace('\\', '/'))
            test_labels.append(labels[idx])

        num_classes = 10

        return train_imgs, train_labels, test_imgs, test_labels, num_classes

    def read_data_dtd(self):
        id_dict = {}
        for i, line in enumerate(open(f'{self.data_path}/dtd_ram_taglist.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        num_classes = len(id_dict)

        # 获取所有图片路径
        dtd_imgs = glob.glob(f"{self.data_path}/images/*/*.jpg")
        dtd_imgs = [img_path.replace('\\', '/') for img_path in dtd_imgs]

        # 提取标签（类别在路径的第4个索引位置）
        dtd_labels = [id_dict[img_path.split('/')[4]] for img_path in dtd_imgs]

        # 打包并打乱
        dtd_dataset = list(zip(dtd_imgs, dtd_labels))
        random.seed(0)
        random.shuffle(dtd_dataset)

        # 按 70% train / 30% test 划分
        train_size = int(0.7 * len(dtd_dataset))
        train_set = dtd_dataset[:train_size]
        test_set = dtd_dataset[train_size:]

        train_imgs, train_labels = zip(*train_set)
        test_imgs, test_labels = zip(*test_set)

        return list(train_imgs), list(train_labels), list(test_imgs), list(test_labels), num_classes

    def get_data_handler(self):
        dataset = self.dataset
        pattern = self.pattern
        datahandler = None
        if dataset == 'CIFAR100':
            train_data, train_label, test_data, test_label = self.read_data_cifar_100()
            if pattern == "train":
                if self.model_name == 'clip':
                    datahandler = CIFAR100_generate_label_clip(self.root, dataset, train_data, train_label, self.input_size)

                elif self.model_name == 'qwen':
                    datahandler = CIFAR100_handler_train_TF_saved(self.root, dataset, train_data, train_label, self.input_size, self.model, self.processor)
                elif self.model_name == 'deepseek':
                    datahandler = CIFAR100_handler_train_DP(self.root, dataset, train_data, train_label, self.input_size, self.model, self.processor)
            elif pattern == "val":
                datahandler = CIFAR100DatasetHandler(test_data, test_label, self.input_size)
        else:
            if dataset == 'tiny-imagenet-200':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_tiny_imagenet_200()
            elif dataset == 'EuroSAT':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_eruosat()
            elif dataset == 'stanford_cars':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_stanford_cars()
            elif dataset == 'caltech-101':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_caltech_101()
            elif dataset == 'food-101':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_food_101()
            elif dataset == 'fmnist':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_fmnist()
            elif dataset == 'cifar10':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_cifar10()
            elif dataset == 'dtd':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_dtd()

            if pattern == "train":
                if self.model_name == 'clip':
                    datahandler = Dataset_gengerate_label_clip(self.root, dataset, train_data, train_label, dataset_name=dataset, num_classes=num_classes,
                                                 input_size=self.input_size)
                elif self.model_name == 'qwen':
                    datahandler = DatasetHandlerTrainClip(train_data, train_label, self.input_size, self.root, dataset, self.model, self.processor, num_classes)
                elif self.model_name == 'deepseek':
                    datahandler = DatasetHandlerTrainDP(train_data, train_label, self.input_size, self.root, dataset, self.model, self.processor, num_classes)
            elif pattern == "val":
                datahandler = BaseDatasetHandler(test_data, test_label, self.input_size)

        return datahandler

    def load_datasets(self) -> Tuple[DataLoader, Dict]:
        data_path = os.path.join(self.root, self.dataset)

        tag_file = data_path + f"/{self.dataset}_ram_taglist.txt"

        with open(tag_file, "r", encoding="utf-8") as f:
            taglist_or = [line.strip() for line in f]

        taglist = taglist_or
        datahandler = self.get_data_handler()
        loader = DataLoader(dataset=datahandler, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        info = {
            "taglist": taglist
        }

        return loader, info

    def divide_labeled_or_not(self):
        data_handler = self.get_data_handler()
        indices_yt_0 = torch.nonzero(torch.eq(torch.tensor(data_handler.YT), 0)).squeeze().tolist()
        indices_yt_1 = torch.nonzero(torch.eq(torch.tensor(data_handler.YT), 1)).squeeze().tolist()
        unlabeled_dataset = Subset(data_handler, indices_yt_0)
        labeled_dataset = Subset(data_handler, indices_yt_1)

        return labeled_dataset, unlabeled_dataset


class GenerateData:
    def __init__(self, root, dataset, input_size, sampler, random_num):
        self.root = root
        self.dataset = dataset
        self.input_size = input_size
        self.sampler = sampler
        self.random_num = random_num
        self.transform = get_transform(input_size)

    def Generate_cifar_100(self):
        # random.seed(1)
        data_train, data_test, data_meta = DatasetLoader(self.root, self.dataset).load_cifar100()
        # .transpose((0,1,3,2))通过 reshape 操作，将原始数据从 (num_samples, 3072) 转换为 (num_samples, 3, 32, 32)，即每张图像被重新格式化为 3 个通道（RGB）的 32x32 的图像。
        test_data = data_test['data'].reshape((data_test['data'].shape[0], 3, 32, 32))

        gen_data = torch.ones(self.random_num, 3, 224, 224)  # 初始化一个大小为 (random_num, 3, 224, 224) 的张量 gen_data，用来存储生成的数据。
        # print([i for i in sampler])
        for i in range(self.random_num):
            # .transpose((1, 2, 0)) 将图像数据的维度从 (3, 32, 32) 转换为 (32, 32, 3)，因为 Image.fromarray 要求输入是 (height, width, channels) 的顺序。
            img = Image.fromarray(np.uint8(test_data[self.sampler[i]]).transpose((1, 2, 0)))
            # print("transform(img) = ", transform(img).size())
            # print("gen_data[i] = ", gen_data[i].size())
            gen_data[i] = self.transform(img)  # 将数组转换为 PIL 图像对象。
        # print("gen_data ", gen_data.size())
        return gen_data

    def Generate_data(self):
        if self.dataset == 'CIFAR100':
            return self.Generate_cifar_100()
        else:
            data = DatasetLoader(self.root, self.dataset)
            test_imgs = []
            if self.dataset == 'tiny-imagenet-200':
                train_imgs, train_labels, test_imgs, test_labels, _ = data.read_data_tiny_imagenet_200()
            elif self.dataset == 'EuroSAT':
                train_imgs, train_labels, test_imgs, test_labels, _ = data.read_data_eruosat()
            elif self.dataset == 'stanford_cars':
                train_imgs, train_labels, test_imgs, test_labels, _ = data.read_data_stanford_cars()
            elif self.dataset == 'caltech-101':
                train_imgs, train_labels, test_imgs, test_labels, _ = data.read_data_caltech_101()
            elif self.dataset == 'food-101':
                train_imgs, train_labels, test_imgs, test_labels, _ = data.read_data_food_101()

            gen_data = torch.ones(self.random_num, 3, 224, 224)
            for i in range(self.random_num):
                img = Image.open(test_imgs[self.sampler[i]])
                gen_data[i] = self.transform(img)
            return gen_data

