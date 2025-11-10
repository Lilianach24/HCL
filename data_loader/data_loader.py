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
from labeling.qwen_labeling_classes import CIFAR100_handler_train_TF_saved, DatasetHandlerTrainClip
from utils.utils import get_transform

class CustomDatasetWithProbs(Dataset):
    def __init__(self, root_dir, dataset_name, model='clip', pattern='train', input_size=224, conflict_only=False):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.model = model
        self.pattern = pattern
        self.input_size = input_size

        self.loader = self._create_dataset_loader()

        if pattern == 'train':
            self.images, self.labels, self.s_values, self.label_prob = self._load_train_data()

            if conflict_only:
                conflict_indices = (self.s_values == 1).nonzero(as_tuple=True)[0]
                self.images = [self.images[i] for i in conflict_indices]
                self.labels = [self.labels[i] for i in conflict_indices]
                self.s_values = self.s_values[conflict_indices]
                self.label_prob = self.label_prob[conflict_indices]
        else:
            self.images, self.labels = self._load_test_data()

        self.transform = self._get_transform()

    def _create_dataset_loader(self):
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
        """load images, labels, s_values, label_probs"""
        if self.dataset_name == "CIFAR100":
            train_data, train_label, _, _ = self.loader.read_data_cifar_100()
            images, labels = train_data, train_label
        elif self.dataset_name == "tiny-imagenet-200":
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_tiny_imagenet_200()
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
        elif self.dataset_name == 'dtd':
            train_imgs, train_labels, _, _, num_classes = self.loader.read_data_dtd()
            images, labels = train_imgs, train_labels
        else:
            raise ValueError(f"Unsupported datasets: {self.dataset_name}")

        # s_values
        s_values = self._load_s_values()

        # label_probs.npy
        label_probs_path = os.path.join(self.root_dir, self.dataset_name, 'CLIP-L14/label_prob.npy')
        label_probs = np.load(label_probs_path)

        return images, labels, s_values, label_probs

    def _load_test_data(self):
        """load test data"""
        if self.dataset_name == "CIFAR100":
            _, _, test_data, test_label = self.loader.read_data_cifar_100()
            images, labels = test_data, test_label
        elif self.dataset_name == "tiny-imagenet-200":
            _, _, test_imgs, test_labels, _ = self.loader.read_data_tiny_imagenet_200()
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
        elif self.dataset_name == 'dtd':
            _, _, test_imgs, test_labels, _ = self.loader.read_data_dtd()
            images, labels = test_imgs, test_labels
        else:
            raise ValueError(f"Unsupported datasets: {self.dataset_name}")

        return images, labels

    def _load_s_values(self):
        """load s_values from corrected_mask.txt"""
        mask_file = os.path.join(self.root_dir, self.dataset_name, "corrected_mask.txt")
        if not os.path.exists(mask_file):
            print("The file does not exist.")

        with open(mask_file, "r") as f:
            s_values = [int(line.strip()) for line in f]
        return torch.tensor(s_values, dtype=torch.int64)

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_data = self.images[idx]
        label = self.labels[idx]

        if isinstance(img_data, np.ndarray):
            image = Image.fromarray(img_data.transpose(1, 2, 0))
        else:
            image = Image.open(img_data).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.pattern == 'train':
            return image, label, self.s_values[idx], torch.from_numpy(self.label_prob[idx]).float()
        else:
            return image, label

class BaseDatasetHandler(Dataset):
    """Base Dataset Handler"""
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
    def __init__(self, X, Y, input_size):
        super().__init__(X, Y, input_size)

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        return x, y


class DatasetLoader:
    """Includes all static methods for data loading and processing."""
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

    def read_data_dtd(self):
        id_dict = {}
        for i, line in enumerate(open(f'{self.data_path}/dtd_ram_taglist.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        num_classes = len(id_dict)

        dtd_imgs = glob.glob(f"{self.data_path}/images/*/*.jpg")
        dtd_imgs = [img_path.replace('\\', '/') for img_path in dtd_imgs]
        dtd_labels = [id_dict[img_path.split('/')[4]] for img_path in dtd_imgs]
        dtd_dataset = list(zip(dtd_imgs, dtd_labels))
        random.seed(0)
        random.shuffle(dtd_dataset)

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
            elif pattern == "val":
                datahandler = CIFAR100DatasetHandler(test_data, test_label, self.input_size)
        else:
            if dataset == 'tiny-imagenet-200':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_tiny_imagenet_200()
            elif dataset == 'EuroSAT':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_eruosat()
            elif dataset == 'caltech-101':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_caltech_101()
            elif dataset == 'food-101':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_food_101()
            elif dataset == 'dtd':
                train_data, train_label, test_data, test_label, num_classes = self.read_data_dtd()

            if pattern == "train":
                if self.model_name == 'clip':
                    datahandler = Dataset_gengerate_label_clip(self.root, dataset, train_data, train_label, dataset_name=dataset, num_classes=num_classes,
                                                 input_size=self.input_size)
                elif self.model_name == 'qwen':
                    datahandler = DatasetHandlerTrainClip(train_data, train_label, self.input_size, self.root, dataset, self.model, self.processor, num_classes)
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



