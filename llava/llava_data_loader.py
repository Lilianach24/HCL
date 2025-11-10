import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pickle
import random
import glob
import scipy.io
from typing import Tuple, Dict
import numpy as np

from llava.llava_labeling_classes import CIFAR100HandlerTrainLLaVA, DatasetHandlerTrainLLaVA
from utils.utils import get_transform


class BaseDatasetHandler(Dataset):
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

class LLaVADatasetLoader:
    def __init__(self, processor, model, root='datasets', dataset='CIFAR100', model_name='llava', pattern='train', input_size=224, batch_size=64, num_workers=0, ):
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
        EuroSAT_imgs = glob.glob(f"{self.data_path}/2750/*/*.jpg")
        EuroSAT_imgs = [img_path.replace('\\', '/') for img_path in EuroSAT_imgs]
        EuroSAT_labels = [id_dict[img_path.split('/')[4]] for img_path in EuroSAT_imgs]
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

    def get_data_handler(self):
        dataset = self.dataset
        pattern = self.pattern
        datahandler = None
        if dataset == 'CIFAR100':
            train_data, train_label, test_data, test_label = self.read_data_cifar_100()
            if pattern == "train":
                datahandler = CIFAR100HandlerTrainLLaVA(self.root, dataset, train_data, train_label, self.input_size, self.model, self.processor)
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

            if pattern == "train":
                datahandler = DatasetHandlerTrainLLaVA(train_data, train_label, self.input_size, self.root, dataset, self.model, self.processor)
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
