import os

import torch
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from typing import Dict, Tuple
import numpy as np
import random


def convert_to_rgb(image):
    return image.convert("RGB")

def get_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_taglist(
        dataset: str
) -> Tuple[Dict]:
    dataset_root = "./datasets/" + dataset

    tag_file = dataset_root + f"/{dataset}_ram_taglist.txt"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist_or = [line.strip() for line in f]
    taglist = taglist_or

    info = {"taglist": taglist}
    return info

# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_num_classes(dataset_name, data_dir="../datasets"):
    """
    Get the number of classes based on the dataset name
    """
    data_path = os.path.join(data_dir, dataset_name)
    taglist_files = {
        "CIFAR100": f"{data_path}/{dataset_name}_ram_taglist.txt",
        "tiny-imagenet-200": f"{data_path}/{dataset_name}_ram_taglist.txt",
        "caltech-101": f"{data_path}/{dataset_name}_ram_taglist.txt",
        "food-101": f"{data_path}/{dataset_name}_ram_taglist.txt",
        "EuroSAT": f"{data_path}/{dataset_name}_ram_taglist.txt",
        "dtd": f"{data_path}/{dataset_name}_ram_taglist.txt",
    }

    if dataset_name not in taglist_files:
        raise ValueError(f"Unsupported datasets: {dataset_name}")

    taglist_file = taglist_files[dataset_name]
    try:
        with open(taglist_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return len(classes)
    except FileNotFoundError:
        print(f"Error: category name file not found {taglist_file}")



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# if __name__ == "__main__":
#     num = get_num_classes(dataset_name="CIFAR100", data_dir="../datasets")
#     print(num)

