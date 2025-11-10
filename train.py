import argparse
import csv
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from data_loader.data_loader import CustomDatasetWithProbs
from model.model import CLIPLinearModel
from utils.logger import set_log
from trainer.trainer import Trainer
from utils.utils import set_seed, get_num_classes


def main(opt):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = opt.root_dir
    dataset_name = opt.dataset_name
    num_classes = get_num_classes(dataset_name, root_dir)
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    learning_rate = float(opt.learning_rate)
    input_size = 224
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

    train_dataset = CustomDatasetWithProbs(
        root_dir=root_dir,
        dataset_name=dataset_name,
        pattern='train',
        input_size=input_size,
        conflict_only = False,
    )

    test_dataset = CustomDatasetWithProbs(
        root_dir=root_dir,
        dataset_name=dataset_name,
        pattern='val',
        input_size=input_size,
        conflict_only = False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CLIPLinearModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.linear.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("Start training the model...")
    trainer = Trainer(opt, model, train_loader, test_loader, optimizer, scheduler, device, logging, logs)
    trainer.train()


if __name__ == "__main__":
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, dest='root_dir', default='./datasets', help='dataset root dir')
    parser.add_argument('--dataset_name', type=str, default='CIFAR100',
                        dest='dataset_name', required=False, help='dataset name')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=64, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=30, dest='num_epochs', help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, dest='learning_rate', help='learning rate')
    parser.add_argument('--output_path', type=str, default='./result', help='output path')
    parser.add_argument('--lambda_weight', type=float, default=1.0, dest='lambda_weight', help='lambda weight')
    opt = parser.parse_args()

    main(opt)
