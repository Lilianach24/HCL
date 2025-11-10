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
    train_dataset = CustomDatasetWithProbs(
        root_dir=root_dir,
        dataset_name=dataset_name,
        pattern='train',
        input_size=input_size,
        conflict_only = False,
    )

    # 创建测试集（使用 BaseDatasetHandler 或类似类）
    test_dataset = CustomDatasetWithProbs(
        root_dir=root_dir,
        dataset_name=dataset_name,
        pattern='val',
        input_size=input_size,
        conflict_only = False,
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 创建模型
    model = CLIPLinearModel(num_classes=num_classes).to(device)
    # 加载预训练模型（如果提供）
    # if len(opt.pretrained_model) > 0:
    #     print(f"加载预训练模型: {opt.pretrained_model}")
    #     # 处理不同设备上的模型加载
    #     state_dict = torch.load(
    #         opt.pretrained_model,
    #         map_location=lambda storage, loc: storage  # 自动映射到当前设备
    #     )
    #     # 适配模型结构（如果有必要）
    #     model_state_dict = model.state_dict()
    #     # 仅加载匹配的参数
    #     loaded_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    #     model.load_state_dict(loaded_state_dict, strict=False)
    #     print(f"成功加载 {len(loaded_state_dict)}/{len(model_state_dict)} 个参数")
    # 定义优化器
    # optimizer = torch.optim.Adam(model.linear.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.linear.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 训练模型
    print("开始训练模型...")
    trainer = Trainer(opt, model, train_loader, test_loader, optimizer, scheduler, device, logging, logs)
    trainer.train()


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
    parser.add_argument('--output_path', type=str, default='./result', help='output path')
    parser.add_argument('--lambda_weight', type=float, default=1.0, dest='lambda_weight', help='lambda weight')
    opt = parser.parse_args()
    main(opt)