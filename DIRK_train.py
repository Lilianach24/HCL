import os
import argparse
import time
import numpy as np
import torch
import random
from torch.backends import cudnn

from DIRK.dirk_dataloader import load_data
from DIRK.dirk_loss import WeightedConLoss, CE_loss
from DIRK.dirk_model import conTea, conStu, SupCon_mLinear
from DIRK.dirk_utils import test, AverageMeter
import logging
import csv
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import get_num_classes

parser = argparse.ArgumentParser(description='DIRK')
# global set
parser.add_argument('--data_dir', type=str, default='./datasets', help='path to datasets')
parser.add_argument('--dataset', default='CIFAR100',
                    choices=['CIFAR100', 'caltech-101', 'food-101', 'stanford_cars', 'tiny-imagenet-200', 'EuroSAT', 'cifar10', 'fmnist'], type=str)
parser.add_argument('--arch', default='linear', type=str)
parser.add_argument('--method', default='DIRK', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--seed', default=3407, type=int)
# optimization
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('-lr_decay_epochs', type=str, default='5, 10, 15, 20, 25', help='where to decay lr, can be a list')
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--m', type=float, default=0.999)
# PLL setting
parser.add_argument('--rate', default=0.6, type=float)
parser.add_argument('--weight', type=float, default=0.5)
# REF setting
parser.add_argument('--queue', type=int, default=4096)
parser.add_argument('--dist_temp', type=float, default=0.7)
parser.add_argument('--feat_temp', type=float, default=0.5)
parser.add_argument('--prot_start', type=int, default=1)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parser.parse_args()
    if args.queue % args.batch_size != 0:
        args.queue = ((args.queue // args.batch_size) + 1) * args.batch_size
        print(f"Queue size adjusted to {args.queue} to be divisible by batch size {args.batch_size}")
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_path = os.path.join("DIRK_results", args.dataset)
    os.makedirs(output_path, exist_ok=True)
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[
                            logging.FileHandler('./{}/result_{}_{}.log'.
                                                format(output_path, args.dataset, t)),
                            logging.StreamHandler()
                        ])
    torch.set_printoptions(linewidth=2000)
    logging.info(args.__dict__)
    main_worker(args, output_path)


def main_worker(args, output_path):
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    logging.info('=> Start Training')

    log_file_path = f"{output_path}/{args.dataset}_{t}.csv"

    # load dataloader
    logging.info("=> creating loader '{}'".format(args.dataset))
    # get_loader
    # train_loader, test_loader, num_class = load_dataset(args, conceal_label)
    train_loader, _, test_loader = load_data(args)
    # print(f"Train loader batch size: {train_loader.batch_size}")
    args.num_class = get_num_classes(dataset_name=args.dataset, data_dir=args.data_dir)
    # print('len(train_loader.dataset) value:', len(train_loader.dataset)) 
    logging.info('=> Average number of partial labels: {}'.format(
        train_loader.dataset.given_partial_label_matrix.sum() / len(train_loader.dataset)))

    # set contrastive loss function 
    loss_cont_fn = WeightedConLoss(temperature=args.feat_temp, dist_temprature=args.dist_temp)

    best_acc1 = 0
    best_val_accuracies = []

    set_seed(42)
    # logging.info(f"=> Starting Test Run {test_run + 1}")
    best_acc1 = 0
    # Teacher = CLIPTeacherModel(num_classes=get_num_classes(args.dataset, args.data_dir)).to(device)
    # Student = CLIPStudentModel(num_classes=get_num_classes(args.dataset, args.data_dir)).to(device)
    Teacher = conTea(args, SupCon_mLinear).cuda()
    Student = conStu(args, SupCon_mLinear).cuda()
    # print(f"Teacher queue size: {Teacher.moco_queue}")
    S_optimizer = torch.optim.AdamW(Student.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = torch.optim.lr_scheduler.StepLR(S_optimizer, step_size=5, gamma=0.1)

    for epoch in range(args.epochs):
        adjust_learning_rate(args, S_optimizer, epoch) 
        start_upd_prot = epoch >= args.prot_start
        teach_loss, cont_loss = train(train_loader, Teacher, Student, S_optimizer, loss_cont_fn, epoch, args,
                                      start_upd_prot)
        logging.info("[Training-Epoch {}]: TeachLoss:{:.4f}\t ContrastiveLoss:{:.4f}".format(epoch, teach_loss,
                                                                                             cont_loss))
        # test
        val_acc = test(args, epoch, test_loader, Student)
        best_acc1 = max(best_acc1, val_acc)

        # logging.info(f"Epoch {epoch + 1}: TeachLoss: {teach_loss:.4f}, ContrastiveLoss: {cont_loss:.4f}, "
        #              f"ValidationAccuracy: {val_acc:.4f}, Best_Valid_Acc: {best_acc1:.4f}\n")
        logging.info(
            "[Testing-Epoch {}]:ValidationAccuracy: {:.4f} --Best_valid_acc: {:.4f}".format(epoch, val_acc,
                                                                                            best_acc1))
        # save
        log_metrics_to_file(log_file_path, epoch, teach_loss, cont_loss, val_acc, best_acc1)
        # scheduler.step()

        # best_val_accuracies.append(best_acc1.cpu().numpy())


def log_metrics_to_file(file_path, epoch, teach_loss, cont_loss, val_acc, best_acc1):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Epoch', 'TeachLoss', 'ContrastiveLoss', 'ValidationAccuracy', 'Best_Valid_Acc'])
        writer.writerow([epoch, teach_loss, cont_loss, val_acc, best_acc1])
    # plot_metrics(file_path)


def train(train_loader, Teacher, Student, S_optimizer, loss_cont_fn, epoch, args, start_upd_prot=False):
    teach_losses = AverageMeter('TeachLoss', ':.2f')
    con_losses = AverageMeter('ContrastiveLoss', ':.2f')

    # switch to train mode
    Student.train()
    Teacher.train()

    for i, (img_w, img_s, img_distill, partY, target, index) in enumerate(train_loader):
        # print("Shape of partY:", partY.shape)
        img_w, img_s, img_distill, partY, target, index = img_w.cuda(), img_s.cuda(), img_distill.cuda(), partY.cuda(), target.cuda(), index.cuda()
        # obtain pools
        features, partYs, dists, rec_conf_t = Teacher(img_w, img_s, img_distill, partY, target=target.unsqueeze(1))
        # obtain Student's output and feature
        output_s, feat_s = Student(img_s, img_distill)
        # print(f"Epoch {epoch}, Batch {i}:")
        # print(f"feat_s shape: {feat_s.shape}")      # [batch_size, 128]
        # print(f"features shape: {features.shape}")  # [batch_size + queue_size, 128]
        # print(f"output_s shape: {output_s.shape}")  # [batch_size, num_classes]
        # print(f"rec_conf_t shape: {rec_conf_t.shape}")  # [batch_size, num_classes]
        # print(f"dists shape: {dists.shape}")        # [batch_size + queue_size, 128]
        # bind features and partial distribution
        features_cont = torch.cat((feat_s, features), dim=0)
        partY_cont = torch.cat((partY, partYs), dim=0)
        dist_cont = torch.cat((rec_conf_t, dists), dim=0)

        batch_size = output_s.shape[0]
        mask_partial = torch.matmul(partY_cont[:batch_size], partY_cont.T)  
        mask_partial[mask_partial != 0] = 1
        _, pseudo_target = torch.max(dist_cont, dim=1)  
        pseudo_target = pseudo_target.contiguous().view(-1, 1)
        mask_pseudo_target = torch.eq(pseudo_target[:batch_size], pseudo_target.T).float()  

        if start_upd_prot:
            mask = mask_partial * mask_pseudo_target
        else:
            mask = None

        # contrastive loss
        if args.weight != 0:
            # weight = args.weight * min(1.0, epoch / 20) 
            loss_cont = loss_cont_fn(features=features_cont, dist=dist_cont, partY=partY_cont, mask=mask, epoch=epoch,
                                     args=args, batch_size=partY.shape[0])
        else:
            loss_cont = torch.tensor(0.0).cuda()
        # teaching loss
        loss_teach = CE_loss(output_s, rec_conf_t)
        # total loss
        loss = loss_teach + args.weight * loss_cont

        teach_losses.update(loss_teach.item(), partY.size(0))
        con_losses.update(loss_cont.item(), partY.size(0))

        # compute gradient and do SGD step
        S_optimizer.zero_grad()
        loss.backward()
        S_optimizer.step()

        momentum_model(Teacher, Student, args.m)

    return teach_losses.avg, con_losses.avg


def momentum_model(model_tea, model_stu, momentum=0.5):
    for param_tea, param_stu in zip(model_tea.parameters(), model_stu.parameters()):
        param_tea.data = param_tea.data * momentum + param_stu.data * (1 - momentum)


def adjust_learning_rate(args, optimizer, epoch):
    import math
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    logging.info('LR: {}'.format(lr))


if __name__ == '__main__':
    main()
