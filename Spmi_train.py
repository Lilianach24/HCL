import math
import os
import time
import random
import shutil
import logging
import warnings
import argparse
from itertools import cycle

import numpy as np
import xlwt

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F

from SPMI.spmi_dataloader import load_data
from SPMI.spmi_loss import v4Loss
from SPMI.spmi_model import Model, clip_mlinear
from utils.utils import AverageMeter, get_num_classes

parser = argparse.ArgumentParser(description='PyTorch implementation of SPMI')
# parser.add_argument('--exp_type', default='rand', type=str, choices=['rand', 'ins'], help='different exp-types')
parser.add_argument('--dataset', default='CIFAR100', type=str,
                    choices=["CIFAR100", "tiny-imagenet-200", "stanford_cars", "caltech-101", "food-101", 'EuroSAT'],
                    help='dataset name')
parser.add_argument('--exp_dir', default='./spmi_result', type=str,
                    help='experiment directory for saving checkpoints and logs')
# parser.add_argument('--pmodel_path', default='./pmodel/cifar10.pt', type=str,
#                     help='pretrained model path for generating instance dependent partial labels')
parser.add_argument('--data_dir', default='./datasets', type=str)
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=['mlp', 'lenet', 'resnet18', 'resnet34', 'resnet50', 'WRN_28_2', 'WRN_28_8', 'WRN_37_2',
#                              'linear'], help='network architecture')
parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--warm_up', default=5, type=float, help='warm up epoch')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('-lr_decay_epochs', type=str, default='5, 10, 15, 20, 25', help='where to decay lr, can be a list')
parser.add_argument('--cosine', action='store_true', default=False, help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--cuda_VISIBLE_DEVICES', default='0', type=str,
                    help='which gpu(s) can be used for distributed training')
parser.add_argument('--num_class', default=100, type=int, help='number of class')
parser.add_argument('--hierarchical', action='store_true', help='for CIFAR100-H training')
parser.add_argument('--partial_rate', default=0.6, type=float, help='ambiguity level (q)')
parser.add_argument('--labeled_num', default=10000, type=int, help='semi-supervised rate')
parser.add_argument('--kl_theta_labeled', default=3, type=float, help='kl theta for pseudo on labeled')
parser.add_argument('--kl_theta_unlabeled', default=2, type=float, help='kl theta for pseudo on unlabeled')
parser.add_argument('--ema_theta', default=0.999, type=float, help='ema for updating candidate label on labeled')

args = parser.parse_args()

timestamp = int(time.time())
t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
# record
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('result_pr_{}_{}'.format(args.partial_rate, t), cell_overwrite_ok=True)

torch.set_printoptions(precision=2, sci_mode=False)

output_path = os.path.join(args.exp_dir, args.dataset)
os.makedirs(output_path, exist_ok=True)
logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[
                        logging.FileHandler('./{}/result_{}_{}.log'.
                                            format(output_path, args.dataset, t)),
                        logging.StreamHandler()
                    ])

iterations = args.lr_decay_epochs.split(',')
args.lr_decay_epochs = list([])
for it in iterations:
    args.lr_decay_epochs.append(int(it))
args.cuda = torch.cuda.is_available()
# args.num_class = get_num_classes(args.dataset)


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 2)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if len(output) == 0:
        res = []
        for k in topk:
            res.append([0])
        return res

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calculate_y_v_k_and_y_v(labels, pred, num_classes):
    cls = (labels * pred).unsqueeze(1).repeat(1, num_classes, 1)
    for k in range(num_classes):
        cls[:, k, k] = 0
    y_v_k = cls / torch.sum(cls, dim=2).unsqueeze(2).repeat(1, 1, num_classes)
    y_v = labels * pred
    y_v = y_v / y_v.sum(dim=1).unsqueeze(1).repeat(1, num_classes)
    return y_v_k, y_v


def main():
    args.comp_num = 0

    if args.seed is not None:
        warnings.warn(
            'You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down'
            'your training considerably! You may see unexpected behavior when restarting from checkpoints.')
        print()

    model_path = 'exp_{}_lr{}_ep{}_{}'.format(args.dataset, args.lr, args.epochs, timestamp)
    # model_path = 'exp_{}{}_lr{}_wd{}_ep{}_pr{}_wu{}_seed{}_{}'. \
    #     format(args.dataset, args.arch, args.lr, args.weight_decay, args.epochs,
    #            args.partial_rate, args.warm_up, args.seed, timestamp)

    args.exp_dir = os.path.join(args.exp_dir, args.dataset, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    logging.info(args)
    print()

    main_worker(int(args.cuda_VISIBLE_DEVICES), args)


def main_worker(gpu, args):
    cudnn.benchmark = True

    args.gpu = gpu

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

    if args.gpu is not None:
        print("Use GPU: {} for training\n".format(args.gpu))

    # print("=> creating model '{}'\n".format(args.arch))

    # combinations = [
    #     ([0]),
    #     ([1]),
    #     ([2]),
    #     ([0, 1]),
    #     ([1, 2]),
    #     ([0, 1, 2])
    #     # (8, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    # ]
    #
    # # # 添加三组随机组合
    # for _ in range(3):
    #     sample_length = random.randint(1, 3)  # 随机选择 sample_classes（1 或 2）
    #     conceal_label = random.sample(range(10), sample_length)  # 随机选择两个 conceal_label，范围是 0 到 9
    #     combinations.append(conceal_label)

    # Prepare Excel sheet for saving results
    sheet.write(0, 0, 'Epoch')
    sheet.write(0, 1, 'Train_loss_cls')
    sheet.write(0, 2, 'Train_acc_cls')
    sheet.write(0, 3, 'Test_acc_cls')
    sheet.write(0, 4, 'Best_acc_cls')
    sheet.write(0, 5, 'lr')
    sheet.write(0, 6, 'error_partial')
    sheet.write(0, 7, 'partial_cand_num')
    sheet.write(0, 8, 'error_unlabeled')
    sheet.write(0, 9, 'unlabeled_cand_num')

    # load dataset
    train_loader, train_partialY_matrix, unlabeled_training_dataloader, test_loader = load_data(args)

    logging.info('Average candidate num: {}'.format(train_partialY_matrix.sum(1).mean()))

    # calculate init uniform confidence
    print('\nCalculating uniform confidence...')
    tempY = train_partialY_matrix.sum(dim=1).unsqueeze(1).repeat(1, train_partialY_matrix.shape[1])
    uniform_confidence = train_partialY_matrix.float() / tempY
    uniform_confidence = uniform_confidence.cuda()

    loss_v4_func = v4Loss(predicted_score_cls=uniform_confidence)

    # Start Training
    # 添加一个外部循环，跑三次训练
    # num_trials = 3
    # best_accuracies = []
    # for trial in range(num_trials):
    #     print(f'\n################ Start Training Trial {trial + 1}... ################')

    # 初始化模型
    model = Model(args, clip_mlinear)
    model = model.cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    best_acc = 0
    args.start_epoch = 0

    args.train_iteration = math.ceil(
        max(len(train_loader.dataset), len(unlabeled_training_dataloader.dataset)) / args.batch_size)

    for epoch in range(args.start_epoch, args.epochs):

        is_best = False

        adjust_learning_rate(args, optimizer, epoch)
        # scheduler_known.step()

        acc_train_cls, loss_v4_log, error_partial, partial_cand_num, error_unlabeled, unlabeled_cand_num = \
            train(train_loader, unlabeled_training_dataloader, model, loss_v4_func, optimizer, epoch, args)

        acc_test = test(model, test_loader, args)

        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True

        sheet.write(epoch + 1, 0, epoch + 1)
        sheet.write(epoch + 1, 1, loss_v4_log.avg)
        sheet.write(epoch + 1, 2, acc_train_cls.avg.item())
        sheet.write(epoch + 1, 3, acc_test.item())
        sheet.write(epoch + 1, 4, best_acc.item())
        sheet.write(epoch + 1, 5, optimizer.param_groups[0]['lr'])
        sheet.write(epoch + 1, 6, error_partial)
        sheet.write(epoch + 1, 7, partial_cand_num)
        sheet.write(epoch + 1, 8, error_unlabeled)
        sheet.write(epoch + 1, 9, unlabeled_cand_num)
        # 生成文件路径
        savepath = '{}/result_{}_pr_{}_seed_{}.xls'.format(
            output_path, args.dataset, args.partial_rate, args.seed)
        # 保存结果到该文件
        book.save(savepath)

        if epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(output_path),
                best_file_name='{}/checkpoint_best.pth.tar'.format(output_path))

        logging.info(
            f'Epoch {epoch + 1}: Train_Acc {acc_train_cls.avg}, Test_Acc {acc_test}, Best_Acc {best_acc}. (lr:{optimizer.param_groups[0]["lr"]})')

        # # 保存每次试验的最佳验证精度
        # best_accuracies.append(best_acc.cpu().numpy())
        # # 计算三次验证精度的平均值和标准差
        # mean_acc = np.mean(best_accuracies)
        # std_acc = np.std(best_accuracies)

        # logging.info(f"Validation Accuracy : {mean_acc:.2f} ± {std_acc:.2f}\n")
        # logging.info(f"Mean Validation Accuracy : {mean_acc:.2f}\n"
        #              f"Std Accuracy: {std_acc:.2f}\n"
        #              f"Best Accuracy for each trial: {best_accuracies}\n")


def train(labeled_loader, unlabeled_loader, model, sup_loss_func, optimizer, epoch, args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if epoch < args.warm_up:
        acc_cls = AverageMeter('Acc@Cls', ':2.2f')
        loss_sup_log = AverageMeter('Loss@sup', ':2.2f')
        model.train()

        # WARM UP -- supervised learning
        labeled_loader.dataset.pre = False
        # labeled_loader_iter = iter(labeled_loader)
        labeled_loader_iter = cycle(labeled_loader)
        for i in range(args.train_iteration):  # align with SSL methods
            # try:
            #     img_w, img2, labels, true_labels, batch_idx = next(labeled_loader_iter)
            # except:
            #     labeled_loader_iter = iter(labeled_loader)
            #     img_w, img2, labels, true_labels, batch_idx = next(labeled_loader_iter)
            img_w, img2, labels, true_labels, batch_idx = next(labeled_loader_iter)

            X_w, Y, index = img_w.cuda(), labels.cuda(), batch_idx.cuda()
            Y_true = true_labels.long().cuda()

            outputs = model(img_q=X_w)
            if isinstance(outputs, tuple):
                logits = outputs[1]  # 获取分类预测结果
            else:
                logits = outputs

            pred = torch.softmax(logits, dim=1)

            # update predicted score (1st epoch use uniform)
            if epoch > 0:
                sup_loss_func.update_weight_byclsout1(cls_predicted_score=pred, batch_index=index, batch_partial_Y=Y,
                                                      args=args)

            sup_loss = sup_loss_func(pred, index)
            loss = sup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record
            loss_sup_log.update(loss.item())
            acc = accuracy(pred, Y_true)[0]
            acc_cls.update(acc[0])

            if (i % 100 == 0) or (i + 1 == args.train_iteration):
                logging.info('Epoch:[{0}][{1}/{2}]\t'
                             'A_cls {Acc_cls.val:.4f} ({Acc_cls.avg:.4f})\t'
                             'L_sup {Loss_sup.val:.4f} ({Loss_sup.avg:.4f})\t'.format(
                    epoch + 1, i + 1, args.train_iteration, Acc_cls=acc_cls, Loss_sup=loss_sup_log))

        return acc_cls, loss_sup_log, 0, 0, 0, 0

    else:
        acc_cls = AverageMeter('Acc@Cls', ':2.2f')
        acc_comp = AverageMeter('Acc@comp', ':2.2f')
        acc_all = AverageMeter('Acc@all', ':2.2f')
        loss_sup_log = AverageMeter('Loss@sup', ':2.2f')
        loss_un_log = AverageMeter('Loss@un', ':2.2f')
        loss_all_log = AverageMeter('Loss@all', ':2.2f')

        # generate pseudo label -- candidate & complementary
        if epoch >= args.warm_up:
            logging.info('\n=====> Generate pseudo complementary label\n')
            err_sup = torch.tensor(0., device=args.gpu)
            cand_num = AverageMeter('cand_num')
            err_un = torch.tensor(0., device=args.gpu)
            comp_num = AverageMeter('comp_num')
            posterior = AverageMeter('posterior')
            last_posterior = unlabeled_loader.dataset.posterior
            consistency_criterion = nn.KLDivLoss(reduction='none').cuda()

            # 计算类别平均后验概率（关键修改）
            if last_posterior.dim() == 2:
                # last_posterior形状为[total_samples, num_classes]
                class_posterior = torch.mean(last_posterior, dim=0, keepdim=True).to(device)  # [1, num_classes]
            else:
                # last_posterior形状为[num_classes]或[1, num_classes]
                class_posterior = last_posterior.to(device)

            model.eval()
            with torch.no_grad():
                # labeled
                labeled_loader.dataset.pre = True
                for i, (images_1, labels, ori_labels, true_labels, index) in enumerate(labeled_loader):
                    X, Y, Y_ori, Y_true, index = images_1.cuda(), labels.cuda(), ori_labels.cuda(), true_labels.long().cuda(), index.cuda()
                    gt = torch.nn.functional.one_hot(Y_true, num_classes=args.num_class).squeeze().cuda()
                    Y = (Y > 0.1).to(torch.int)  # transform hard label
                    Y_ori = Y_ori.to(device)
                    cls_sup = model(img_q=X)
                    pred = torch.softmax(cls_sup, dim=1)
                    # print(f"pred.shape: {pred.shape}")
                    # print(f"last_posterior.shape: {last_posterior.shape}")
                    # print(f"Y_ori.shape: {Y_ori.shape}")
                    # remove label -- KL(P(y_v/k|x,w)||P(y_v|x,w))
                    y_v_k, y_v = calculate_y_v_k_and_y_v(Y, pred, args.num_class)
                    kl = consistency_criterion(F.log_softmax(y_v_k, dim=1),
                                               y_v.unsqueeze(1).repeat(1, args.num_class, 1)).sum(2)
                    kl[Y == 0] = torch.inf
                    kl_min = torch.min(kl, dim=1)
                    kl[Y == 0] = -torch.inf
                    kl_max = torch.max(kl, dim=1)
                    val_idx = torch.logical_and(torch.sum(Y, dim=1) != 1, kl_max[0] > args.kl_theta_labeled)
                    pur_idx = 1 - torch.nn.functional.one_hot(kl_min[1], num_classes=args.num_class)
                    pur_idx[torch.logical_not(val_idx)] = 1
                    cand_label = Y - (1 - pur_idx)

                    # add label -- p > class posterior τ
                    last_posterior = last_posterior.to(pred.device)
                    cand_label = torch.logical_or(
                        cand_label,
                        torch.logical_and(pred > last_posterior, Y_ori)
                    ).to(torch.float)
                    sup_loss_func.update_weight_byclsout1(cls_predicted_score=pred, batch_index=index,
                                                          batch_partial_Y=cand_label, args=args)
                    labeled_loader.dataset.set_pseudo_cand_labels(cand_label, index, args.ema_theta)

                    err_sup += torch.sum((1 - cand_label) * gt)
                    cand_num.update(torch.mean(torch.sum(cand_label, dim=1)), len(cand_label))

                # unlabeled
                unlabeled_loader.dataset.pre = True
                for i, (img_w, comp_labels_u, true_labels, batch_idx) in enumerate(unlabeled_loader):
                    img_w, comp_labels_u, true_labels, batch_idx = img_w.cuda(), comp_labels_u.cuda(), true_labels.cuda(), batch_idx.cuda()
                    labels_u = (1 - comp_labels_u).cuda()
                    gt = torch.nn.functional.one_hot(true_labels, num_classes=args.num_class).squeeze().cuda()

                    outputs = model(img_q=img_w)
                    pred = torch.softmax(outputs, dim=1)

                    # recode posterior use last model
                    posterior.update(torch.mean(pred, dim=0), len(pred))

                    # init pseudo complementary label (Notice: use complementary label to store in unlabeled)
                    if epoch == args.warm_up:
                        pseudo_complementary_label = (
                                -torch.log(pred) > torch.mean(-torch.log(pred), dim=1, keepdim=True)).to(
                            torch.float)  # CCE -logx
                        unlabeled_loader.dataset.set_pseudo_complementary_labels(pseudo_complementary_label, batch_idx,
                                                                                 init=True)
                        comp_labels_u = pseudo_complementary_label
                        labels_u = (1 - comp_labels_u)

                    if args.dataset == 'CIFAR100':
                        comp_labels_u = (comp_labels_u > 0.1).to(torch.float)  # transform hard label
                        labels_u = (1 - comp_labels_u)

                    comp_labels_u = comp_labels_u.to(device)
                    comp_labels_u = torch.logical_and(
                        comp_labels_u,
                        (pred <= class_posterior)
                    ).to(torch.float)

                    # remove label -- KL(P(y_v/k|x,w)||P(y_v|x,w))
                    y_v_k, y_v = calculate_y_v_k_and_y_v(labels_u, pred, args.num_class)
                    kl = consistency_criterion(F.log_softmax(y_v_k, dim=1),
                                               y_v.unsqueeze(1).repeat(1, args.num_class, 1)).sum(2)
                    kl[labels_u == 0] = torch.inf
                    kl_min = torch.min(kl, dim=1)
                    kl[labels_u == 0] = -torch.inf
                    kl_max = torch.max(kl, dim=1)
                    val_idx = torch.logical_and(torch.sum(labels_u, dim=1) != 1, kl_max[0] > args.kl_theta_unlabeled)
                    comp_labels_u[val_idx] = torch.logical_or(
                        comp_labels_u[val_idx],
                        torch.nn.functional.one_hot(kl_min[1][val_idx],
                                                    num_classes=args.num_class).squeeze()).to(torch.float)

                    # add label
                    last_posterior = last_posterior.to(pred.device)
                    comp_labels_u = torch.logical_and(
                        comp_labels_u,
                        (pred <= last_posterior.to(pred.device))
                    ).to(torch.float)

                    unlabeled_loader.dataset.set_pseudo_complementary_labels(comp_labels_u, batch_idx)

                    err_un += torch.sum(comp_labels_u * gt)
                    comp_num.update(torch.mean(torch.sum(comp_labels_u, dim=1)), len(comp_labels_u))

                    if (i % 100 == 0) or (i + 1 == len(unlabeled_loader)):
                        logging.info('Epoch:[{0}][{1}/{2}]\t'.format(
                            epoch + 1, i + 1, len(unlabeled_loader)))

                # unlabeled_loader.dataset.set_posterior(posterior.avg)
                unlabeled_loader.dataset.set_posterior(posterior.avg.unsqueeze(0).repeat(len(unlabeled_loader.dataset), 1).cpu())
                logging.info('cand num error/all: {}/{}'.format(int(err_sup.item()), len(labeled_loader.dataset)))
                logging.info('avg pseudo cand num: {}'.format(cand_num.avg))
                logging.info(
                    'pseudo comp num error/all: {}/{}'.format(int(err_un.item()), len(unlabeled_loader.dataset)))
                logging.info('avg pseudo comp num: {}'.format(comp_num.avg))
                logging.info('pseudo comp num: {}'.format(
                    torch.sum(unlabeled_loader.dataset.pseudo_complementary_labels, dim=0)))
                logging.info('posterior: {}'.format(posterior.avg))
                error_partial = int(err_sup.item())
                partial_cand_num = cand_num.avg.item()
                error_unlabeled = int(err_un.item())
                unlabeled_cand_num = args.num_class - comp_num.avg.item()

        # union training
        logging.info('\n=====> Training\n')
        model.train()

        # get labeled and unlabeled data at the same time
        labeled_loader.dataset.pre = False
        unlabeled_loader.dataset.pre = False
        labeled_loader_iter = iter(labeled_loader)
        unlabeled_loader_iter = iter(unlabeled_loader)
        for i in range(args.train_iteration):
            try:
                img_w_l, labels_l, ori_labels_l, true_labels_l, batch_idx_l = next(labeled_loader_iter)
            except:
                labeled_loader_iter = iter(labeled_loader)
                img_w_l, labels_l, ori_labels_l, true_labels_l, batch_idx_l = next(labeled_loader_iter)
            try:
                img_s_u, comp_labels_u, true_labels_u, batch_idx_u = next(unlabeled_loader_iter)
            except:
                unlabeled_loader_iter = iter(unlabeled_loader)
                img_s_u, comp_labels_u, true_labels_u, batch_idx_u = next(unlabeled_loader_iter)

            # labeled
            img_w_l, labels_l, true_labels_l, batch_idx_l = \
                img_w_l.cuda(), labels_l.cuda(), true_labels_l.long().cuda(), batch_idx_l.cuda()

            # unlabeled
            # only use valid pseudo label (no all 0 and 1)
            img_s_u = img_s_u.to(device)
            comp_labels_u = comp_labels_u.to(device)
            true_labels_u = true_labels_u.to(device)
            batch_idx_u = batch_idx_u.to(device)

            comp_label_num = torch.sum(comp_labels_u, dim=1)
            valid_idx = (comp_label_num != 0) * (comp_label_num != args.num_class)
            img_s_u, comp_labels_u, true_labels_u, batch_idx_u = \
                img_s_u[valid_idx], comp_labels_u[valid_idx], true_labels_u[valid_idx], batch_idx_u[valid_idx]
            labels_u = (1 - comp_labels_u)

            # model
            input_s = torch.cat((img_w_l, img_s_u), dim=0)
            outputs = model(img_q=input_s)
            pred = torch.softmax(outputs, dim=1)

            k_l = len(labels_l)

            cls_out_w = pred[:k_l]
            un_out_s = pred[k_l:]

            # supervised loss
            sup_loss_func.update_weight_byclsout1(cls_predicted_score=cls_out_w, batch_index=batch_idx_l,
                                                  batch_partial_Y=labels_l, args=args)
            sup_loss = sup_loss_func(cls_out_w, batch_idx_l)

            loss_sup_log.update(sup_loss.item())
            acc = accuracy(cls_out_w, true_labels_l)[0]
            acc_cls.update(acc[0])

            # partial loss for unlabeled
            un_pseudo_s = labels_u.clone() * un_out_s
            un_pseudo_s = (un_pseudo_s / un_pseudo_s.sum(dim=1).repeat(args.num_class, 1).transpose(0, 1)).detach()
            unsup_loss = -torch.mean(torch.sum(un_pseudo_s * torch.log(un_out_s), dim=1))

            loss_un_log.update(unsup_loss.item())
            acc_c = accuracy(un_pseudo_s, true_labels_u)[0]
            acc_comp.update(acc_c[0])
            acc_all.update((acc[0] + acc_c[0]) / 2)

            # total loss
            loss = sup_loss + unsup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all_log.update(loss.item())
            if (i % 50 == 0) or (i + 1 == args.train_iteration):
                logging.info('Epoch:[{0}][{1}/{2}]\t'
                             'Acc_cls {Acc_cls.val:.4f} ({Acc_cls.avg:.4f})\t'
                             'Acc_un {Acc_un.val:.4f} ({Acc_un.avg:.4f})\t'
                             'Acc_all {Acc_all.val:.4f} ({Acc_all.avg:.4f})\n'
                             'L_sup {Loss_sup.val:.4f} ({Loss_sup.avg:.4f})\t'
                             'L_un {L_un.val:.4f} ({L_un.avg:.4f})\t'
                             'L_all {Loss_all.val:.4f} ({Loss_all.avg:.4f})\t'.format(
                    epoch + 1, i + 1, args.train_iteration,
                    Acc_cls=acc_cls, Acc_un=acc_comp, Acc_all=acc_all,
                    Loss_sup=loss_sup_log, L_un=loss_un_log, Loss_all=loss_all_log))

    return acc_all, loss_all_log, error_partial, partial_cand_num, error_unlabeled, unlabeled_cand_num


def test(model, test_loader, args):
    with torch.no_grad():
        logging.info('\n=====> Evaluation...\n')
        model.eval()

        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")

        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            # print(f"Input shape before model: {images.shape}")

            outputs = model(img_q=images)
            if isinstance(outputs, tuple):
                logits = outputs[1]  # 提取第一个元素作为 logits
            else:
                logits = outputs  # 直接使用张量

            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

            if (batch_idx % 10 == 0) or (batch_idx + 1 == len(test_loader)):
                logging.info(
                    'Test:[{0}/{1}]\t'
                    'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                    'Prec@5 {top5.val:.2f} ({top5.avg:.2f})\t' \
                        .format(batch_idx + 1, len(test_loader), top1=top1_acc, top5=top5_acc)
                )

        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)

        logging.info('\n****** Top1 Accuracy is %.2f%%, Top5 Accuracy is %.2f%% ******\n' \
                     % (acc_tensors[0], acc_tensors[1]))

    return acc_tensors[0]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


if __name__ == '__main__':
    main()