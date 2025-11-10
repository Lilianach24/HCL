import math

import torch
import torch.nn.functional as F

from utils.utils import AverageMeter


def generate_instancedependent_candidate_labels(model, train_X, train_Y,RATE=0.4):
    with torch.no_grad():
        k = int(torch.max(train_Y) - torch.min(train_Y) + 1)
        n = train_Y.shape[0]
        model = model.cuda()
        train_Y = torch.nn.functional.one_hot(train_Y, num_classes=k) # 真实标签始终被保留，且对应位置为 1。
        avg_C = 0
        partialY_list = []
        rate, batch_size = RATE, 2000
        step = math.ceil(n / batch_size)

        for i in range(0, step):
            b_end = min((i + 1) * batch_size, n)

            train_X_part = train_X[i * batch_size: b_end].cuda()

            outputs = model(train_X_part)

            train_p_Y = train_Y[i * batch_size: b_end].clone().detach() # train_p_Y 初始化为 train_Y 的克隆：

            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(train_p_Y == 1)] = 0 # 将真实标签的概率置为 0，确保不会重复选择真实标签作为候选。
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0

            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()

            train_p_Y[torch.where(z == 1)] = 1.0  # 将这些标签作为候选标签。
            partialY_list.append(train_p_Y)

        partialY = torch.cat(partialY_list, dim=0).float()

        assert partialY.shape[0] == train_X.shape[0]

    avg_C = torch.sum(partialY) / partialY.size(0)

    return partialY


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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
        return res[0]


def test(args, epoch, test_loader, model):
    with torch.no_grad():
        model.eval()
        top1_acc = AverageMeter("Top1")

        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images, images, eval_only=True)
            acc1 = accuracy(outputs, labels)
            top1_acc.update(acc1[0])

    return top1_acc.avg
