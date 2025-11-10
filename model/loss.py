import torch
from clip import clip
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# 损失函数
def c_loss(output, label):
    """自定义损失函数"""
    if output.size()[0] == 0:
        return torch.tensor(0.0, device=device)
    else:
        loss = nn.MSELoss(reduction='mean').to(device)
        class_num = output.size(1)  # 动态获取类别数
        one_hot = F.one_hot(label.to(torch.int64), class_num) * 2 - 1
        sig_out = output * one_hot.to(device)
        y_label = torch.ones(sig_out.size()).to(device)
        return loss(sig_out, y_label)


# 根据公式计算自定义损失
def calculate_custom_loss(outputs, labels, s_values, label_probs, lambda_weight=1.0):
    """
    根据公式计算损失:
    E(x,Y,s=0)∑_{i=1}^{k} P(y=i|Y,s=0,x)l[f(x),i] + E(x,Y,s=1)l[f(x),Y]
    """
    batch_size, num_classes = outputs.shape

    # 初始化损失
    loss_s0 = torch.tensor(0.0, device=device)  # s=0的损失
    loss_s1 = torch.tensor(0.0, device=device)  # s=1的损失

    count_s0 = 0  # s=0的样本数
    count_s1 = 0  # s=1的样本数

    # 对每个样本分别计算损失
    for i in range(batch_size):
        if s_values[i] == 0:
            # s=0的情况: 计算∑_{i=1}^{k} P(y=i|Y,s=0,x)l[f(x),i]
            sample_output = outputs[i].unsqueeze(0)  # [1, num_classes]

            # 计算对每个可能类别的损失
            class_losses = []
            for c in range(num_classes):
                # 创建当前类别的标签
                class_label = torch.tensor([c], device=device)
                # 计算损失
                class_loss = c_loss(sample_output, class_label)
                class_losses.append(class_loss)

            # 将所有类别损失组合
            class_losses = torch.stack(class_losses)  # [num_classes]

            # 获取 p_CLIP
            p_clip = label_probs[i]  # [num_classes]
            if lambda_weight == 1.0:
                p_mixed = p_clip
            elif lambda_weight == 0.0:
                p_mixed = F.softmax(outputs[i], dim=0)
            else:
                # 获取 p_model
                p_model = F.softmax(outputs[i], dim=0)  # [num_classes]
                # 混合 soft labels
                p_mixed = lambda_weight * p_clip + (1 - lambda_weight) * p_model  # [num_classes]
            # 加权求和: ∑ P * loss
            weighted_loss = torch.sum(p_mixed * class_losses)

            loss_s0 += weighted_loss
            count_s0 += 1
        else:
            # s=1的情况: 计算l[f(x),Y]
            sample_output = outputs[i].unsqueeze(0)  # [1, num_classes]
            sample_label = labels[i].unsqueeze(0)  # [1]

            loss_s1 += c_loss(sample_output, sample_label)
            count_s1 += 1

    # 计算平均损失
    if count_s0 > 0:
        loss_s0 /= count_s0
    if count_s1 > 0:
        loss_s1 /= count_s1

    # 返回总损失
    return loss_s0 + loss_s1
