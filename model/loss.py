import torch
from clip import clip
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# loss function
def c_loss(output, label):
    """Custom loss function"""
    if output.size()[0] == 0:
        return torch.tensor(0.0, device=device)
    else:
        loss = nn.MSELoss(reduction='mean').to(device)
        class_num = output.size(1) 
        one_hot = F.one_hot(label.to(torch.int64), class_num) * 2 - 1
        sig_out = output * one_hot.to(device)
        y_label = torch.ones(sig_out.size()).to(device)
        return loss(sig_out, y_label)


def calculate_custom_loss(outputs, labels, s_values, label_probs, lambda_weight=1.0):
    batch_size, num_classes = outputs.shape

    # initialization
    loss_s0 = torch.tensor(0.0, device=device)  # s=0
    loss_s1 = torch.tensor(0.0, device=device)  # s=1

    count_s0 = 0  # s=0 sample
    count_s1 = 0  # s=1 sample

    for i in range(batch_size):
        if s_values[i] == 0:
            sample_output = outputs[i].unsqueeze(0)  # [1, num_classes]

            class_losses = []
            for c in range(num_classes):
                class_label = torch.tensor([c], device=device)
                class_loss = c_loss(sample_output, class_label)
                class_losses.append(class_loss)

            class_losses = torch.stack(class_losses)  # [num_classes]

            p_clip = label_probs[i]  # [num_classes]
            if lambda_weight == 1.0:
                p_mixed = p_clip
            elif lambda_weight == 0.0:
                p_mixed = F.softmax(outputs[i], dim=0)
            else:
                p_model = F.softmax(outputs[i], dim=0)  # [num_classes]
                p_mixed = lambda_weight * p_clip + (1 - lambda_weight) * p_model  # [num_classes]

            weighted_loss = torch.sum(p_mixed * class_losses)
            loss_s0 += weighted_loss
            count_s0 += 1
        else:
            sample_output = outputs[i].unsqueeze(0)  # [1, num_classes]
            sample_label = labels[i].unsqueeze(0)  # [1]

            loss_s1 += c_loss(sample_output, sample_label)
            count_s1 += 1

    # Calculate average loss
    if count_s0 > 0:
        loss_s0 /= count_s0
    if count_s1 > 0:
        loss_s1 /= count_s1

    return loss_s0 + loss_s1

