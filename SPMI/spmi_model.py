# modify from https://github.com/facebookresearch/moco/blob/main/moco/builder.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import CLIPLinearModel


class Model(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()

        # encoder
        self.encoder_q = base_encoder(input_dim=768, num_class=args.num_class)
        # self.encoder_q = base_encoder(name=args.arch, num_class=args.num_class)

    def forward(self, img_q):
        features, logits = self.encoder_q(img_q)
        return logits


class clip_mlinear(nn.Module):
    def __init__(self, input_dim, num_class):
        super(clip_mlinear, self).__init__()
        # 只有一个线性层
        # self.classifier = nn.Linear(input_dim, num_class)
        # self.fc1 = nn.Linear(input_dim, 512)
        # self.fc2 = nn.Linear(512, num_class)
        self.fc = CLIPLinearModel(num_classes=num_class)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.fc1(x))
        x = self.fc(x)
        return x

        # 输入直接通过线性层得到输出
        # return self.classifier(x)
