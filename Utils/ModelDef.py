from typing import Any

import torch
from torch import nn


class Unit(nn.Module):

    def __init__(self, inChn, outChn, ceil_mode=False, padding=0) -> None:
        super(Unit, self).__init__()

        self.conv1 = nn.Conv2d(inChn, outChn, (3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(outChn, outChn, (1, 1))
        self.relu2 = nn.ReLU()
        # self.bn = nn.BatchNorm2d(outChn)
        self.pool = nn.MaxPool2d((2, 2), stride=2, padding=padding, ceil_mode=ceil_mode)

        # self.module = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2, self.bn, self.pool)
        self.module = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2, self.pool)

    def forward(self, x):
        return self.module(x)


class PstModel(nn.Module):

    def __init__(self) -> None:
        super(PstModel, self).__init__()

        part = [nn.Conv2d(2, 4, (1, 1)),  # 21 * 8 * 4
                nn.Conv2d(4, 8, (3, 3), padding=1),  # 21 * 8 * 8
                nn.ReLU(),  # 21 * 8 * 8
                nn.MaxPool2d((2, 2), stride=2, padding=1, ceil_mode=True),
                Unit(8, 16, ceil_mode=False, padding=1),
                Unit(16, 32, ceil_mode=True, padding=1),
                Unit(32, 64, ceil_mode=False, padding=0),
                nn.Conv2d(64, 128, (1, 1))
                ]

        self.cnnModel = nn.ModuleList(part)

        part = [nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 20),
                nn.Softmax(dim=1)]

        self.linearModel = nn.ModuleList(part)

    def forward(self, x):

        # print(x.shape)
        for layer in self.cnnModel:
            x = layer(x)
            # print(x.shape)

        x = torch.squeeze(x)
        x = torch.Tensor.view(x, (x.shape[0], -1))

        for layer in self.linearModel:
            # print(x.shape)
            x = layer(x)

        # print(x)
        return x


class PstLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.numPst = 3

    def forward(self, out: torch.Tensor, target: torch.Tensor):
        # loss = 0 - (torch.log(out) * target).sum() / out.shape[0]

        loss = -(torch.log(out[target > 0]).sum()) / out.shape[0]

        # loss = torch.abs(out - target).sum() / out.shape[0]

        # idx = (-target).argsort()[:, :3]
        #
        # for i in range(out.shape[0]):
        #
        #     out[i, idx[i]] = 1-out[i, idx[i]]
        #
        #
        # loss = out.sum()

        return loss

# data = torch.rand((2, 2, 21, 8), dtype=torch.float32)
#
# model = PstModel()
#
# out = model(data)
