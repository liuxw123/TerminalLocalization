from torch import nn

from config import *


class PstModel(nn.Module):

    def __init__(self) -> None:
        super(PstModel, self).__init__()

        part = [nn.Linear(41, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU,
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, numKnownPoint),
                nn.Softmax()]

        self.model = nn.ModuleList(part)


    def forward(self, x):

        for layer in self.model:
            x = layer(x)

        return x

