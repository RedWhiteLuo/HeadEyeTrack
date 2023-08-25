import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Sequential, Linear, LeakyReLU, Dropout, Tanh, ELU

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EyeTrackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            # in-> [N, 3, 32, 128]
            Conv2d(3, 2, kernel_size=(5, 5), padding=2),
            LeakyReLU(),
            #   -> [N, 3, 32, 128]
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2d(2, 20, kernel_size=(5, 5), padding=2),
            ELU(),
            #   -> [N, 16, 16, 64]
            Conv2d(20, 10, kernel_size=(5, 5), padding=2),
            Tanh(),
            #   -> [N, 10, 16, 64]
            Flatten(1, 3),
            Dropout(0.002),
            #   -> [N, 10240]
            Linear(10240, 2)
            # out-> [N, 2]
        )

    def forward(self, x):
        return self.model(x)
