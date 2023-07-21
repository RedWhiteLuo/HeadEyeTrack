import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

input = torch.randn(20, 160, 50)
m = nn.Conv1d(160, 2, 50)
output = m(input)
print(output.shape)
