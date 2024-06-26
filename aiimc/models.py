import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


__all__ = [
    'MLP',
]

# Define an MLP model for MNIST/ fashionMNIST
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.L1 = nn.Linear(28 * 28, 512)
        self.L2 = nn.Linear(512, 256)
        self.L3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.name = '3Layer_MLP_MNIST'

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        x = self.L3(x)
        return x