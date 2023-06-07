import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        # Call the constructor of MLP's parent class Module
        # to perform the necessary initialization
        super().__init__()
        self.hidden = nn.Linear(20, 256) # hidden layer
        self.out = nn.Linear(256, 10) # output layer

    # forward propagation
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


if __name__ == '__main__':
    X = torch.rand(2, 20)
    net = MLP()
    print(net(X))