import torch
from torch import nn


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


if __name__ == '__main__':
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    print(net(X))
    print(net[2].state_dict())
    print(net.state_dict())
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)
    print(net[2].weight.grad == None)
    print(*[(name, param.shape) for name, param in net.named_parameters()])

    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    print(rgnet)
    print(rgnet(X))
    print(rgnet[0][1][0].bias.data)
