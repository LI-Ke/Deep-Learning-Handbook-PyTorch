import numpy as np
import torch
from torch import nn

from D2L.utils.timer import Timer


def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2))
    return net


class Benchmark:
    """Evaluate running time"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')


if __name__ == '__main__':
        # x = torch.randn(size=(1, 512))
        #
        # net = get_net()
        # with Benchmark('without torchscript'):
        #     for i in range(1000): net(x)
        #
        # net = torch.jit.script(net)
        # with Benchmark('with torchscript'):
        #     for i in range(1000): net(x)

    device = 'cuda'
    a = torch.randn(size=(1000, 1000), device=device)
    b = torch.mm(a, a)

    with Benchmark('numpy'):
        for _ in range(10):
            a = np.random.normal(size=(1000, 1000))
            b = np.dot(a, a)

    with Benchmark('torch'):
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)