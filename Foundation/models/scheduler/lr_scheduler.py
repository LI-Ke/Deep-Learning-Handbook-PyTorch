import math
import torch
from torch import nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from Foundation.data.data import load_data_fashion_mnist
from Foundation.models.cnn.lenet import evaluate_accuracy_gpu
from Foundation.utils.accumulator import Accumulator
from Foundation.utils.eval import accuracy


def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))
    return model


def train(net, train_iter, test_iter, num_epochs, loss, optimizer, device,
          scheduler=None):
    net.to(device)
    for epoch in range(num_epochs):
        metric = Accumulator(3) # train_loss,train_acc,num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                print((epoch + i / len(train_iter)), train_loss, train_acc)

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(epoch+1, test_acc)

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # UsingPyTorchIn-Builtscheduler
                scheduler.step()
            else:
                # Usingcustomdefinedscheduler
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')


class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)


class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr


class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
                 warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                   * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                    self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


if __name__ == '__main__':
    loss = nn.CrossEntropyLoss()
    batch_size, device = 256, 'cuda'
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epochs = 0.3, 30
    net = net_fn()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # train(net, train_iter, test_iter, num_epochs, loss, optimizer, device)

    lr = 0.1
    optimizer.param_groups[0]["lr"] = lr
    print(f'learning rate is now {optimizer.param_groups[0]["lr"]:.2f}')

    scheduler = SquareRootScheduler(lr=0.1)
    # plt.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
    # plt.show()
    # train(net, train_iter, test_iter, num_epochs, loss, optimizer, device,
    #       scheduler)

    # factor scheduler
    # scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
    # plt.plot(torch.arange(50), [scheduler(t) for t in range(50)])
    # plt.show()

    # multistep scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.5)

    def get_lr(optimizer, scheduler):
        lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        return lr
    # plt.plot(torch.arange(num_epochs), [get_lr(optimizer, scheduler)
    #                                     for t in range(num_epochs)])
    # plt.show()

    # cosine scheduler
    scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
    # plt.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
    # plt.show()

    # warmup
    scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
    plt.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
    plt.show()

