import torch
import torchvision
from torch import nn
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from D2L.data.data import get_dataloader_workers
from D2L.models.cnn.lenet import evaluate_accuracy_gpu
from D2L.models.performance.train_multi_gpu_concise import resnet18
from D2L.utils.accumulator import Accumulator
from D2L.utils.eval import accuracy
from D2L.utils.timer import Timer


def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../../dataset", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=is_train, num_workers=get_dataloader_workers())
    return dataloader


def train_batch_ch13(net, X, y, loss, optimizer, devices):
    """use multiple GPUs to train mini batch"""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    optimizer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    optimizer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=['cuda']):
    """use multiple GPUs to train the model"""
    timer, num_batches = Timer(), len(train_iter)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # training loss, training acc，#sample，count
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(epoch + (i + 1) / num_batches,
                      (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(epoch + 1, test_acc)
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


def train_with_data_aug(train_augs, test_augs, net, batch_size, devices,
                        lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


if __name__ == '__main__':
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])

    batch_size, devices, net = 256, ['cuda'], torchvision.models.resnet18(10, 3)

    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    train_with_data_aug(train_augs, test_augs, net, batch_size, devices)

