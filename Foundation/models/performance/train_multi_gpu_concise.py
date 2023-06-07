import torch
from torch import nn

from D2L.data.data import load_data_fashion_mnist
from D2L.models.cnn.lenet import evaluate_accuracy_gpu
from D2L.models.cnn.resnet import Residual
from D2L.utils.timer import Timer


def resnet18(num_classes, in_channels=1):
    """slightly modified ResNet-18"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # this model uses smaller conv kernels, paddings and strides,
    # and removes the MaxPooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net


def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = ['cuda:0']
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # set model on multi-gpus
    net = nn.DataParallel(net, device_ids=devices)
    optimizer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = Timer(), 10
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        timer.stop()
        print(epoch + 1, (evaluate_accuracy_gpu(net, test_iter)))


if __name__ == '__main__':
    net = resnet18(10)
    train(net, num_gpus=1, batch_size=256, lr=0.1)
