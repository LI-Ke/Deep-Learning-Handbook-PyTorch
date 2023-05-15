import torch
from torch import nn

from D2L.data.data import load_data_fashion_mnist
from D2L.models.cnn.lenet import train_gpu

if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10))

    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.01, 10
    train_gpu(net, train_iter, test_iter, num_epochs, lr, 'cuda')