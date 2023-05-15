import torch
from torch import nn

from D2L.data.data import load_data_fashion_mnist
from D2L.utils.accumulator import Accumulator
from D2L.utils.eval import accuracy
from D2L.utils.timer import Timer


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Use GPU"""
    if isinstance(net, nn.Module):
        net.eval() # set to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # correct prediction number, total prediction number
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_gpu(net, train_iter, test_iter, num_epochs, lr, device):
    """Use GPU to train the model"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on ', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # total training loss, total accuracy, sample number
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch}, train loss {train_l}, train acc {train_acc}, test acc {test_acc}')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10), nn.Softmax(dim=-1))

    # X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape: \t', X.shape)

    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.05, 10
    train_gpu(net, train_iter, test_iter, num_epochs, lr, 'cuda')