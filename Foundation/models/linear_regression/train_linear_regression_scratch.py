import torch

from Foundation.data.data import data_iter
from Foundation.models.loss import squared_loss
from Foundation.models.model import linreg
from Foundation.models.optim import sgd
from Foundation.utils.func import synthetic_data


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    print('features:', features[0], '\nlabel:', labels[0])

    batch_size = 10

    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'Estimated error of w: {true_w - w.reshape(true_w.shape)}')
    print(f'Estimated error of b: {true_b - b}')