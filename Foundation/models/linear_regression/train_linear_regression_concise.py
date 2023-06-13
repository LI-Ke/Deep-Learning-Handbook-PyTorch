import torch
from torch import nn

from Foundation.data.data import load_array
from Foundation.utils.func import synthetic_data


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[0], '\nlabel:', labels[0])

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    print(next(iter(data_iter)))

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()

    lr = 0.03
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l: f}')

    w = net[0].weight.data
    b = net[0].bias.data
    print(f'Estimated error of w: {true_w - w.reshape(true_w.shape)}')
    print(f'Estimated error of b: {true_b - b}')