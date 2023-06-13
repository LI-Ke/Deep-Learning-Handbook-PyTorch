import torch
from torch import nn

from Foundation.data.data import load_data_fashion_mnist
from Foundation.utils.train import train


if __name__ == '__main__':
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print(X)

    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(num_inputs, num_hiddens1),
                        nn.ReLU(),
                        # add Dropout layer after the 1st FC layer
                        nn.Dropout(dropout1),
                        nn.Linear(num_hiddens1, num_hiddens2),
                        nn.ReLU(),
                        # add Dropout layer after the 2nd FC layer
                        nn.Dropout(dropout2),
                        nn.Linear(num_hiddens2, num_outputs))


    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, loss, num_epochs, optimizer)

