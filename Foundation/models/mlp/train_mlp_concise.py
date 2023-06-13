import torch
from torch import nn

from Foundation.data.data import load_data_fashion_mnist
from Foundation.utils.eval import predict_fashion_mnist
from Foundation.utils.train import train


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(num_inputs, num_hiddens),
                        nn.ReLU(),
                        nn.Linear(num_hiddens, num_outputs))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction='none')

    num_epochs, lr = 5, 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, loss, num_epochs, optimizer)
    predict_fashion_mnist(net, test_iter)

