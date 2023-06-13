import torch
from torch import nn

from Foundation.data.data import load_data_fashion_mnist
from Foundation.models.activation import relu
from Foundation.utils.eval import predict_fashion_mnist
from Foundation.utils.train import train


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(torch.matmul(X, W1)) + b1
    return torch.matmul(H, W2) + b2


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    loss = nn.CrossEntropyLoss(reduction='none')

    num_epochs, lr = 10, 0.1
    optimizer = torch.optim.SGD(params, lr=lr)
    train(net, train_iter, test_iter, loss, num_epochs, optimizer)
    predict_fashion_mnist(net, test_iter)

