import torch

from D2L.data.data import load_data_fashion_mnist
from D2L.models.linear_regression.train_linear_regression_scratch import sgd
from D2L.models.loss import cross_entropy
from D2L.models.model import softmax
from D2L.utils.eval import accuracy, evaluate_accuracy, predict_fashion_mnist
from D2L.utils.train import train


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def updater(batch_size, lr=0.1):
    return sgd([W, b], lr, batch_size)


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])

    print(accuracy(y_hat, y) / len(y))

    evaluate_accuracy(net, test_iter)

    num_epochs = 10
    train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict_fashion_mnist(net, test_iter)