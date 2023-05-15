import torch


def sgd(params, lr, batch_size):
    """minibatch stochastic gradient descent"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()