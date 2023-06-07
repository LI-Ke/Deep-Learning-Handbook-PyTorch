import torch

from D2L.utils.accumulator import Accumulator
from D2L.utils.eval import accuracy, evaluate_accuracy


def train_epoch(net, train_iter, loss, updater):
    """train the model one epoch"""
    # set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()

    # total training loss, total training accuracy, sample numbers
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2] # return training loss and train accuracy


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, train loss {train_metrics[0]:.5f}, train acc {train_metrics[1]:.5f}, test acc {test_acc:.5f}')
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc