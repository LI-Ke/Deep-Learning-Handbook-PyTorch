import torch


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)