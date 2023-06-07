import torch


def linreg(X, w, b):
    """linear regression model"""
    return torch.matmul(X, w) + b


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

