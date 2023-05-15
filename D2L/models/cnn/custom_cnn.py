import torch
from torch import nn


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i+h, j:j+w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def comp_conv2d(conv2d, X):
    # use (1, 1) to represent the batch_size and the in_channels
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # remove the first 2 dimensions
    return Y.reshape(Y.shape[2:])


def corr2d_multi_in(X, K):
    """Multiple in_channels"""
    return sum(corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    """Multiple in_channels and out_channels"""
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


if __name__ == '__main__':
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, K))

    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(X)
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)
    print(corr2d(X.t(), K))

    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {l.sum():.3f}')

    print(conv2d.weight.data.reshape((1, 2)))

    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print(comp_conv2d(conv2d, X).shape)

    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    print(corr2d_multi_in(X, K))

    K = torch.stack((K, K + 1, K + 2), 0)
    print(K.shape)

    print(corr2d_multi_in_out(X, K))

    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    assert float(torch.abs(Y1 - Y2).sum()) < 1e-6

    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), 'avg'))
