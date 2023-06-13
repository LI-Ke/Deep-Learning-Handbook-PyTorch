import  matplotlib.pyplot as plt
import torch
from torch import nn

from Foundation.models.attention.vis import show_heatmaps


def f(x):
    return 2 * torch.sin(x) + x**0.8


def plot_kernel_reg(y_hat):
    plt.plot(x_test, y_truth, label='Truth')
    plt.plot(x_test, y_hat, label='Pred')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0,5)
    plt.ylim(-1, 5)
    plt.plot(x_train, y_train, 'o', alpha=0.5)
    plt.legend()
    plt.show()


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries and attention_weights shapes: (queries，k-v pairs)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values shape(queries，k-v pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)


if __name__ == '__main__':
    n_train = 50  # training samples
    x_train, _ = torch.sort(torch.rand(n_train) * 5) # sorted training samples
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # train labels
    x_test = torch.arange(0, 5, 0.1)  # test samples
    y_truth = f(x_test) # test labels
    n_test = len(x_test)
    # y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    # plot_kernel_reg(y_hat)

    # X_repeat shape:(n_test,n_train),
    # X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # attention_weights shape：(n_test,n_train),
    # attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    # y_hat = torch.matmul(attention_weights, y_train)
    # plot_kernel_reg(y_hat)
    #
    # show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
    #               xlabel='Sorted training inputs',
    #               ylabel='Sorted testing inputs')
    #
    # weights = torch.ones((2, 10)) * 0.1
    # values = torch.arange(20.0).reshape((2, 10))
    # torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))

    # X_tile shape:(n_train，n_train)，each row contains the same training input
    X_tile = x_train.repeat((n_train, 1))
    # Y_tile shape:(n_train，n_train)，each row contains the same training output
    Y_tile = y_train.repeat((n_train, 1))
    # keys shape:('n_train'，'n_train'-1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # values shape:('n_train'，'n_train'-1)
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')

    # keys shape:(n_test，n_train)
    keys = x_train.repeat((n_test, 1))
    # value shape:(n_test，n_train)
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    plot_kernel_reg(y_hat)
    show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')