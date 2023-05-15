import torch
from torch import nn

from D2L.data.data import load_data_fashion_mnist
from D2L.utils.train import train


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # in this case, all elements will be removed
    if dropout == 1:
        return torch.zeros_like(X)
    # in this case, all elements will be kept
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.fc1 = nn.Linear(num_inputs, num_hiddens1)
        self.fc2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.fc3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        h1 = self.relu(self.fc1(X.reshape((-1, self.num_inputs))))
        if self.training:
            h1 = dropout_layer(h1, dropout1)
        h2 = self.relu(self.fc2(h1))
        if self.training:
            h2 = dropout_layer(h2, dropout2)
        out = self.fc3(h2)
        return out


if __name__ == '__main__':
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print(X)
    print(dropout_layer(X, 0.))
    print(dropout_layer(X, 0.5))
    print(dropout_layer(X, 1.))

    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5

    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, loss, num_epochs, optimizer)

