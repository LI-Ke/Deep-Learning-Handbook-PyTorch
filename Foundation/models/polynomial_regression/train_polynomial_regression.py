import math
import numpy as np
import torch
from torch import nn

from Foundation.data.data import load_array
from Foundation.utils.train import train_epoch


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1,1)), batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1,1)), batch_size, is_train=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, optimizer)
    print('weight:', net[0].weight.data.numpy())


if __name__ == '__main__':
    """
       y = 5 + 1.12x - 3.4 * (x ** 2) / 2! + 5.6 * (x ** 3) / 3! + epsilon 
    """
    max_degree = 20
    n_train, n_test = 100, 100
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!

    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
    train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])