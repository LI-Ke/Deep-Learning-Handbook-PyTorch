import torch
from torch import nn
import matplotlib.pyplot as plt

from Foundation.data.data import load_array
from Foundation.utils.accumulator import Accumulator
from Foundation.utils.timer import Timer


def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D


def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G


def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = Timer()
        metric = Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        print(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
    plt.scatter(data[:100, 0], data[:100, 1], c='b')
    plt.scatter(fake_X[:100, 0], fake_X[:100, 1], c='r')
    plt.show()


if __name__ == '__main__':
    X = torch.normal(0.0, 1, (1000, 2))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([1, 2])
    data = torch.matmul(X, A) + b

    plt.scatter(data[:100, 0].detach().numpy(), data[:100, 1].detach().numpy(), c='b')
    plt.show()
    print(f'The covariance matrix is\n{torch.matmul(A.T, A)}')

    batch_size = 8
    data_iter = load_array((data,), batch_size)

    # generator
    net_G = nn.Sequential(nn.Linear(2, 2))

    # discriminator
    net_D = nn.Sequential(
        nn.Linear(2, 5), nn.Tanh(),
        nn.Linear(5, 3), nn.Tanh(),
        nn.Linear(3, 1))

    lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
    train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
          latent_dim, data[:100].detach().numpy())