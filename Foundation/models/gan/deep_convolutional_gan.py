import warnings

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn

from Foundation.data.data import download_extract, get_dataloader_workers
from Foundation.models.gan.simple_gan import update_D, update_G
from Foundation.utils.accumulator import Accumulator
from Foundation.utils.func import show_images
from Foundation.utils.timer import Timer


class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))


class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, device='cuda'):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas': [0.5,0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        timer = Timer()
        metric = Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
            break
        # Show generated examples
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
            [torch.cat([
                fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
             for i in range(len(fake_x)//7)], dim=0)
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        print(epoch, (loss_D, loss_G))
    show_images(imgs, num_rows=1, num_cols=1)
    plt.show()
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')


if __name__ == '__main__':
    data_dir = download_extract('pokemon', '../../dataset')
    pokemon = torchvision.datasets.ImageFolder(data_dir)

    batch_size = 256
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    pokemon.transform = transformer
    data_iter = torch.utils.data.DataLoader(
        pokemon, batch_size=batch_size,
        shuffle=True, num_workers=get_dataloader_workers())

    warnings.filterwarnings('ignore')
    for X, y in data_iter:
        imgs = X[:20, :, :, :].permute(0, 2, 3, 1) / 2 + 0.5
        show_images(imgs, num_rows=4, num_cols=5)
        plt.show()
        break

    n_G = 64
    net_G = nn.Sequential(
        G_block(in_channels=100, out_channels=n_G * 8,
                strides=1, padding=0),  # Output: (64 * 8, 4, 4)
        G_block(in_channels=n_G * 8, out_channels=n_G * 4),  # Output: (64 * 4, 8, 8)
        G_block(in_channels=n_G * 4, out_channels=n_G * 2),  # Output: (64 * 2, 16, 16)
        G_block(in_channels=n_G * 2, out_channels=n_G),  # Output: (64, 32, 32)
        nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
                           kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh())  # Output: (3, 64, 64)

    x = torch.zeros((1, 100, 1, 1))
    print(net_G(x).shape)

    n_D = 64
    net_D = nn.Sequential(
        D_block(n_D),  # Output: (64, 32, 32)
        D_block(in_channels=n_D, out_channels=n_D * 2),  # Output: (64 * 2, 16, 16)
        D_block(in_channels=n_D * 2, out_channels=n_D * 4),  # Output: (64 * 4, 8, 8)
        D_block(in_channels=n_D * 4, out_channels=n_D * 8),  # Output: (64 * 8, 4, 4)
        nn.Conv2d(in_channels=n_D * 8, out_channels=1,
                  kernel_size=4, bias=False))

    x = torch.zeros((1, 3, 64, 64))
    print(net_D(x).shape)

    latent_dim, lr, num_epochs = 100, 0.005, 1
    train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
