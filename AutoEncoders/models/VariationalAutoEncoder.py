import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(VariationalAutoencoder, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.fc21 = nn.Linear(512, 10)
        self.fc22 = nn.Linear(512, 10)

        self.relu = nn.ReLU()

        self.fc3 = nn.Linear(10, 512)
        self.fc4 = nn.Linear(512, input_size)

    def encoder(self, x):
        encoded = self.relu(self.fc1(x))
        return self.fc21(encoded), self.fc22(encoded)

    def decoder(self, z):
        decoded = self.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(decoded))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


def vae_loss_fn(x, recon_x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD