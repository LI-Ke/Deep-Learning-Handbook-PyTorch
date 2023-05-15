import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(x, max_x):
    return torch.eye(max_x + 1, device=x.device)[x]


class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_size, class_size, hidden_size=10):
        super(ConditionalVariationalAutoencoder, self).__init__()

        self.class_size = class_size
        input_size_with_label = input_size + self.class_size + 1
        hidden_size_with_label = hidden_size + self.class_size + 1

        self.fc1 = nn.Linear(input_size_with_label, 512)
        self.fc21 = nn.Linear(512, hidden_size)
        self.fc22 = nn.Linear(512, hidden_size)

        self.relu = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size_with_label, 512)
        self.fc4 = nn.Linear(512, input_size)

    def encoder(self, x, labels):
        x = torch.cat((x, labels), dim=1)
        encoded = self.relu(self.fc1(x))
        return self.fc21(encoded), self.fc22(encoded)

    def decoder(self, z, labels):
        z = torch.cat((z, labels), dim=1)
        decoded = self.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(decoded))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, labels):
        labels = one_hot(labels, self.class_size).float().to(x.device)
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, labels)
        return recon_x, mu, logvar
