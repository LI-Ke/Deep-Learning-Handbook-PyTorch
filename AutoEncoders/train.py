import numpy as np
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from AutoEncoders.models.AutoEncoder import Autoencoder
from AutoEncoders.models.ConditionalVariationalAutoEncoder import ConditionalVariationalAutoencoder
from AutoEncoders.models.ConvolutionalAutoEncoder import CNNAutoencoder
from AutoEncoders.models.VariationalAutoEncoder import VariationalAutoencoder, vae_loss_fn


def calculate_loss(model, dataloader, loss_fn=nn.MSELoss(), flatten=True, conditional=False):
    losses = []
    for batch, labels in dataloader:
        batch = batch.to(device)
        labels = labels.to(device)

        if flatten:
            batch = batch.view(batch.size(0), 28 * 28)

        if conditional:
            loss = loss_fn(batch, model(batch, labels))
        else:
            loss = loss_fn(batch, model(batch))

        losses.append(loss)

    return (sum(losses) / len(losses)).item()  # calculate mean


def show_visual_progress(model, test_dataloader, rows=5, flatten=True, vae=False, conditional=False, title=None, output=None):
    if title:
        plt.title(title)

    iter(test_dataloader)

    image_rows = []

    for idx, (batch, label) in enumerate(test_dataloader):
        if rows == idx:
            break

        batch = batch.to(device)

        if flatten:
            batch = batch.view(batch.size(0), 28 * 28)

        if not conditional:
            images = model(batch).detach().cpu().numpy().reshape(batch.size(0), 28, 28)
        else:
            images = model(batch, label).detach().cpu().numpy().reshape(batch.size(0), 28, 28)

        image_idxs = [list(label.numpy()).index(x) for x in range(10)]
        combined_images = np.concatenate([images[x].reshape(28, 28) for x in image_idxs], 1)
        image_rows.append(combined_images)

    plt.imshow(np.concatenate(image_rows))

    if title:
        title = title.replace(" ", "_")
        if output:
            if not os.path.exists(output):
                os.mkdir(output)
            title = os.path.join(output, title)
        plt.savefig(title)


def evaluate(losses, autoencoder, dataloader, flatten=True, vae=False, conditional=False, title="", output=None):
    if vae and conditional:
        model = lambda x, y: autoencoder(x, y)[0]
    elif vae:
        model = lambda x: autoencoder(x)[0]
    else:
        model = autoencoder

    loss = calculate_loss(model, dataloader, flatten=flatten, conditional=conditional)
    show_visual_progress(model, dataloader, flatten=flatten, vae=vae, conditional=conditional, title=title, output=output)

    losses.append(loss)


def train(net, dataloader, test_dataloader, epochs=5, flatten=False, loss_fn=nn.MSELoss(), title=None, output=None, vae=False, conditional=False):
    optim = torch.optim.Adam(net.parameters())

    train_losses = []
    validation_losses = []

    for i in range(epochs):
        for batch, labels in dataloader:
            batch = batch.to(device)
            labels = labels.to(device)

            if flatten:
                batch = batch.view(batch.size(0), 28 * 28)

            optim.zero_grad()
            if vae:
                if conditional:
                    recon_x, mu, logvar = net(batch, labels)
                    # recon_x = recon_x[:, :784]
                else:
                    recon_x, mu, logvar = net(batch)
                loss = vae_loss_fn(batch, recon_x, mu, logvar)
            else:
                loss = loss_fn(batch, net(batch))

            loss.backward()
            optim.step()

            train_losses.append(loss.item())
        if title:
            image_title = f'{title} - Epoch {i}'
        evaluate(validation_losses, net, test_dataloader, flatten, vae, conditional, title=image_title, output=output)
        print('epoch: {}, train loss: {}, validation loss: {}'.format(i, loss.item(), validation_losses[i]))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    out_path = 'output'

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataset = datasets.MNIST('./dataset', transform=transform, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST('./dataset', transform=transform, download=True, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # autoencoder = Autoencoder(28*28).to(device)
    # train(autoencoder, train_dataloader, test_dataloader, epochs=10, flatten=True, title='Autoencoder', output=out_path)

    # cnnAutoencoder = CNNAutoencoder().to(device)
    # train(cnnAutoencoder, train_dataloader, test_dataloader, epochs=10, title='Convolutional Autoencoder', output=out_path)

    # vae = VariationalAutoencoder(28 * 28).to(device)
    # train(vae, train_dataloader, test_dataloader, epochs=10, flatten=True, title='Variational Autoencoder', output=out_path, vae=True)

    cvae = ConditionalVariationalAutoencoder(28 * 28, 10).to(device)
    train(cvae, train_dataloader, test_dataloader, epochs=10, flatten=True, title='Conditional VAE', output=out_path, vae=True, conditional=True,)
