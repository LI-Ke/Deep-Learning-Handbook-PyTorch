import random
import os
import hashlib
import tarfile

import requests
import zipfile

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

from D2L.constant.constant import DATA_HUB
from D2L.utils.func import show_images
from D2L.utils.timer import Timer


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The samples are read randomly
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def load_array(data_arrays, batch_size, is_train=True):
    """create a PyTorch dataset iterator"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_fashion_mnist_labels(labels):
    """Returns the text labels of the Fashion-MNIST dataset"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def get_dataloader_workers():
    """Use 4 processes to read dataset"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """Download Fashion-MNIST dataset, then load it into memory"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../../dataset", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def download(name, cache_dir=os.path.join('.', 'dataset/kaggle_house_pred')):
    """Download a file in DATAHUB and return the local file name"""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, cache_dir, folder=None):
    """Download and decompress zip/tar files"""
    fname = download(name, cache_dir)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be decompressed'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all(cache_dir):
    """Download all files in DATA_HUB"""
    for name in DATA_HUB:
        download(name, cache_dir)


if __name__ == '__main__':
    trans = transforms.ToTensor()

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../dataset", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../dataset", train=False, transform=trans, download=True)

    print(len(mnist_train), len(mnist_test))
    print(mnist_train[0][0].shape)

    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=get_dataloader_workers())

    timer = Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')

    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
