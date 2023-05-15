import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn

from D2L.data.data import download_extract
from D2L.models.cnn.lenet import evaluate_accuracy_gpu
from D2L.models.computer_vision.augmentation import train_batch_ch13
from D2L.models.performance.train_multi_gpu_concise import resnet18
from D2L.utils.accumulator import Accumulator
from D2L.utils.timer import Timer


def read_csv_labels(fname):
    with open(fname, 'r') as f:
        # skip first line
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


def copyfile(filename, target_dir):
    """copy file to target dir"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    """decompose valid set from training set"""
    # the number of samples in the class with the fewest samples
    # in the training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # number of samples of each class in the valid set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label


def reorg_test(data_dir):
    """prepare test set for prediction"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))


def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


def get_net():
    num_classes = 10
    net = resnet18(num_classes, 3)
    return net


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), Timer()
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels,
                                      loss, optimizer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(epoch + (i + 1) / num_batches,
                     (metric[0] / metric[2], metric[1] / metric[2]))
        if valid_iter is not None:
            valid_acc = evaluate_accuracy_gpu(net, valid_iter)
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
                     f' examples/sec on {str(devices)}')




if __name__ == '__main__':
    demo = True
    if demo:
        data_dir = download_extract('cifar10_tiny', '../../dataset')
    else:
        data_dir = '../../dataset/cifar-10/'

    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    print('# training examples:', len(labels))
    print('# labels:', len(set(labels.values())))

    batch_size = 32 if demo else 128
    valid_ratio = 0.1
    reorg_cifar10_data(data_dir, valid_ratio)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(40),
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                 ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])

    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]

    train_iter, train_valid_iter = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)
        for dataset in (train_ds, train_valid_ds)]
    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                             drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                            drop_last=False)

    num_classes = 10
    loss = nn.CrossEntropyLoss(reduction="none")
    devices, num_epochs, lr, wd = ['cuda'], 20, 2e-4, 5e-4
    lr_period, lr_decay, net = 4, 0.9, get_net()
    train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay)

    net, preds = get_net(), []
    train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
          lr_decay)
    for X, _ in test_iter:
        y_hat = net(X.to(devices[0]))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    sorted_ids = list(range(1, len(test_ds) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
    df.to_csv('../../dataset/kaggle_cifar10_tiny/train_valid_test/predictions.csv', index=False)