import os
import torch
import torchvision
from torch import nn

from Foundation.data.data import download_extract
from Foundation.models.computer_vision.object_recognition import read_csv_labels, reorg_train_valid, reorg_test
from Foundation.utils.accumulator import Accumulator
from Foundation.utils.timer import Timer


def reorg_dog_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # define a new output layer with 120 outputs
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    finetune_net = finetune_net.to(devices[0])
    # freeze pretrained params
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net


def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # only train self-defined layers
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), Timer()
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            optimizer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            optimizer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(epoch + (i + 1) / num_batches, (metric[0] / metric[1]))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            print(epoch + 1, f'valid loss {valid_loss.detach().cpu()}')
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
                     f' examples/sec on {str(devices)}')


if __name__ == '__main__':
    demo = True
    if demo:
        data_dir = download_extract('dog_tiny', '../../dataset')
    else:
        data_dir = os.path.join('../..', 'dataset', 'dog-breed-identification')

    batch_size = 32 if demo else 128
    valid_ratio = 0.1
    reorg_dog_data(data_dir, valid_ratio)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                 ratio=(3.0 / 4.0, 4.0 / 3.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.4,
                                           contrast=0.4,
                                           saturation=0.4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])

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

    loss = nn.CrossEntropyLoss(reduction='none')

    devices, num_epochs, lr, wd = ['cuda'], 10, 1e-4, 1e-4
    lr_period, lr_decay, net = 2, 0.9, get_net(devices)
    train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
          lr_decay)

    preds = []
    for data, label in test_iter:
        output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
        preds.extend(output.cpu().detach().numpy())
    ids = sorted(os.listdir(
        os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
    with open('../../dataset/kaggle_dog_tiny/train_valid_test/prediction.csv', 'w') as f:
        f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
        for i, output in zip(ids, preds):
            f.write(i.split('.')[0] + ',' + ','.join(
                [str(num) for num in output]) + '\n')