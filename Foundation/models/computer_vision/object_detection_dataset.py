import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision

from Foundation.data.data import download_extract
from Foundation.models.computer_vision.object_detection import show_bboxes
from Foundation.utils.func import show_images


def read_data_bananas(cache_dir, is_train=True):
    """read bananas dataset images and labels"""
    data_dir = download_extract('banana-detection', cache_dir)
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
            'bananas_val', 'images', f'{img_name}')))
        # the target includes（class，upperleft x, upperleft y，lowerright x，lowerright y），
        # all images have the same class banana (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, is_train):
        self.features, self.labels = read_data_bananas(cache_dir, is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
                                                   is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


def load_data_bananas(cache_dir, batch_size):
    """load bananas dataset"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(cache_dir, is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(cache_dir, is_train=False),
                                           batch_size)
    return train_iter, val_iter


if __name__ == '__main__':
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_bananas('../../dataset/object_detection', batch_size)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape)

    imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
    axes = show_images(imgs, 2, 5, scale=2)
    for ax, label in zip(axes, batch[1][0:10]):
        show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
    plt.show()