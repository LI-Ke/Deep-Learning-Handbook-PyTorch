import torch
import torchvision
from PIL.Image import Image
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from D2L.data.data import download_extract
from D2L.models.computer_vision.augmentation import train_ch13
from D2L.models.computer_vision.semantic_segmentation_dataset import load_data_voc, VOC_COLORMAP, read_voc_images
from D2L.utils.func import show_images


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]


if __name__ == '__main__':
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    num_classes = 21
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                        kernel_size=64, padding=16, stride=32))

    conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                    bias=False)
    conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)

    batch_size, crop_size = 16, (320, 480)
    train_iter, test_iter = load_data_voc(batch_size, crop_size, '../../dataset/semantic_segmentation')

    num_epochs, lr, wd, devices = 5, 0.001, 1e-3, ['cuda']
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

    voc_dir = download_extract('voc2012', '../../dataset/semantic_segmentation', 'VOCdevkit/VOC2012')
    test_images, test_labels = read_voc_images(voc_dir, False)
    n, imgs = 4, []
    for i in range(n):
        crop_rect = (0, 0, 320, 480)
        X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
        pred = label2image(predict(X))
        imgs += [X.permute(1, 2, 0), pred.cpu(),
                 torchvision.transforms.functional.crop(
                     test_labels[i], *crop_rect).permute(1, 2, 0)]
    show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);