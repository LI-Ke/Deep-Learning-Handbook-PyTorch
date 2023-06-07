import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image


rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])


def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)


def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


def extract_features(net, X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(net, content_img, image_shape, content_layers, style_layers, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(net, content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles(net, style_img, image_shape, content_layers, style_layers, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(net, style_X, content_layers, style_layers)
    return style_X, styles_Y


def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()


def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # compute content loss, style loss and total variation loss
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # compute total loss
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, optimizer


def train(net, X, contents_Y, styles_Y, content_layers, style_layers, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, optimizer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, 0.8)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            net, X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(epoch + 1, [float(sum(contents_l)),
                              float(sum(styles_l)), float(tv_l)])
    return X


if __name__ == '__main__':
    # content_img = plt.imread('../../dataset/rainier.jpg')
    content_img = Image.open('../../dataset/rainier.jpg')
    plt.imshow(content_img)
    plt.show()

    # style_img = plt.imread('../../dataset/autumn-oak.jpg')
    style_img = Image.open('../../dataset/autumn-oak.jpg')
    plt.imshow(style_img)
    plt.show()

    pretrained_net = torchvision.models.vgg19(weights='DEFAULT')
    style_layers, content_layers = [0, 5, 10, 19, 28], [25]
    net = nn.Sequential(*[pretrained_net.features[i] for i in
                          range(max(content_layers + style_layers) + 1)])

    content_weight, style_weight, tv_weight = 1, 1e3, 10
    device, image_shape = 'cuda', (300, 450)
    net = net.to(device)
    content_X, contents_Y = get_contents(net, content_img, image_shape, content_layers, style_layers, device)
    _, styles_Y = get_styles(net, style_img, image_shape, content_layers, style_layers, device)
    output = train(net, content_X, contents_Y, styles_Y, content_layers, style_layers, device, 0.3, 500, 50)
    output = postprocess(output)
    plt.imshow(output)
    plt.show()

