import torch
from torch import nn
from torch.nn import functional as F

from Foundation.data.data import load_data_fashion_mnist
from Foundation.models.cnn.lenet import evaluate_accuracy_gpu
from Foundation.models.optim import sgd
from Foundation.utils.timer import Timer


# definition of model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat


def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params


def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)


def split_batch(X, y, devices):
    """split X and y to different devices"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))


def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # calculate loss on each GPU respectively
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
            X_shards, y_shards, device_params)]
    for l in ls: # BP on each GPU
        l.backward()
    # aggregate gradients and broadcast the sum to each GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce(
                [device_params[c][i].grad for c in range(len(devices))])
    # update parameters on each GPU
    for param in device_params:
        sgd(param, lr, X.shape[0]) # use the entire size of the mini-batch


def train(num_gpus, batch_size, lr):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = ['cuda' for i in range(num_gpus)]
    # copy parameters to all GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    timer = Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # process single batch on multi-GPUs
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
    # evaluate the model on GPU0
    print(epoch + 1, (evaluate_accuracy_gpu(
        lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))


if __name__ == '__main__':
    # parameter initialization
    scale = 0.01
    W1 = torch.randn(size=(20, 1, 3, 3)) * scale
    b1 = torch.zeros(20)
    W2 = torch.randn(size=(50, 20, 5, 5)) * scale
    b2 = torch.zeros(50)
    W3 = torch.randn(size=(800, 128)) * scale
    b3 = torch.zeros(128)
    W4 = torch.randn(size=(128, 10)) * scale
    b4 = torch.zeros(10)
    params = [W1, b1, W2, b2, W3, b3, W4, b4]

    loss = nn.CrossEntropyLoss(reduction='none')
    train(num_gpus=1, batch_size=256, lr=0.2)

