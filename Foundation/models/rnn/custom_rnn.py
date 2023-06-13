import math
import torch
from torch import nn
from torch.nn import functional as F

from Foundation.data.seq_dataloader import load_data_time_machine
from Foundation.models.optim import sgd
from Foundation.utils.accumulator import Accumulator
from Foundation.utils.timer import Timer


def get_params(vocab_size, num_hidden, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # hidden layer parameters
    W_xh = normal((num_inputs, num_hidden))
    W_hh = normal((num_hidden, num_hidden))
    b_h = torch.zeros(num_hidden, device=device)
    # output layer parameters
    W_hq = normal((num_hidden, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hidden, device):
    return (torch.zeros((batch_size, num_hidden), device=device), )


def rnn(inputs, state, params):
    # inputs shape：(num_steps，batch_size，vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X's shape：(batch_size，vocab_size)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    """implement RNN from scratch"""
    def __init__(self, vocab_size, num_hidden, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hidden = vocab_size, num_hidden
        self.params = get_params(vocab_size, num_hidden, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hidden, device)


def predict(prefix, num_preds, net, vocab, device):
    """generate new characters after the prefix"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]: # warm_up
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds): # predict lengths of num_preds
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch(net, train_iter, loss, optimizer, device, use_random_iter):
    """train one epoch"""
    state, timer = None, Timer()
    metric = Accumulator(2) # total loss, token number
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # initialize state when first iteration or use random iterator
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # for GRU
                state.detach_()
            else:
                # for LSTM
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            optimizer.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # mean function has been called
            optimizer(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """train model"""
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        optimizer = torch.optim.SGD(net.parameters(), lr)
    else:
        optimizer = lambda batch_size: sgd(net.params, lr, batch_size)
    new_predict = lambda prefix: predict(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(
            net, train_iter, loss, optimizer, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(new_predict('time traveller'))
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec {str(device)}')
    print(new_predict('time traveller'))
    print(new_predict('traveller'))


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps, cache_dir='../../dataset/time_machine')
    print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

    X = torch.arange(10).reshape((2, 5))
    print(X)
    print(X.T)
    print(F.one_hot(X.T, 28).shape)

    num_hidden = 512
    net = RNNModelScratch(len(vocab), num_hidden, 'cuda', get_params,
                          init_rnn_state, rnn)
    # state = net.begin_state(X.shape[0], 'cuda')
    # Y, new_state = net(X.to('cuda'), state)
    # print(Y.shape), print(len(new_state)), print(new_state[0].shape)
    #
    # print(predict('time traveller ', 10, net, vocab, 'cuda'))

    num_epochs, lr = 500, 1
    train(net, train_iter, vocab, lr, num_epochs, 'cuda')