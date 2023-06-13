import torch
from torch import nn

from Foundation.data.seq_dataloader import load_data_time_machine
from Foundation.models.rnn.custom_rnn import RNNModelScratch, train
from Foundation.models.rnn.train_rnn_concise import RNNModel


def get_params(vocab_size, num_hidden, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hidden)),
                normal((num_hidden, num_hidden)),
                torch.zeros(num_hidden, device=device))

    W_xz, W_hz, b_z = three() # update gate parameters
    W_xr, W_hr, b_r = three() # reset gate parameters
    W_xh, W_hh, b_h = three() # candidate hidden state parameters
    # output layer parameters
    W_hq = normal((num_hidden, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hidden, device):
    return (torch.zeros((batch_size, num_hidden), device=device), )


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps, '../../dataset/time_machine')
    # gru from scratch
    vocab_size, num_hiddens, device = len(vocab), 256, 'cuda'
    num_epochs, lr = 500, 1
    # model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
    #                             init_gru_state, gru)
    # train(model, train_iter, vocab, lr, num_epochs, device)

    # gru concise
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    train(model, train_iter, vocab, lr, num_epochs, device)