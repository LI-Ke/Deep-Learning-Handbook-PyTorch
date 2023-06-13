import torch
from torch import nn

from Foundation.data.seq_dataloader import load_data_time_machine
from Foundation.models.rnn.custom_rnn import RNNModelScratch, train
from Foundation.models.rnn.train_rnn_concise import RNNModel


def get_lstm_params(vocab_size, num_hidden, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hidden)),
                normal((num_hidden, num_hidden)),
                torch.zeros(num_hidden, device=device))

    W_xi, W_hi, b_i = three() # input gate parameters
    W_xf, W_hf, b_f = three() # forget gate parameters
    W_xo, W_ho, b_o = three() # output gate parameters
    W_xc, W_hc, b_c = three() # candidate memory cell parameters
    # output layer parameters
    W_hq = normal((num_hidden, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hidden, device):
    return (torch.zeros((batch_size, num_hidden), device=device),
            torch.zeros((batch_size, num_hidden), device=device))


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
    W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps, '../../dataset/time_machine')
    # lstm scratch
    vocab_size, num_hiddens, device = len(vocab), 256, 'cuda'
    num_epochs, lr = 500, 1
    # model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
    #                             init_lstm_state, lstm)
    # train(model, train_iter, vocab, lr, num_epochs, device)

    # lstm concise
    num_inputs = vocab_size
    # lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    # model = RNNModel(lstm_layer, len(vocab))
    # model = model.to(device)
    # train(model, train_iter, vocab, lr, num_epochs, device)

    # deep lstm
    num_layers, lr = 2, 2
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    train(model, train_iter, vocab, lr * 1.0, num_epochs, device)
