import torch
from torch import nn
from torch.nn import functional as F

from Foundation.data.seq_dataloader import load_data_time_machine
from Foundation.models.rnn.custom_rnn import predict, train


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # if rnn is bidirectional, num_directions is 2，otherwise 1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # the input shape of FC layer should be (num_steps * batch, number_hidden)
        # the output shape of FC layer should be (num_steps * batch, vocab_size)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU uses tensor as hidden state
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM uses tuple as hidden state
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps, '../../dataset/time_machine')

    num_hidden = 256
    rnn_layer = nn.RNN(len(vocab), num_hidden)

    state = torch.zeros((1, batch_size, num_hidden))
    print(state.shape)

    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, state_new.shape)

    device = 'cuda'
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    predict('time traveller', 10, net, vocab, device)

    num_epochs, lr = 500, 1
    train(net, train_iter, vocab, lr, num_epochs, device)