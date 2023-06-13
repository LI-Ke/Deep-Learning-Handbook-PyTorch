import os
import torch
from torch import nn

from Foundation.data.data import download_extract, load_array
from Foundation.models.cnn.lenet import evaluate_accuracy_gpu
from Foundation.models.computer_vision.augmentation import train_ch13
from Foundation.models.nlp.text_preprocessing import tokenize, Vocab
from Foundation.models.nlp.word_similarity_and_analogy import TokenEmbedding
from Foundation.models.rnn.machine_translation import truncate_pad
from Foundation.utils.accumulator import Accumulator
from Foundation.utils.eval import accuracy
from Foundation.utils.timer import Timer


def read_imdb(data_dir, is_train):
    """read IMDb review dataset and labels"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


def load_data_imdb(cache_dir, batch_size, num_steps=500):
    """get data iterator and IMDB vocabulary"""
    data_dir = download_extract('aclImdb', cache_dir, 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = tokenize(train_data[0], token='word')
    test_tokens = tokenize(test_data[0], token='word')
    vocab = Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = load_array((train_features, torch.tensor(train_data[1])),
                            batch_size)
    test_iter = load_array((test_features, torch.tensor(test_data[1])),
                           batch_size, is_train=False)
    return train_iter, test_iter, vocab


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #  set bidirectional as True
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs shape（batch_size，num_steps）
        # first dimension of LSTM input is num_step
        # thus inputs should be transposed before get embedding
        # output shape（num_step，batch_size，embed_size）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # get hidden states of different time step，
        # outputs shape（num_step，batch_size，2*num_hidden）
        outputs, _ = self.encoder(embeddings)
        # concatenate starting and ending hidden states as input of FC layer
        #（batch_size，4*num_hidden）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


def predict_sentiment(net, vocab, sequence):
    """predict sentiment of text"""
    sequence = torch.tensor(vocab[sequence.split()], device=['cuda'])
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


if __name__ == '__main__':
    batch_size = 64
    train_iter, test_iter, vocab = load_data_imdb('../../dataset/imdb', batch_size)
    print(len(vocab))

    embed_size, num_hiddens, num_layers = 100, 100, 2
    devices = ['cuda']
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(init_weights)

    glove_embedding = TokenEmbedding('glove.6b.100d', '../../dataset/embeddings')
    embeds = glove_embedding[vocab.idx_to_token]
    print(embeds.shape)

    # freeze parameters
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False
    lr, num_epochs = 0.01, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    print(predict_sentiment(net, vocab, 'this movie is so great'))
    print(predict_sentiment(net, vocab, 'this movie is so bad'))