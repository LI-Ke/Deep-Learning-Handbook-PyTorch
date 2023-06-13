import os
import torch
from torch import nn

from Foundation.data.data import download_extract, load_array
from Foundation.models.cnn.lenet import evaluate_accuracy_gpu
from Foundation.models.computer_vision.augmentation import train_ch13
from Foundation.models.nlp.sentiment_analysis_biRNN import load_data_imdb, predict_sentiment
from Foundation.models.nlp.text_preprocessing import tokenize, Vocab
from Foundation.models.nlp.word_similarity_and_analogy import TokenEmbedding
from Foundation.models.rnn.machine_translation import truncate_pad
from Foundation.utils.accumulator import Accumulator
from Foundation.utils.eval import accuracy
from Foundation.utils.timer import Timer


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # this embedding layer is not trainable
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # there are not parameters in pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # create 1d conv layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # concat two embeddings
        # embedding shape（batch_size，vocab_size，embed_size）
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # put channel as the second dimension
        embeddings = embeddings.permute(0, 2, 1)
        # encoding shape （batch_size，num_channel，1）
        # remove last dimension and concat tensors
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


if __name__ == '__main__':
    batch_size = 64
    train_iter, test_iter, vocab = load_data_imdb('../../dataset/imdb', batch_size)
    print(len(vocab))

    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    devices = ['cuda']
    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

    def init_weights(m):
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    glove_embedding = TokenEmbedding('glove.6b.100d', '../../dataset/embeddings')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.requires_grad = False

    lr, num_epochs = 0.001, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    print(predict_sentiment(net, vocab, 'this movie is so great'))
    print(predict_sentiment(net, vocab, 'this movie is so bad'))