import torch
from torch import nn
from torch.nn import functional as F

from D2L.models.computer_vision.augmentation import train_ch13
from D2L.models.nlp.SNLI_dataset import load_data_snli
from D2L.models.nlp.word_similarity_and_analogy import TokenEmbedding


def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)


class Attend(nn.Module):
    def __init__(self, num_inputs, num_hidden, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hidden, flatten=False)

    def forward(self, A, B):
        # A/B shape：（batch_size，A/B num_tokens，embed_size）
        # f_A/f_B shape：（batch_size，A/B num_tokens，num_hidden）
        f_A = self.f(A)
        f_B = self.f(B)
        # e shape：（batch_size，a num_tokens，b num_tokens）
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # beta shape：（batch_size，a num_tokens，embed_size），
        # align sequence B to sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # alpha shape：（batch_size，b num_tokens，embed_size），
        # align sequence A to sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha


class Compare(nn.Module):
    def __init__(self, num_inputs, num_hidden, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hidden, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B


class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hidden, flatten=True)
        self.linear = nn.Linear(num_hidden, num_outputs)

    def forward(self, V_A, V_B):
        # calculate sum for each compare vector
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat


class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hidden,
                 num_inputs_attend=100, num_inputs_compare=200,
                 num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hidden)
        self.compare = Compare(num_inputs_compare, num_hidden)
        # 3 possible outputs
        self.aggregate = Aggregate(num_inputs_agg, num_hidden, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat


def predict_snli(net, vocab, premise, hypothesis):
    """inference the logic between premise and hypothesis"""
    net.eval()
    premise = torch.tensor(vocab[premise], device='cuda')
    hypothesis = torch.tensor(vocab[hypothesis], device='cuda')
    label = torch.argmax(net([premise.reshape((1, -1)),
                              hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
        else 'neutral'


if __name__ == '__main__':
    batch_size, num_steps = 256, 50
    train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)
    print(len(vocab))

    embed_size, num_hidden, devices = 100, 200, ['cuda']
    net = DecomposableAttention(vocab, embed_size, num_hidden)
    glove_embedding = TokenEmbedding('glove.6b.100d', '../../dataset/embeddings')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)

    lr, num_epochs = 0.001, 4
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices)
    print(predict_snli(net, vocab,
                       ['he', 'is', 'good', '.'],
                       ['he', 'is', 'bad', '.']))

