import math
import torch
from torch import nn

from D2L.models.attention.vis import show_heatmaps
from D2L.models.rnn.seq2seq import sequence_mask


def masked_softmax(X, valid_lens):
    """softmax operation by masking elements along the last axis"""
    # X:3D tensor，valid_lens:1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Masked elements on the last axis are replaced with a very small
        # negative value, so that their softmax output is 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """Additive attention"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # after dimension expansion，
        # queries shape：(batch_size，num_queries，1，num_hidden)
        # key shape：(batch_size，1，k-v pairs，num_hidden)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v has only one output, thus remove the last dimension
        # scores shape：(batch_size，num_queries, k-v pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values shape：(batch_size，k-v pairs，value dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """Scaled dot product attention"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries shape：(batch_size，num_queries，d)
    # keys shape：(batch_size，k-v pairs，d)
    # values shape：(batch_size，k-v pairs，value dimension)
    # valid_lens shape:(batch_size，) or (batch_size，num_queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == '__main__':
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    # attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
    #                               dropout=0.1)
    # attention.eval()
    # print(attention(queries, keys, values, valid_lens))
    # show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
    #               xlabel='Keys', ylabel='Queries')

    queries = torch.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))
    show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')