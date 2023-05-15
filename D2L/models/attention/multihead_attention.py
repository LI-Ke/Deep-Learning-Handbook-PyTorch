import math
import torch
from torch import nn

from D2L.models.attention.attention_score import DotProductAttention


def transpose_qkv(X, num_heads):
    """reshape in order to compute the multihead-attention in parallel"""
    # input X shape: (batch_size，queries_num or k-v pairs，num_hidden)
    # output X shape: (batch_size，queries_num or k-v pairs，num_heads，
    # num_hidden/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # output X shape:(batch_size，num_heads，queries_num or k-v pairs,
    # num_hidden/num_heads)
    X = X.permute(0, 2, 1, 3)
    # output shape:(batch_size*num_heads,queries_num or k-v pairs,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """Multihead attention"""
    def __init__(self, key_size, query_size, value_size, num_hidden,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hidden, bias=bias)
        self.W_k = nn.Linear(key_size, num_hidden, bias=bias)
        self.W_v = nn.Linear(value_size, num_hidden, bias=bias)
        self.W_o = nn.Linear(num_hidden, num_hidden, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values shape:
        # (batch_size，queries or k-v paris，num_hiddens)
        # valid_lens: (batch_size，) or (batch_size, queries)
        # after the transpose，output queries，keys，values shape:
        # (batch_size*num_heads，queries_num or k-v pairs，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            # num_heads
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        # output shape:(batch_size*num_heads, queries, num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # output_concat shape:(batch_size，queries，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


if __name__ == '__main__':
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
    attention.eval()

    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)

    # self-attention
    print(attention(X, X, X, valid_lens).shape)
