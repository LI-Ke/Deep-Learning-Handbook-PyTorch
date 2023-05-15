import math
import pandas as pd
import torch
from torch import nn

from D2L.models.attention.Bahdanau_attention import AttentionDecoder
from D2L.models.attention.multihead_attention import MultiHeadAttention
from D2L.models.attention.positional_encoding import PositionalEncoding
from D2L.models.attention.vis import show_heatmaps
from D2L.models.rnn.encoder_decoder import Encoder, EncoderDecoder
from D2L.models.rnn.machine_translation import load_data_nmt
from D2L.models.rnn.seq2seq import train_seq2seq, predict_seq2seq, bleu


class PositionWiseFFN(nn.Module):
    """Position-wise FFN"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """Residual connection and then layer normalization"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, key_size, query_size, value_size, num_hidden,
                 norm_shape, ffn_num_input, ffn_num_hidden, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hidden, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hidden, num_hidden)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    """Transformer encoder"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hidden, norm_shape, ffn_num_input, ffn_num_hidden,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        self.embedding = nn.Embedding(vocab_size, num_hidden)
        self.pos_encoding = PositionalEncoding(num_hidden, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hidden,
                                              norm_shape, ffn_num_input, ffn_num_hidden,
                                              num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # Since the positional encoding values are between -1 and 1
        # embeddings should multiply embedding_dim
        # then add the positional encoding
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hidden))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = \
                blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """the ith block in decoder"""
    def __init__(self, key_size, query_size, value_size, num_hidden,
                 norm_shape, ffn_num_input, ffn_num_hidden, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hidden, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hidden, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hidden, num_hidden)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # during training stage, all tokens of output sequence will be processed at the same time
        # thus state[2][self.i] will be initialized by None。
        # during inference stage，
        # Thus state[2][self.i] contains the output representation of the
        # ith decoder block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens shape:(batch_size,num_steps),
            # each row [1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # self attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # encoder-decoder attention
        # enc_outputs shape:(batch_size,num_steps,num_hidden)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hidden, norm_shape, ffn_num_input, ffn_num_hidden,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hidden
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hidden)
        self.pos_encoding = PositionalEncoding(num_hidden, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hidden,
                                              norm_shape, ffn_num_input, ffn_num_hidden,
                                              num_heads, dropout, i))
        self.dense = nn.Linear(num_hidden, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器⾃注意⼒权重
            self._attention_weights[0][i] = \
                blk.attention1.attention.attention_weights
            # “编码器－解码器”⾃注意⼒权重
            self._attention_weights[1][i] = \
                blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


if __name__ == '__main__':
    # add_norm = AddNorm([3, 4], 0.5)
    # add_norm.eval()
    # print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)
    #
    # X = torch.ones((2, 100, 24))
    # valid_lens = torch.tensor([3, 2])
    # encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    # encoder_blk.eval()
    # print(encoder_blk(X, valid_lens).shape)
    #
    # encoder = TransformerEncoder(
    #     200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    # encoder.eval()
    # print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)
    #
    # decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    # decoder_blk.eval()
    # X = torch.ones((2, 100, 24))
    # state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    # print(decoder_blk(X, state)[0].shape)

    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, 'cuda'
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    train_iter, src_vocab, tgt_vocab = load_data_nmt('../../dataset/machine_translation', batch_size, num_steps)
    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ', f'bleu {bleu(translation, fra, k=2):.3f}')

    enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads, -1, num_steps))
    print(enc_attention_weights.shape)

    show_heatmaps(enc_attention_weights.cpu(), xlabel='Key positions',
        ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
        figsize=(7, 3.5))

    dec_attention_weights_2d = [head[0].tolist()
                                for step in dec_attention_weight_seq
                                for attn in step for blk in attn for head in blk]
    dec_attention_weights_filled = torch.tensor(
        pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
    dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
    dec_self_attention_weights, dec_inter_attention_weights = \
        dec_attention_weights.permute(1, 2, 3, 0, 4)
    print(dec_self_attention_weights.shape, dec_inter_attention_weights.shape)

    show_heatmaps(
        dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
        xlabel='Key positions', ylabel='Query positions',
        titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

    show_heatmaps(
        dec_inter_attention_weights, xlabel='Key positions',
        ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
        figsize=(7, 3.5))
