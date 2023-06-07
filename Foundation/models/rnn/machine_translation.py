import os

import torch

from D2L.data.data import download_extract, load_array
from D2L.models.nlp.text_preprocessing import Vocab


def read_data_nmt(cache_dir):
    """load eng-fra dataset"""
    data_dir = download_extract('fra-eng', cache_dir)
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text):
    """preprocess eng-fra dataset"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # use space to replace non-breaking space
    # set characters to lower case
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """tokenize eng-fra dataset"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps] # truncate
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(cache_dir, batch_size, num_steps, num_examples=600):
    """return dataset's iterator and vocabulary"""
    text = preprocess_nmt(read_data_nmt(cache_dir))
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab



if __name__ == '__main__':
    raw_text = read_data_nmt('../../dataset/machine_translation')
    print(raw_text[:75])

    text = preprocess_nmt(raw_text)
    print(text[:80])

    source, target = tokenize_nmt(text)
    print(source[:6], target[:6])

    src_vocab = Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(len(src_vocab))

    print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

    train_iter, src_vocab, tgt_vocab = load_data_nmt('../../dataset/machine_translation', batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X valid len:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y valid len:', Y_valid_len)
        break