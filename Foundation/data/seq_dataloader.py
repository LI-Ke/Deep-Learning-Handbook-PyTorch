import random

import torch

from D2L.models.nlp.text_preprocessing import load_corpus_time_machine


def seq_data_iter_random(corpus, batch_size, num_steps):
    """using random sampling to generate a mini batch"""
    # random number range includes num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # minus 1 to consider the label
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        # return sequence of length num_steps starting from pos
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """using sequential sampling to generate a mini batch"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """iterator loading sequential data"""
    def __init__(self, batch_size, num_steps, cache_dir, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(cache_dir, max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, cache_dir,
                           use_random_iter=False, max_tokens=10000):
    """return time machine dataset iterator and vocabulary"""
    data_iter = SeqDataLoader(batch_size, num_steps, cache_dir, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


if __name__ == '__main__':
    my_seq = list(range(35))
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

    data_iter, vocab = load_data_time_machine(32, num_steps=35, cache_dir='../dataset/time_machine')