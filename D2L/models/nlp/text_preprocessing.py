import collections
import re

from D2L.data.data import download


def read_time_machine(cache_dir):
    """ load time machine dataset to list"""
    with open(download('time_machine', cache_dir=cache_dir), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """decompose text to words or characters"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Error：unknown type：' + token)


def count_corpus(tokens):
    """count token frequency"""
    # tokens are 1D or 2D lists
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # flatten tokens list
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """vocabulary"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # rank by frequency
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # set unknown token index to 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self): # unknown token index is 0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def load_corpus_time_machine(cache_dir, max_tokens=-1):
    """return corpus and vocabulary of the time machine dataset"""
    lines = read_time_machine(cache_dir)
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == '__main__':
    lines = read_time_machine('../../dataset/time_machine')
    print(f'# number of lines: {len(lines)}')
    print(lines[0])
    print(lines[10])

    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])

    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])

    corpus, vocab = load_corpus_time_machine('../../dataset/time_machine')
    len(corpus), len(vocab)