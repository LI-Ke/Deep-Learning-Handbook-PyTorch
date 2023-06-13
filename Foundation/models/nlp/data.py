import math
import os
import random
import torch
from torch import nn

from Foundation.data.data import download_extract, get_dataloader_workers
from Foundation.models.nlp.text_preprocessing import count_corpus, Vocab


def read_ptb(cache_dir):
    """load PTB data to list"""
    data_dir = download_extract('ptb', cache_dir)
    # Read the training set.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


def subsample(sentences, vocab):
    """Subsampling highly frequent words"""
    # exclude '<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())
    # if keep the token during the subsampling, then return true
    def keep(token):
        return (random.uniform(0, 1) <
                math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)


def compare_counts(token, sentences, subsampled):
    return (f'"{token}" numbers：'
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')


def get_centers_and_contexts(corpus, max_window_size):
    """get center words and context words of skip-gram"""
    centers, contexts = [], []
    for line in corpus:
        # at least 2 words to generate 'center-context' pair
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)): # center i of context window
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # remove center from context
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


class RandomGenerator:
    """Random sampling from {1,...,n} samples"""
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # store K random samples
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_negatives(all_contexts, vocab, counter, K):
    """get noisy words in negative samples"""
    # index 1, 2, ...(index 0 is unknown word)
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # noisy word can not be in context
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    """mini batch with negative samples for skip-gram"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += \
            [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
        contexts_negatives), torch.tensor(masks), torch.tensor(labels))


class PTBDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index],
                self.negatives[index])

    def __len__(self):
        return len(self.centers)


def load_data_ptb(cache_dir, batch_size, max_window_size, num_noise_words):
    num_workers = get_dataloader_workers()
    sentences = read_ptb(cache_dir)
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab



if __name__ == '__main__':
    # sentences = read_ptb('../../dataset/PTB')
    # print(f'# sentences数: {len(sentences)}')
    # vocab = Vocab(sentences, min_freq=10)
    # print(f'vocab size: {len(vocab)}')
    # subsampled, counter = subsample(sentences, vocab)
    # corpus = [vocab[line] for line in subsampled]

    # tiny_dataset = [list(range(7)), list(range(7, 10))]
    # print('dataset', tiny_dataset)
    # for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    #     print('center', center, 'contexts', context)

    # all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
    # print(f'# “center-context pair” numbers: {sum([len(contexts) for contexts in all_contexts])}')

    # generator = RandomGenerator([2, 3, 4])
    # [generator.draw() for _ in range(10)]

    # all_negatives = get_negatives(all_contexts, vocab, counter, 5)

    # x_1 = (1, [2, 2], [3, 3, 3, 3])
    # x_2 = (1, [2, 2, 2], [3, 3])
    # batch = batchify((x_1, x_2))
    names = ['centers', 'contexts_negatives', 'masks', 'labels']
    # for name, data in zip(names, batch):
    #     print(name, '=', data)

    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = load_data_ptb('../../dataset/PTB', batch_size,
                                     max_window_size, num_noise_words)
    for batch in data_iter:
        for name, data in zip(names, batch):
            print(name, 'shape:', data.shape)
        break

