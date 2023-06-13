import os
import re
import torch
from torch import nn

from Foundation.data.data import download_extract, get_dataloader_workers
from Foundation.models.nlp.text_preprocessing import tokenize, Vocab
from Foundation.models.rnn.machine_translation import truncate_pad


def read_snli(data_dir, is_train):
    """extract SNLI data to premises, hypotheses, labels"""

    def extract_text(s):
        # remove useless information
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # replace multiple space with one single space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
    if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] \
                  in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


class SNLIDataset(torch.utils.data.Dataset):
    """load SNLI dataset"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0])
        all_hypothesis_tokens = tokenize(dataset[1])
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens +
                               all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
            for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_data_snli(batch_size, num_steps=50):
    """download SNLI dataset and get data iterator and vocabulary"""
    num_workers = get_dataloader_workers()
    data_dir = '../../dataset/NLI/snli_1.0'
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab


if __name__ == '__main__':
    data_dir = '../../dataset/NLI/snli_1.0'
    train_data = read_snli(data_dir, is_train=True)
    for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
        print('前提：', x0)
        print('假设：', x1)
        print('标签：', y)

    test_data = read_snli(data_dir, is_train=False)
    for data in [train_data, test_data]:
        print([[row for row in data[2]].count(i) for i in range(3)])

    train_iter, test_iter, vocab = load_data_snli(128, 50)
    print(len(vocab))

    for X, Y in train_iter:
        print(X[0].shape)
        print(X[1].shape)
        print(Y.shape)
        break