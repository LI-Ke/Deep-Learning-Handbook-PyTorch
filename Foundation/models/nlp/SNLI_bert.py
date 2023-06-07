import json
import multiprocessing
import os
import torch
from torch import nn

from D2L.data.data import download_extract, get_dataloader_workers
from D2L.models.computer_vision.augmentation import train_ch13
from D2L.models.nlp.SNLI_dataset import read_snli
from D2L.models.nlp.bert import BERTModel, get_tokens_and_segments
from D2L.models.nlp.text_preprocessing import Vocab, tokenize


def load_pretrained_model(pretrained_model, cache_dir, num_hidden,
                          ffn_num_hidden, num_heads, num_layers, dropout,
                          max_len, devices):
    data_dir = download_extract(pretrained_model, cache_dir)
    # define empty vocabulary
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = BERTModel(len(vocab), num_hidden, norm_shape=[256],
                     ffn_num_input=256, ffn_num_hidden=ffn_num_hidden,
                     num_heads=num_heads, num_layers=num_layers,
                     dropout=dropout, max_len=max_len,
                     key_size=256, query_size=256, value_size=256,
                     hid_in_features=256, mlm_in_features=256, nsp_in_features=256)
    # load pretrained bert parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))
    return bert, vocab


class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]

        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments, self.valid_lens) = \
            self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # use 4 threading
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                    * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # keep '<CLS>'、'<SEP>' and '<SEP>'positions for input of bert
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)


class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))


if __name__ == '__main__':
    devices = ['cuda']
    bert, vocab = load_pretrained_model(
        'bert.small', '../../dataset/pretrain_bert', num_hidden=256, ffn_num_hidden=512,
        num_heads=4, num_layers=2, dropout=0.1, max_len=512, devices=devices)

    batch_size, max_len, num_workers = 512, 128, get_dataloader_workers()
    data_dir = '../../dataset/NLI/snli_1.0'
    train_set = SNLIBERTDataset(read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(read_snli(data_dir, False), max_len, vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            num_workers=num_workers)
    net = BERTClassifier(bert)

    lr, num_epochs = 1e-4, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)