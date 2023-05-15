import os
import torch
from torch import nn

from D2L.data.data import download_extract


class TokenEmbedding:
    """GloVe embedding"""
    def __init__(self, embedding_name, cache_dir):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name, cache_dir)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name, cache_dir):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = download_extract(embedding_name, cache_dir)
        # GloVe website：https://nlp.stanford.edu/projects/glove/
        # fastText website：https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r', encoding="utf8") as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # skip heading information
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


def knn(W, x, k):
    cos = torch.mv(W, x.reshape(-1,)) / (
            torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
            torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]


def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):
        print(f'{embed.idx_to_token[int(i)]}：cosine similarity={float(c):.3f}')


def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]


if __name__ == '__main__':
    glove_6b50d = TokenEmbedding('glove.6b.50d', '../../dataset/embeddings')
    print(len(glove_6b50d))

    print(glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367])

    print(get_similar_tokens('chip', 3, glove_6b50d))

    print(get_analogy('man', 'woman', 'son', glove_6b50d))