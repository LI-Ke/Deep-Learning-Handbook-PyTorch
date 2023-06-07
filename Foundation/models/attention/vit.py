import torch
from torch import nn

from D2L.data.data import load_data_fashion_mnist
from D2L.models.attention.multihead_attention import MultiHeadAttention
from D2L.models.computer_vision.augmentation import train_ch13


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)


class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))


class ViTBlock(nn.Module):
    def __init__(self, num_hidden, norm_shape, mlp_num_hidden, num_heads,
                 dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hidden, num_hidden, num_hidden,
                                            num_hidden, num_heads,
                                            dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hidden, num_hidden, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))


class ViT(nn.Module):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hidden, mlp_num_hidden,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hidden)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hidden))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hidden))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hidden, num_hidden, mlp_num_hidden,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hidden),
                                  nn.Linear(num_hidden, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])


if __name__ == '__main__':
    # img_size, patch_size, num_hidden, batch_size = 96, 16, 512, 4
    # patch_emb = PatchEmbedding(img_size, patch_size, num_hidden)
    # X = torch.zeros(batch_size, 3, img_size, img_size)
    # print(patch_emb(X).shape)
    #
    # X = torch.ones((2, 100, 24))
    # encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
    # encoder_blk.eval()
    # print(encoder_blk(X).shape)

    img_size, patch_size = 96, 16
    num_hidden, mlp_num_hidden, num_heads, num_blks = 512, 2048, 8, 2
    emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
    net = ViT(img_size, patch_size, num_hidden, mlp_num_hidden, num_heads,
                num_blks, emb_dropout, blk_dropout, lr)

    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=(img_size, img_size))
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, optimizer, num_epochs, ['cuda'])
