import torch
import torch.nn as nn
from einops import repeat, rearrange
from postional import PositionalEncoding
import pytorch_lightning as pl


class linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, seq, seq1, seq2):
        a = rearrange(seq, 'a b -> a b 1')
        b = rearrange(seq1, 'a b -> a b 1')
        c = rearrange(seq2, 'a b -> a b 1')
        seq = torch.cat((a,b,c), dim=2)
        out = self.linear(seq)
        out = rearrange(out, 'a b 1 -> a b')
        return out

class skeleTransLayer(nn.Module):
    def __init__(self, num_classes, d_model, nhead, seq_len, nlayers, mask= True):
        super(skeleTransLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.mask = mask
        self.hidden = 8

        encoder_layer = nn.TransformerEncoderLayer(2*self.d_model, nhead, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len+1, 2*self.d_model)) #here
        self.pos_encoding = PositionalEncoding(2*self.d_model, max_len=seq_len)
        self.linear = nn.Linear(2*self.d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 2*self.d_model))
        self.linear_pre = nn.Linear(2*self.d_model, 512)

    def forward(self, seq, mask):
        b, f, v, c = seq.shape
        out = seq.view(b, f, v*c)
        k, n, m = out.shape
        cls_tokens = repeat(self.cls_token, '1 1 m -> k 1 m', k = k )
        h = torch.cat((cls_tokens, out), dim=1)
        pos_embeddings = repeat(self.pos_embedding, '1 n m -> k n m', k = k)
        h = pos_embeddings[:, :n+1, :] + h
        if self.mask == True:
            h = self.encoder(src =  h, src_key_padding_mask = mask)
        else:
            h = self.encoder(src =  h)
        h = h.mean(dim=1)
        res = self.linear(h)
        return res
