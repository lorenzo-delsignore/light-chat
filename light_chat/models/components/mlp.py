import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, emb_in=2, emb_hidden=100, context=3):
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.emb_in = emb_in
        self.context = context
        self.enc = nn.Parameter(torch.randn((27, emb_in), generator=g))
        self.first_W = nn.Parameter(torch.randn((emb_in * context, emb_hidden), generator=g))
        self.first_bias = nn.Parameter(torch.randn(emb_hidden, generator=g))
        self.second_W = nn.Parameter(torch.randn((emb_hidden, 27), generator=g))
        self.second_bias = nn.Parameter(torch.randn(27, generator=g))

    def forward(self, x):
        ngram_enc = self.enc[x["ngram"]]
        hidden = torch.tanh((ngram_enc.view(-1, self.emb_in * self.context) @ self.first_W) + self.first_bias)
        logits = (hidden @ self.second_W) + self.second_bias
        return logits
