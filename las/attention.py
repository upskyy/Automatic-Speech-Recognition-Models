import torch
from torch import Tensor
from typing import Tuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 key_dim: int
                 ) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_key_dim = np.sqrt(key_dim)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor
                ) -> Tuple[Tensor, Tensor]:
        key = key.transpose(1, 2)  # transpose 이후 key shape : (batch, enc_D << 1, enc_T)
        attn_distribution = torch.bmm(query, key) / self.sqrt_key_dim
        attn_distribution = F.softmax(attn_distribution, dim=2)  # (batch, dec_T, enc_T)
        context = torch.bmm(attn_distribution, value)  # context shape : (batch, dec_T, enc_D << 1)

        return context, attn_distribution


class LocationAwareAttention(nn.Module):
    def __init__(self,
                 decoder_dim: int,
                 attn_dim: int,
                 smoothing: bool
                 ) -> None:
        super(LocationAwareAttention, self).__init__()
        self.decoder_dim = decoder_dim
        self.attn_dim = attn_dim
        self.conv = nn.Conv1d(in_channels=1, out_channels=attn_dim, kernel_size=3, padding=1)
        self.query_linear = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.key_linear = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim))
        self.fc = nn.Linear(attn_dim, 1, bias=True)
        self.smoothing = smoothing

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                last_attn_distribution: Tensor
                ) -> Tuple[Tensor, Tensor]:

        if last_attn_distribution is None:
            last_attn_distribution = key.new_zeros(key.size(0), key.size(1))  # (B, enc_T)

        last_attn_distribution = last_attn_distribution.unsqueeze(1)
        last_attn_distribution = self.conv(last_attn_distribution).transpose(1, 2)  # (B, enc_T, attn_dim)

        attn_distribution = self.fc(torch.tanh(self.query_linear(query)
                                               + self.key_linear(key)
                                               + last_attn_distribution
                                               + self.bias)).squeeze(-1)

        if self.smoothing:
            attn_distribution = torch.sigmoid(attn_distribution)
            attn_distribution = torch.div(attn_distribution, attn_distribution.sum(dim=1).unsqueeze(1))

        else:
            attn_distribution = F.softmax(attn_distribution, dim=-1)  # (B, enc_T)

        context = torch.bmm(attn_distribution.unsqueeze(1), value)  # (B, 1, enc_D << 1)

        return context, attn_distribution
