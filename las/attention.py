import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor
                ) -> Tensor:
        key = key.transpose(1, 2)  # key shape : (batch, hidden(256) << 1, seq_len)
        attention_distribution = F.softmax(torch.bmm(query, key), dim=2)  # (batch, seq_len, seq_len)
        context_vector = torch.mm(attention_distribution, value)  # context shape : (batch, seq_len, hidden(256) << 1)

        return context_vector
