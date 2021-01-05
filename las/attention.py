import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, key, value):
        key = key.transpose(1, 2).contiguous()  # key shape : (batch, 2 * hidden(256), seq_len)
        attention_distribution = F.softmax(torch.einsum('bij,bjk->bik', query, key), dim=2)  # (batch, seq_len, seq_len)
        context_vector = torch.mm(attention_distribution, value)  # context shape : (batch, seq_len, 2 * hidden(256))

        return context_vector
