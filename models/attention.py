# MIT License
#
# Copyright (c) 2021 Sangchun Ha
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

import torch
from torch import Tensor
from typing import Tuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all keys, divide each by sqrt(key_dim),
    and apply a softmax function to obtain the weights on the values

    Args: key_dim
        key_dim (int): dimension of key

    Inputs: query, key, value
        - **query** (batch, q_len, hidden_dim): tensor containing projection vector for decoder
        - **key** (batch, k_len, hidden_dim): tensor containing projection vector for encoder
        - **value** (batch, v_len, hidden_dim): value and key are the same in the las structure

    Returns: context, attn_distribution
        - **context** (batch, dec_len, dec_hidden): tensor containing the context vector from attention mechanism
        - **attn_distribution** (batch, dec_len, enc_len): tensor containing the attention from the encoder outputs
    """
    def __init__(
            self,
            key_dim: int,
    ) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_key_dim = np.sqrt(key_dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        key = key.transpose(-2, -1)  # transpose 이후 key shape : (batch, enc_D << 1, enc_T)
        attn_distribution = torch.bmm(query, key) / self.sqrt_key_dim
        attn_distribution = F.softmax(attn_distribution, dim=-1)  # (batch, dec_T, enc_T)
        context = torch.bmm(attn_distribution, value)  # context shape : (batch, dec_T, enc_D << 1)

        return context, attn_distribution


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            key_dim: int,
            num_head: int,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.head_dim = key_dim // num_head
        self.scaled_dot = ScaledDotProductAttention(self.head_dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch = query.size(0)

        query = query.view(batch, -1, self.num_head, self.head_dim)
        key = key.view(batch, -1, self.num_head, self.head_dim)
        value = value.view(batch, -1, self.num_head, self.head_dim)

        query = query.permute(0, 2, 1, 3).contiguous().view(batch * self.num_head, -1, self.head_dim)
        key = key.permute(0, 2, 1, 3).contiguous().view(batch * self.num_head, -1, self.head_dim)
        value = value.permute(0, 2, 1, 3).contiguous().view(batch * self.num_head, -1, self.head_dim)

        context, attn_distribution = self.scaled_dot(query, key, value)

        context = context.view(batch, self.num_head, -1, self.head_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch, -1, self.num_head * self.head_dim)  # (B, T, D)

        return context, attn_distribution


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.

    Args:
        decoder_dim (int): dimension of decoder
        attn_dim (int): dimension of attention
        smoothing (bool): flag indication smoothing or not.

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **key** (batch, k_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **value** (batch, v_len, hidden_dim): value and key are the same in the las structure
        - **last_attn_distribution** (batch_size, k_len): tensor containing previous timestep`s attention weight

    Returns: context, attn_distribution
        - **context** (batch, output_len, hidden_dim): tensor containing the feature from encoder outputs
        - **attn_distribution** (batch, k_len): tensor containing the attention weight from the encoder outputs.
    """
    def __init__(
            self,
            decoder_dim: int,
            attn_dim: int,
            smoothing: bool,
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

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            last_attn_distribution: Tensor,
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
