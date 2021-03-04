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

from typing import Tuple
from torch import Tensor
from models.las.encoder import Encoder
from models.las.decoder import Decoder
import torch.nn as nn


class ListenAttendSpell(nn.Module):
    """
    Listen, Attend and Spell structure proposed in https://arxiv.org/abs/1508.01211

    Args:
        encoder (Encoder): Encoder class instance
        decoder (Decoder): Decoder class instance

    Inputs: encoder_inputs, encoder_inputs_lens, decoder_inputs, teacher_forcing_ratio
        - **encoder_inputs**: Parsed audio of batch size number
        - **encoder_inputs_lens**: Tensor of sequence lengths
        - **decoder_inputs**: Used as ground truth when using teacher_forcing
        - **teacher_forcing_ratio**: The probability that teacher forcing will be used.

    Returns: encoder_output_prob, encoder_output_lens, decoder_output_prob
        - **encoder_output_prob**: Tensor containing log probability for ctc loss
        - **encoder_output_lens**: Tensor of sequence lengths produced by convolution
        - **decoder_output_prob**: Tensor expressing the log probability value of each word
    """
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
    ) -> None:
        super(ListenAttendSpell, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            encoder_inputs: Tensor,
            encoder_inputs_lens: Tensor,
            decoder_inputs: Tensor,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagate a `inputs` for encoder training.

        Args:
            encoder_inputs (torch.FloatTensor): A input sequence passed to encoder. ``(batch, seq_length, dimension)``
            encoder_inputs_lens (torch.LongTensor): The length of input tensor. ``(batch)``
            decoder_inputs (torch.LongTensor): A input sequence passed to decoder. ``(batch, seq_length)``
            teacher_forcing_ratio (float): Ratio of teacher forcing

        Returns:
            **encoder_output_prob** (Tensor): ``(batch, seq_length, dimension)``
            **encoder_output_lens** (Tensor): ``(batch, seq_length, num_vocabs)``
            if use_joint_ctc_attention is False, return None.
            **decoder_output_prob** (Tensor): ``(batch, seq_length, num_vocabs)``
        """
        self.encoder.rnn.flatten_parameters()
        encoder_output, encoder_output_prob, encoder_output_lens = self.encoder(encoder_inputs, encoder_inputs_lens)

        self.decoder.rnn.flatten_parameters()
        decoder_output_prob = self.decoder(decoder_inputs, encoder_output, teacher_forcing_ratio)

        return encoder_output_prob, encoder_output_lens, decoder_output_prob


