from typing import Tuple
from torch import Tensor
from models.las.encoder import Encoder
from models.las.decoder import Decoder
import torch.nn as nn


class ListenAttendSpell(nn.Module):
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
        self.encoder.rnn.flatten_parameters()
        encoder_output, encoder_output_prob, encoder_output_lens = self.encoder(encoder_inputs, encoder_inputs_lens)

        self.decoder.rnn.flatten_parameters()
        decoder_output_prob = self.decoder(decoder_inputs, encoder_output, teacher_forcing_ratio)

        return encoder_output_prob, encoder_output_lens, decoder_output_prob


