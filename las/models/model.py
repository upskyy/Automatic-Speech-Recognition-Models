import torch.nn as nn
from typing import Tuple
from torch import Tensor


class LAS(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module
    ) -> None:
        super(LAS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            encoder_inputs: Tensor,
            encoder_inputs_lens: Tensor,
            decoder_inputs: Tensor,
            teacher_forcing_ratio: float = 1.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        self.encoder.rnn.flatten_parameters()
        encoder_output, encoder_output_prob, encoder_output_lens = self.encoder(encoder_inputs, encoder_inputs_lens)

        self.decoder.rnn.flatten_parameters()
        decoder_output_prob = self.decoder(decoder_inputs, encoder_output, teacher_forcing_ratio)

        return encoder_output_prob, encoder_output_lens, decoder_output_prob

#
# listener = Encoder(num_vocabs=2000)
# speller = Decoder(num_vocabs=2000)
# las = LAS(listener, speller)
# print(las)

"""
listener = Encoder()
speller = Decoder(num_vocabs=2000)
las = LAS(listener, speller)

audio_feature = torch.FloatTensor([1, 2, 3] * 800).view(3, 80, 10)  # B, D, T
target = torch.LongTensor([1, 3, 5] * 20).view(3, 20)  # B, T

output = las(audio_feature, target, teacher_forcing_ratio=1.0)

print(output)
print(output.size())
"""