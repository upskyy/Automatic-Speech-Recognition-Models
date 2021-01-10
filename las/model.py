import torch
import torch.nn as nn
from torch import Tensor
from las.encoder import Encoder
from las.decoder import Decoder


class LAS(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module
                 ) -> None:
        super(LAS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                encoder_inputs: Tensor,
                decoder_inputs: Tensor,
                teacher_forcing_ratio: float = 1.0
                ) -> list:
        encoder_output = self.encoder(encoder_inputs)
        decoder_output = self.decoder(decoder_inputs, encoder_output, teacher_forcing_ratio)

        return decoder_output


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