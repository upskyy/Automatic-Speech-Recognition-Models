import torch
import random
from torch import Tensor
from typing import Optional, Any, Tuple
import torch.nn as nn
import torch.nn.functional as F
from las.attention import ScaledDotProductAttention, LocationAwareAttention


class Decoder(nn.Module):
    supported_rnns = {'rnn': nn.RNN,
                      'lstm': nn.LSTM,
                      'gru': nn.GRU
                      }

    def __init__(self,
                 num_vocabs: int,
                 embedding_dim: int = 512,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 max_len: int = 150,
                 dropout: float = 0.3,
                 rnn_type: str = 'lstm',
                 pad_id: int = 0,
                 sos_id: int = 1,
                 eos_id: int = 2,
                 attn_type: str = 'location',
                 smoothing: bool = False
                 ) -> None:
        super(Decoder, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(embedding_dim, hidden_size, num_layers, dropout, bidirectional=False, batch_first=True)
        self.num_vocabs = num_vocabs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.embedding_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.attn_type = attn_type
        self.embedding = nn.Embedding(num_vocabs, embedding_dim)
        self.fc = nn.Linear(hidden_size << 1, num_vocabs)

        if self.attn_type == 'location':
            self.attention = LocationAwareAttention(hidden_size, hidden_size, smoothing)
        elif self.attn_type == 'scaled_dot':
            self.attention = ScaledDotProductAttention(hidden_size)

    def forward_step(self,
                     inputs: Tensor,
                     encoder_output: Tensor,
                     attn_distribution: Optional[Any] = None
                     ) -> Tuple[Tensor, Tensor]:

        embedded = self.embedding(inputs)  # embedded shape : (batch, dec_T, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        rnn_output, _ = self.rnn(embedded)  # rnn_output shape : (batch, dec_T, dec_D)

        if self.attn_type == 'location':
            context_vector, attn_distribution = self.attention(rnn_output, encoder_output, encoder_output,
                                                               attn_distribution)

        elif self.attn_type == 'scaled_dot':
            context_vector, attn_distribution = self.attention(rnn_output, encoder_output, encoder_output)

        context_vector = torch.cat((context_vector, rnn_output), dim=-1)  # shape : (batch, dec_T, dec_D << 1)

        output = self.fc(context_vector)  # output shape : (batch, dec_T, num_vocabs)
        output = F.log_softmax(output, dim=-1)

        return output, attn_distribution

    def forward(self,
                inputs: Tensor,
                encoder_output: Tensor,
                teacher_forcing_ratio: float = 1.0,
                ) -> Tensor:

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        inputs, max_len = self._validate_args(inputs, encoder_output, use_teacher_forcing)
        attn_distribution = None
        decoder_output = list()

        if use_teacher_forcing:
            inputs = inputs[:, :-1]
            if self.attn_type == 'location':
                for i in range(inputs.size(1)):
                    inputs = inputs[:, i]
                    output, attn_distribution = self.forward_step(inputs, encoder_output, attn_distribution)
                    output = output.squeeze(1)
                    decoder_output.append(output)
                decoder_output = torch.stack(decoder_output, dim=1)

            else:
                decoder_output, attn_distribution = self.forward_step(inputs, encoder_output, attn_distribution)

        else:
            for _ in range(max_len):
                output, attn_distribution = self.forward_step(inputs, encoder_output,
                                                              attn_distribution)  # output : (B, 1, num_vocabs)
                output = output.squeeze(1)
                decoder_output.append(output)
                inputs = output.topk(1)[1]  # [value, index]
                # if inputs == self.eos_id:             ???
                #     break                             ???

            decoder_output = torch.stack(decoder_output, dim=1)

        return decoder_output  # (B, T, D)

    def _validate_args(self,
                       inputs: Tensor,
                       encoder_output: Tensor,
                       use_teacher_forcing: bool,
                       ) -> Tuple[Tensor, int]:

        batch = encoder_output.size(0)
        if not use_teacher_forcing:
            inputs = torch.LongTensor([self.sos_id] * batch).unsqueeze(1)
            max_len = self.max_len
        else:
            max_len = inputs.size(1) - 1  # eos

        return inputs, max_len

