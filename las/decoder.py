import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
from las.attention import DotProductAttention


class Decoder(nn.Module):
    supported_rnns = {'rnn': nn.RNN,
                      'lstm': nn.LSTM,
                      'gru': nn.GRU
                      }

    def __init__(self,
                 num_vocabs: int,
                 embedding_dim: int,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 rnn_type: str = 'lstm',
                 pad_id: int = 0,
                 sos_id: int = 1,
                 eos_id: int = 2
                 ) -> None:
        super(Decoder, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(embedding_dim, hidden_size, num_layers, dropout, bidirectional=False, batch_first=True)
        self.num_characters = num_vocabs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_vocabs, embedding_dim)
        self.fc = nn.Linear(hidden_size << 1, num_vocabs)
        self.attention = DotProductAttention()

    def forward(self,
                inputs: Tensor,
                encoder_output: Tensor
                ) -> Tuple[Tensor, Tensor]:
        seq_len = inputs.size()[1]  # input_data shape : (batch, seq_len)
        embedded = self.embedding(inputs)  # embedded shape : (batch, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        rnn_output, _ = self.rnn(embedded)   # rnn_output shape : (batch, seq_len, hidden_size)

        context_vector = self.attention(rnn_output, encoder_output, encoder_output)
        context_vector = torch.cat((context_vector, rnn_output), dim=2)  # shape : (batch, seq_len, hidden_size << 1)

        context_vector = context_vector.view(-1, self.hidden_size << 1).contiguous()
        output = self.fc(context_vector)  # output shape : (batch * seq_len, num_vocabs)
        output = F.log_softmax(output, dim=1)

        return output, seq_len
