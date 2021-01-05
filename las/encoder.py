from torch import Tensor
import torch.nn as nn


class Encoder(nn.Module):
    supported_rnns = {'rnn': nn.RNN,
                      'lstm': nn.LSTM,
                      'gru': nn.GRU
                      }

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 rnn_type: str = 'lstm'
                 ) -> None:
        super(Encoder, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_size, num_layers, dropout, bidirectional, batch_first=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

    def forward(self,
                inputs: Tensor
                ) -> Tensor:
        output, _ = self.rnn(inputs)

        return output
