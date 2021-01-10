from torch import Tensor
import torch.nn as nn


class Encoder(nn.Module):
    supported_rnns = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU
    }

    def __init__(
            self,
            input_size: int = 80,
            hidden_size: int = 256,
            num_layers: int = 3,
            dropout: float = 0.3,
            bidirectional: bool = True,
            rnn_type: str = 'lstm'
    ) -> None:
        super(Encoder, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_size, num_layers, True, True, dropout, bidirectional)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

    def forward(
            self,
            inputs: Tensor
    ) -> Tensor:
        inputs = inputs.transpose(1, 2)
        output, _ = self.rnn(inputs)

        return output


"""

listener = Encoder()
audio_feature = torch.FloatTensor([1, 2, 3] * 800).view(3, 80, 10)  # (B, D, T)
output = listener(audio_feature)
print(output.size())   


# torch.Size([3, 10, 512])

"""