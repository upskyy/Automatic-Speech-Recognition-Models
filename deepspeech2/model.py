import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class MaskConv(nn.Module):
    def __init__(
            self,
            sequential: nn.Sequential
    ) -> None:
        super(MaskConv, self).__init__()
        self.sequential = sequential

    def forward(
            self,
            inputs: Tensor,
            seq_lens: Tensor
    ) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)

            mask = torch.BoolTensor(output.size()).fill_(0)
            if output.is_cuda:
                mask = mask.cuda()

            seq_lens = self.get_seq_lens(module, seq_lens)

            for idx, seq_len in enumerate(seq_lens):
                seq_len = seq_len.item()
                if (mask[idx].size(2) - seq_len) > 0:
                    mask[idx].narrow(2, seq_len, mask[idx].size(2 - seq_len)).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lens

    def get_seq_lens(
            self,
            module: nn.Module,
            seq_lens: Tensor
    ) -> Tensor:
        if isinstance(module, nn.Conv2d):
            seq_lens = ((seq_lens + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) /
                       module.stride[1] + 1)

        return seq_lens.int()



class DeepSpeech2(nn.Module):
    supported_rnns = {'rnn': nn.RNN,
                      'lstm': nn.LSTM,
                      'gru': nn.GRU
                      }

    def __init__(self,
                 input_size: int,
                 num_vocabs: int,
                 hidden_size: int = 512,
                 num_layers: int = 5,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 rnn_type: str = 'gru',
                 device: str = 'cpu'
                 ) -> None:
        super(DeepSpeech2, self).__init__()
        input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
        input_size <<= 5
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_size, num_layers, dropout, bidirectional, batch_first=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True)
        )
        self.rnn_output_size = hidden_size << 1 if bidirectional else hidden_size
        self.fc = nn.Linear(self.rnn_output_size, num_vocabs)

    def forward(self,
                inputs: Tensor,
                input_lengths: Tensor
                ) -> Tensor:
        conv_output = self.conv(inputs)  # conv_output shape : (B, C, D, T)
        conv_output = conv_output.permute(0, 3, 1, 2)
        batch, seq_len, num_channels, hidden_size = conv_output.size()
        conv_output = conv_output.view(batch, seq_len, -1).contiguous()  # (batch, seq_len, num_channels * hidden)

        rnn_output, _ = self.rnn(conv_output)
        output = self.fc(rnn_output.view(-1, self.rnn_output_size).contiguous())
        output = F.log_softmax(output, dim=1)  # output shape : (batch*seq_len, num_vocabs)

        return output



