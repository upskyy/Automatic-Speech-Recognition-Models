import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
                 rnn_type: str = 'gru'
                 ) -> None:
        super(DeepSpeech2, self).__init__()
        input_size = int((input_size + 2 * 20 - 41) / 2 + 1)
        input_size = int((input_size + 2 * 10 - 21) / 2 + 1)
        input_size <<= 5
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_size, num_layers, dropout, bidirectional, batch_first=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
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
                inputs: Tensor
                ) -> Tensor:
        conv_output = self.conv(inputs)  # conv_output shape : (batch, num_channels, hidden, seq_len)
        batch, num_channels, hidden_size, seq_len = conv_output.size()
        conv_output = conv_output.view(batch, seq_len, -1).contiguous()  # (batch, seq_len, num_channels * hidden)

        conv_output = pack_padded_sequence(input=conv_output, lengths=seq_len, batch_first=True,
                                           enforce_sorted=False)  # 정렬 x
        rnn_output, _ = self.rnn(conv_output)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)

        output = self.fc(rnn_output.view(-1, self.rnn_output_size).contiguous())
        output = F.log_softmax(output, dim=1)  # output shape : (batch*seq_len, num_vocabs)

        return output



"""
model = DeepSpeech2(input_size, output_size, hidden_size=512, num_layers=5, dropout=0.3, bidirectional=True,
                    rnn_type='gru')
input_data =
output = model(input_data)
output = output.view(seq_len, batch, -1).contiguous()
output_lengths = torch.full(size=(batch,), fill_value=seq_len)
target =
target_lengths =                  # shape : (batch,)   각 target의 길이를 표현
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
ctc_loss = nn.CTCloss()
loss = ctc_loss(output, target, output_lengths, target_lengths)

optimizer.zero_grad()
loss.backward()
optimizer.step()
"""
