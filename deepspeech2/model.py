import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DeepSpeech2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, num_layers=5, dropout=0.3, bidirectional=True,
                 rnn_type='gru'):
        super(DeepSpeech2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.Hardtanh()
        )
        if rnn_type.lower() == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout, bidirectional, batch_first=True)
        elif rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout, bidirectional, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout, bidirectional, batch_first=True)
        self.fully_connected = nn.Linear(2 * hidden_size, output_size)

    def forward(self, input_data):
        conv_output = self.conv(input_data)  # conv_output shape : (batch, num_channels(96), hidden, seq_len)
        batch, num_channels, hidden_size, seq_len = conv_output.size()
        conv_output = conv_output.view(batch, seq_len, -1).contiguous()  # (batch, seq_len, num_channels * hidden)
        # num_channels * hidden == input_size
        conv_output = pack_padded_sequence(input=conv_output, lengths=seq_len, batch_first=True,
                                           enforce_sorted=False)  # 정렬 x
        rnn_output, _ = self.rnn(conv_output)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
        output = self.fully_connected(rnn_output.view(-1, 2 * hidden_size).contiguous())
        output = F.log_softmax(output, dim=1)  # output shape : (batch*seq_len, output_size)
        output = output.view(seq_len, batch, -1).contiguous()

        return output


"""
model = DeepSpeech2(input_size, output_size, hidden_size=512, num_layers=5, dropout=0.3, bidirectional=True,
                    rnn_type='gru')
input_data =
output = model(input_data)
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