import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, bidirectional=True,
                 rnn_type='lstm'):
        super(Encoder, self).__init__()
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

    def forward(self, input_data):
        conv_output = self.conv(input_data)  # conv_output shape : (batch, num_channels, hidden_size, seq_len)
        batch, num_channels, hidden_size, seq_len = conv_output.size()
        conv_output = conv_output.view(batch, seq_len, -1).contiguous()  # (batch, seq_len, num_channels * hidden_size)
        conv_output = pack_padded_sequence(input=conv_output, lengths=seq_len, batch_first=True,
                                           enforce_sorted=False)  # 정렬 x
        output, _ = self.rnn(conv_output)
        output, input_lengths = pad_packed_sequence(output, batch_first=True)

        return output
