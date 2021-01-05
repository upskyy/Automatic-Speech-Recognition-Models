import torch
import torch.nn as nn
import torch.nn.functional as F
from las.attention import DotProductAttention


class Decoder(nn.Module):
    def __init__(self, num_characters, embedding_dim, hidden_size=512, num_layers=2, dropout=0.3, rnn_type='lstm'):
        super(Decoder, self).__init__()
        self.num_characters = num_characters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        if rnn_type.lower() == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, dropout, batch_first=True)
        elif rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, dropout, batch_first=True)
        self.embedding = nn.Embedding(num_characters, embedding_dim)
        self.fully_connected = nn.Linear(2 * hidden_size, num_characters)
        self.attention = DotProductAttention()

    def forward(self, input_data, encoder_output):
        seq_len = input_data.size()[1]
        embedded = self.embedding(input_data)  # input_data shape : (batch, seq_len)
        rnn_output, _ = self.rnn(embedded)  # embedded shape : (batch, seq_len, embedding_dim)
        # query = rnn_output  # rnn_output shape : (batch, seq_len, hidden_size)
        context_vector = self.attention(rnn_output, encoder_output, encoder_output)
        context_vector = torch.cat((context_vector, rnn_output), dim=2)  # shape : (batch, seq_len, 2 * hidden_size)

        context_vector = context_vector.view(-1, 2 * self.hidden_size).contiguous()
        output = self.fully_connected(context_vector)  # output shape : (batch * seq_len, num_characters)
        output = F.softmax(output, dim=1)

        return output, seq_len
