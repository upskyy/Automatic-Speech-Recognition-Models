import torch
import random
from torch import Tensor
from typing import Optional, Any, Tuple
import torch.nn as nn
import torch.nn.functional as F
from models.attention import ScaledDotProductAttention, LocationAwareAttention


class Decoder(nn.Module):
    """
    Converts higher feature values to character probability values using attention

    Args:
        device (torch.device): 'cuda' or 'cpu'
        num_vocabs (int): the number of vocabulary
        embedding_dim (int): dimension of embedding (default : 512)
        hidden_size (int): the number of features in the decoder hidden state (default : 512)
        num_layers (int): the number of recurrent layers (default: 2)
        max_len (int): max length of label (default : 120)
        dropout (float): dropout probability of encoder (default: 0.3)
        rnn_type (str): type of RNN cell (default: lstm)
        attn_mechanism (str): type of attention mechanism (default: location)
        smoothing (bool) : flag indication smoothing or not (default: False)
        sos_id (int): index of the start of sentence (default: 1)
        eos_id (int): index of the end of sentence (default: 2)

    Inputs: inputs, encoder_output, teacher_forcing_ratio
        - **inputs** (batch, seq_len): Used as ground truth when using teacher_forcing
        - **encoder_output** (batch, seq_len, dimension): tensor with containing the outputs of the encoder.
          Used in the attention mechanism
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used.

    Returns: decoder_output_prob
        - **decoder_output_prob** (batch, seq_len, num_vocabs): Tensor expressing the log probability value of each word
    """
    supported_rnns = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU
    }

    def __init__(
            self,
            device: torch.device,
            num_vocabs: int,
            embedding_dim: int = 512,
            hidden_size: int = 512,
            num_layers: int = 2,
            max_len: int = 120,
            dropout: float = 0.3,
            rnn_type: str = 'lstm',
            attn_mechanism: str = 'location',
            smoothing: bool = False,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(Decoder, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(embedding_dim, hidden_size, num_layers, True, True, dropout, bidirectional=False)
        self.num_vocabs = num_vocabs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.embedding_dropout = nn.Dropout(p=dropout)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.attn_mechanism = attn_mechanism
        self.device = device
        self.embedding = nn.Embedding(num_vocabs, embedding_dim)
        self.fc = nn.Linear(hidden_size << 1, num_vocabs)

        if self.attn_mechanism == 'location':
            self.attention = LocationAwareAttention(hidden_size, hidden_size, smoothing)
        elif self.attn_mechanism == 'scaled_dot':
            self.attention = ScaledDotProductAttention(hidden_size)

    def forward_step(
            self,
            inputs: Tensor,
            encoder_output: Tensor,
            attn_distribution: Optional[Any] = None,
    ) -> Tuple[Tensor, Tensor]:
        context_vector = None

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)

        embedded = self.embedding(inputs).to(self.device)  # embedded shape : (batch, dec_T, embedding_dim)
        embedded = self.embedding_dropout(embedded)

        rnn_output, _ = self.rnn(embedded)  # rnn_output shape : (batch, dec_T, dec_D)

        if self.attn_mechanism == 'location':
            context_vector, attn_distribution = self.attention(rnn_output, encoder_output, encoder_output, attn_distribution)
        elif self.attn_mechanism == 'scaled_dot':
            context_vector, attn_distribution = self.attention(rnn_output, encoder_output, encoder_output)

        context_vector = torch.cat((context_vector, rnn_output), dim=-1)  # shape : (batch, dec_T, dec_D << 1)

        output = self.fc(torch.tanh(context_vector))  # output shape : (batch, dec_T, num_vocabs)
        output = F.log_softmax(output, dim=-1)

        return output, attn_distribution

    def forward(
            self,
            inputs: Tensor,
            encoder_output: Tensor,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tensor:

        attn_distribution = None
        decoder_output_prob = list()

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        inputs, batch, max_len = self._validate_args(inputs, encoder_output, use_teacher_forcing)

        if use_teacher_forcing:
            inputs = inputs[inputs != self.eos_id].contiguous().view(batch, -1)
            if self.attn_mechanism == 'location':
                for i in range(inputs.size(1)):
                    input_data = inputs[:, i]
                    output, attn_distribution = self.forward_step(input_data, encoder_output, attn_distribution)
                    output = output.squeeze(1)
                    decoder_output_prob.append(output)
                decoder_output_prob = torch.stack(decoder_output_prob, dim=1)

            else:
                decoder_output_prob, attn_distribution = self.forward_step(inputs, encoder_output, attn_distribution)

        else:
            for _ in range(max_len):
                output, attn_distribution = self.forward_step(inputs, encoder_output, attn_distribution)
                output = output.squeeze(1)
                decoder_output_prob.append(output)
                inputs = output.topk(1)[1]  # [value, index]

            decoder_output_prob = torch.stack(decoder_output_prob, dim=1)

        return decoder_output_prob  # (B, T, D)

    def _validate_args(
            self,
            inputs: Tensor,
            encoder_output: Tensor,
            use_teacher_forcing: bool,
    ) -> Tuple[Tensor, int, int]:
        assert encoder_output is not None
        batch = encoder_output.size(0)

        if not use_teacher_forcing:
            inputs = torch.LongTensor([self.sos_id] * batch).unsqueeze(1)
            max_len = self.max_len

            if torch.cuda.is_available():
                inputs = inputs.cuda()

        else:
            max_len = inputs.size(1) - 1  # eos

        return inputs, batch, max_len
