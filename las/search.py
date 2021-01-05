import torch
import torch.nn as nn


class GreedySearch(nn.Module):
    def __init__(self, id2char):
        super(GreedySearch, self).__init__()
        self.id2char = id2char

    def forward(self, decoder_output, seq_len):
        max_indexes = torch.argmax(decoder_output, dim=1)
        sentences = list()
        temp = list()
        for idx, max_index in enumerate(max_indexes):
            if idx % seq_len == 0:
                sentences.append("".join(temp))
                temp = list()
            temp.append(self.id2char[max_index])

        return sentences
