import torch
from torch import Tensor
import torch.nn as nn


class GreedySearch(nn.Module):
    def __init__(self,
                 id2char: dict
                 ) -> None:
        super(GreedySearch, self).__init__()
        self.id2char = id2char

    def forward(self,
                decoder_output: Tensor,
                seq_len: int
                ) -> list:
        max_indexes = torch.argmax(decoder_output, dim=1)
        sentences = list()
        temp = list()
        for idx, max_index in enumerate(max_indexes):
            if idx % seq_len == 0:
                sentences.append("".join(temp))
                temp = list()
            temp.append(self.id2char[max_index])

        return sentences
