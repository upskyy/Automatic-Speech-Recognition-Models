# MIT License
#
# Copyright (c) 2021 Sangchun Ha
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

from dataclasses import dataclass


@dataclass
class DeepSpeech2Config:
    architecture: str = 'deepspeech2'
    input_size: int = 80
    hidden_size: int = 512
    num_layers: int = 3
    dropout: float = 0.3
    bidirectional: bool = True
    rnn_type: str = 'gru'
    mode: str = 'train'
