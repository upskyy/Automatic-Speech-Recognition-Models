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
class ListenAttendSpellConfig:
    architecture: str = 'las'
    input_size: int = 80
    encoder_hidden_size: int = 256
    decoder_hidden_size: int = 512
    encoder_layers: int = 3
    decoder_layers: int = 2
    num_head: int = 8
    dropout: float = 0.3
    bidirectional: bool = True
    rnn_type: str = 'lstm'
    teacher_forcing_ratio: float = 1.0
    use_joint_ctc_attention: bool = False
    max_len: int = 120
    attn_mechanism: str = 'location'
    smoothing: bool = False
    mode: str = 'train'


@dataclass
class JointCTCAttentionLASConfig(ListenAttendSpellConfig):
    use_joint_ctc_attention: bool = True
    ctc_weight: float = 0.2
    cross_entropy_weight: float = 0.8


