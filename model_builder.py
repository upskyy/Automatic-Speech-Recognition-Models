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

import torch
import torch.nn as nn
from models.las.encoder import Encoder
from models.las.decoder import Decoder
from models.las.model import ListenAttendSpell
from models.deepspeech2.model import DeepSpeech2
from omegaconf import DictConfig


def build_model(config: DictConfig, device: torch.device):
    if config.model.architecture == 'las':
        return build_las_model(config, device)

    elif config.model.architecture == 'deepspeech2':
        return build_ds2_model(config)


def load_test_model(config: DictConfig, device: torch.device) -> nn.Module:
    model = torch.load(config.eval.model_path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model = model.module

    model.encoder.device = device
    model.decoder.device = device

    return model


def build_encoder(config: DictConfig) -> Encoder:
    return Encoder(
        config.train.num_vocabs,
        config.model.input_size,
        config.model.encoder_hidden_size,
        config.model.encoder_layers,
        config.model.dropout,
        config.model.bidirectional,
        config.model.rnn_type,
        config.model.use_joint_ctc_attention
    )


def build_decoder(config: DictConfig, device: torch.device) -> Decoder:
    return Decoder(
        device,
        config.train.num_vocabs,
        config.model.decoder_hidden_size,
        config.model.decoder_hidden_size,
        config.model.decoder_layers,
        config.model.max_len,
        config.model.dropout,
        config.model.rnn_type,
        config.model.attn_mechanism,
        config.model.smoothing,
        config.train.sos_id,
        config.train.eos_id,
    )


def build_las_model(config: DictConfig, device: torch.device) -> ListenAttendSpell:
    encoder = build_encoder(config)
    decoder = build_decoder(config, device)

    return ListenAttendSpell(encoder, decoder)


def build_ds2_model(config: DictConfig) -> DeepSpeech2:
    return DeepSpeech2(
        config.train.num_vocabs,
        config.model.input_size,
        config.model.hidden_size,
        config.model.num_layers,
        config.model.dropout,
        config.model.bidirectional,
        config.model.rnn_type,
    )
