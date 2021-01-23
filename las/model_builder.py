from las.models.encoder import Encoder
from las.models.decoder import Decoder
from las.models.model import LAS
import torch.nn as nn


def build_encoder(args):
    return Encoder(
        args.num_vocabs,
        args.input_size,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        args.rnn_type
    )


def build_decoder(args, device):
    return Decoder(
        args.num_vocabs,
        args.embedding_dim,
        args.hidden_size,
        args.num_layers,
        args.max_len,
        args.dropout,
        args.rnn_type,
        args.pad_id,
        args.sos_id,
        args.eos_id,
        args.attn_mechanism,
        args.smoothing,
        device
    )


def build_model(encoder: nn.Module, decoder: nn.Module):
    return LAS(encoder, decoder)
