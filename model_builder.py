from models.encoder import Encoder
from models.decoder import Decoder
from models.model import ListenAttendSpell
import torch


def build_encoder(args, device: torch.device) -> Encoder:
    return Encoder(
        device,
        args.num_vocabs,
        args.input_size,
        args.encoder_hidden,
        args.encoder_layers,
        args.dropout,
        args.bidirectional,
        args.rnn_type,
        args.use_joint_ctc_attention,
    )


def build_decoder(args, device: torch.device) -> Decoder:
    return Decoder(
        device,
        args.num_vocabs,
        args.decoder_hidden,
        args.decoder_hidden,
        args.decoder_layers,
        args.max_len,
        args.dropout,
        args.rnn_type,
        args.sos_id,
        args.eos_id,
        args.attn_mechanism,
        args.smoothing,
    )


def build_model(encoder: Encoder, decoder: Decoder) -> ListenAttendSpell:
    return ListenAttendSpell(encoder, decoder)
