from dataclasses import dataclass


@dataclass
class ListenAttendSpellConfig:
    architecture: str = 'las'
    input_size: int = 80
    encoder_hidden_size: int = 256
    decoder_hidden_size: int = 512
    encoder_layers: int = 3
    decoder_layers: int = 2
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


