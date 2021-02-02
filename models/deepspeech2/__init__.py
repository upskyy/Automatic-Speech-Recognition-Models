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
