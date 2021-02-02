from dataclasses import dataclass


@dataclass
class MelSpectrogramConfig:
    extension: str = "pcm"
    sampling_rate: int = 16000
    n_mel: int = 80
    frame_length: int = 20
    frame_stride: int = 10
