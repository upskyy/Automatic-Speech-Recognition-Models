from dataclasses import dataclass


@dataclass
class EvaluateConfig:
    dataset_path: str = ''
    audio_path: str = ''
    label_path: str = 'D:/label/aihub_labels.csv'
    model_path: str = ''
    save_transcripts_path: str = ''
    print_interval: int = 10
    num_vocabs: int = 2001
    pad_id: int = 0
    sos_id: int = 1
    eos_id: int = 2
    blank_id: int = 2000
    batch_size: int = 4
    num_workers: int = 4
    cuda: bool = True
    seed: int = 22
    mode: str = eval
