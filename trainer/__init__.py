from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset_path: str = 'D:/dataset/transcripts.txt'
    audio_path: str = 'E:/KsponSpeech'
    label_path: str = 'D:/label/aihub_labels.csv'

    print_interval: int = 10
    num_vocabs: int = 2001
    pad_id: int = 0
    sos_id: int = 1
    eos_id: int = 2
    blank_id: int = 2000

    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-06

    cuda: bool = True
    seed: int = 22


@dataclass
class ListenAttendSpellTrainConfig(TrainConfig):
    model_save_path: str = 'las_model.pt'
    epochs: int = 20


@dataclass
class DeepSpeech2TrainConfig(TrainConfig):
    model_save_path: str = 'deepspeech2_model.pt'
    epochs: int = 50
