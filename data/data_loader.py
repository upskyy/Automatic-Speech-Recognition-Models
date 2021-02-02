import torch
import os
import numpy as np
import librosa
from torch import Tensor
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence


class SpectrogramDataset(Dataset):
    def __init__(
            self,
            default_audio_path: str,
            audio_paths: list,
            transcript_list: list,
            sampling_rate: int = 16000,
            n_mel: int = 80,
            frame_length: float = 0.020,
            frame_stride: float = 0.010,
            extension: str = 'pcm',
            sos_id: int = 1,
            eos_id: int = 2
    ) -> None:
        super(SpectrogramDataset, self).__init__()
        self.default_audio_path = default_audio_path
        self.audio_paths = audio_paths
        self.transcript_list = transcript_list
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.sampling_rate = sampling_rate
        self.n_mel = n_mel
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.extension = extension
        self.n_fft = int(round(self.sampling_rate * self.frame_length))
        self.hop_length = int(round(self.sampling_rate * self.frame_stride))

    def load_audio(self, path: str, extension: str = 'pcm') -> np.ndarray:
        if extension == 'pcm':
            sound = np.memmap(path, dtype='h', mode='r').astype('float32')

            return sound / 32767

        elif extension == 'wav':
            sound, _ = librosa.load(path, sr=self.sampling_rate)

            return sound

    def parse_audio(self, path: str) -> Tensor:
        sound = self.load_audio(path, self.extension)

        melspectrogram = librosa.feature.melspectrogram(
            sound,
            sr=self.sampling_rate,
            n_mels=self.n_mel,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        log_melspectrogram = librosa.amplitude_to_db(melspectrogram)
        log_melspectrogram = torch.FloatTensor(log_melspectrogram)

        return log_melspectrogram

    def parse_transcript(self, transcript: str) -> Tensor:
        transcript = list(map(int, transcript.split()))
        transcript = [self.sos_id] + transcript + [self.eos_id]
        transcript = torch.LongTensor(transcript)

        return transcript

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        audio_feature = self.parse_audio(os.path.join(self.default_audio_path, self.audio_paths[idx]))
        transcript = self.parse_transcript(self.transcript_list[idx])

        return audio_feature, transcript

    def __len__(self) -> int:
        return len(self.audio_paths)


def _collate_fn(batch) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)

    seq_lengths = [data[0].size(1) for data in batch]
    seq_lengths = torch.IntTensor(seq_lengths)

    target_lengths = [len(data[1]) for data in batch]
    target_lengths = torch.IntTensor(target_lengths)

    seqs = [data[0].transpose(0, 1) for data in batch]
    targets = [data[1] for data in batch]

    pad_seqs = pad_sequence(seqs, batch_first=True)
    pad_seqs = pad_seqs.contiguous().transpose(1, 2)  # (B, D, T)

    pad_targets = pad_sequence(targets, batch_first=True)  # (B, T)

    return pad_seqs, pad_targets, seq_lengths, target_lengths


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs) -> None:
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, sampler, batch_size: int = 1) -> None:
        super(BucketingSampler, self).__init__(sampler)
        number_list = list(range(len(sampler)))
        self.buckets = [number_list[begin:begin + batch_size] for begin in range(0, len(number_list), batch_size)]

    def __iter__(self):
        for bucket in self.buckets:
            np.random.shuffle(bucket)
            yield bucket

    def __len__(self) -> int:
        return len(self.buckets)
