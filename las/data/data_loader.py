import torch
from torch import Tensor
from typing import Tuple
import os
import numpy as np
import librosa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


def load_audio(path: str, extension: str = 'pcm') -> np.ndarray:
    if extension == 'pcm':
        sound = np.memmap(path, dtype='h', mode='r').astype('float32')

        return sound / 32767

    elif extension == 'wav':
        sound, _ = librosa.load(path, sr=16000)

        return sound


class SpectrogramDataset(Dataset):
    def __init__(
            self,
            audio_path: str,
            transcript_path: str,
            num_dataset: int,
            sampling_rate: int = 16000,
            n_mel: int = 80,
            frame_length: float = 0.020,
            frame_stride: float = 0.010,
            extension: str = 'pcm',
            sos_id: int = 1,
            eos_id: int = 2
    ) -> None:
        super(SpectrogramDataset, self).__init__()
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.num_dataset = num_dataset
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.sampling_rate = sampling_rate
        self.n_mel = n_mel
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.extension = extension
        self.n_fft = int(round(self.sampling_rate * self.frame_length))
        self.hop_length = int(round(self.sampling_rate * self.frame_stride))

    def parse_audio(self, path: str) -> np.ndarray:
        sound = load_audio(path, self.extension)

        melspectrogram = librosa.feature.melspectrogram(
            sound,
            sr=self.sampling_rate,
            n_mels=self.n_mel,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        log_melspectrogram = librosa.amplitude_to_db(melspectrogram)

        return log_melspectrogram

    def parse_transcript(self, path: str):
        with open(path, 'r') as f:
            transcript = list(map(int, f.read().split()))
            transcript = [self.sos_id] + transcript + [self.eos_id]

        return transcript

    def parse_filename(self, idx: int, extension: str):
        if idx < 10:
            return '00000' + str(idx) + '.' + extension
        elif idx < 100:
            return '0000' + str(idx) + '.' + extension
        elif idx < 1000:
            return '000' + str(idx) + '.' + extension
        elif idx < 10000:
            return '00' + str(idx) + '.' + extension
        elif idx < 100000:
            return '0' + str(idx) + '.' + extension
        else:
            return str(idx)

    def __getitem__(self, idx):
        audio_feature = self.parse_audio(os.path.join(self.audio_path, self.parse_filename(idx, self.extension)))
        transcript = self.parse_transcript(os.path.join(self.transcript_path, self.parse_filename(idx, 'txt')))

        return audio_feature, transcript

    def __len__(self):
        return self.num_dataset


def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)

    return seqs, targets, seq_lengths, target_lengths


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, sampler, batch_size: int = 1):
        super(BucketingSampler, self).__init__(sampler)
        self.sampler = sampler
        ids = list(range(len(sampler)))
        self.buckets = [ids[begin:begin + batch_size] for begin in range(0, len(ids), batch_size)]

    def __iter__(self):
        for bucket in self.buckets:
            np.random.shuffle(bucket)
            yield bucket

    def __len__(self):
        return len(self.buckets)
