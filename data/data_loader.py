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
import os
import numpy as np
import librosa

from torch import Tensor
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from data.specaugment import SpecAugment
from data.feature import (
    Spectrogram,
    MelSpectrogram,
    MFCC,
    FilterBank,
)


def load_audio(path: str, extension: str = 'pcm', sampling_rate: int = 16000) -> np.ndarray:
    if extension == 'pcm':
        sound = np.memmap(path, dtype='h', mode='r').astype('float32')

        return sound / 32767

    elif extension == 'wav':
        sound, _ = librosa.load(path, sr=sampling_rate)

        return sound


class SpectrogramDataset(Dataset, object):
    def __init__(
            self,
            default_audio_path: str,
            audio_paths: list,
            transcript_list: list,
            sampling_rate: int = 16000,
            n_dim: int = 80,
            frame_length: float = 0.020,
            frame_stride: float = 0.010,
            extension: str = 'pcm',
            feature_extraction: str = 'melspectrogram',
            normalize: bool = True,
            spec_augment: bool = True,
            freq_mask_parameter: int = 18,
            num_time_mask: int = 1,
            num_freq_mask: int = 1,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(SpectrogramDataset, self).__init__()
        self.default_audio_path = default_audio_path
        self.audio_paths = audio_paths
        self.transcript_list = transcript_list
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.extension = extension
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.spec_augment = spec_augment
        self.specaugment = SpecAugment(freq_mask_parameter, num_freq_mask, num_time_mask)
        n_fft = int(round(sampling_rate * frame_length))
        hop_length = int(round(sampling_rate * frame_stride))

        if feature_extraction == 'spectrogram':
            self.feature_extractor = Spectrogram(n_fft, hop_length)
        elif feature_extraction == 'melspectrogram':
            self.feature_extractor = MelSpectrogram(n_fft, hop_length, sampling_rate, n_dim)
        elif feature_extraction == 'mfcc':
            self.feature_extractor = MFCC(n_fft, hop_length, sampling_rate, n_dim)
        # elif feature_extraction == 'filterbank':
        #     self.feature_extractor = FilterBank(frame_length, frame_stride, sampling_rate, n_dim)

    def parse_audio(self, path: str) -> torch.FloatTensor:
        sound = load_audio(path, self.extension, self.sampling_rate)

        feature = self.feature_extractor(sound)

        if self.normalize:
            feature = (feature - np.mean(feature)) / np.std(feature)

        if self.spec_augment:
            feature = self.specaugment(feature)

        return feature

    def parse_transcript(self, transcript: str) -> torch.LongTensor:
        transcript = list(map(int, transcript.split()))
        transcript = [self.sos_id] + transcript + [self.eos_id]
        transcript = torch.LongTensor(transcript)

        return transcript

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
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


class AudioDataLoader(DataLoader, object):
    def __init__(self, *args, **kwargs) -> None:
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler, object):
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
