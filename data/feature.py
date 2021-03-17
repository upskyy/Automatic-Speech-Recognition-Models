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

import numpy as np
import torch
import librosa
from scipy import signal
from torch import Tensor


class Spectrogram(object):
    def __init__(
            self,
            n_fft: int,
            hop_length: int,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, sound: np.ndarray, normalize: bool) -> torch.FloatTensor:
        stft = librosa.stft(sound, n_fft=self.n_fft, hop_length=self.hop_length, window=signal.windows.hamming)
        spectrogram, _ = librosa.magphase(stft)
        spectrogram = np.log1p(spectrogram)

        if normalize:
            spectrogram -= spectrogram.mean()
            spectrogram /= np.std(spectrogram)

        return torch.FloatTensor(spectrogram)


class MelSpectrogram(object):
    def __init__(
            self,
            n_fft: int,
            hop_length: int,
            sampling_rate: int = 16000,
            n_mel: int = 80,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.n_mel = n_mel

    def __call__(self, sound: np.ndarray, normalize: bool) -> torch.FloatTensor:
        melspectrogram = librosa.feature.melspectrogram(
            sound,
            sr=self.sampling_rate,
            n_mels=self.n_mel,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        log_melspectrogram = librosa.amplitude_to_db(melspectrogram)

        if normalize:
            log_melspectrogram -= log_melspectrogram.mean()
            log_melspectrogram /= np.std(log_melspectrogram)

        return torch.FloatTensor(log_melspectrogram)


class MFCC(object):
    def __init__(
            self,
            n_fft: int,
            hop_length: int,
            sampling_rate: int = 16000,
            n_mfcc: int = 40,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc

    def __call__(self, sound: np.ndarray, normalize: bool) -> torch.FloatTensor:
        mfcc = librosa.feature.mfcc(
            sound,
            sr=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        if normalize:
            mfcc -= mfcc.mean()
            mfcc /= np.std(mfcc)

        return torch.FloatTensor(mfcc)


class FilterBank:
    def __init__(
            self,
            frame_length: float = 0.020,
            frame_stride: float = 0.010,
            sampling_rate: int = 16000,
            n_mel: int = 80,
    ) -> None:
        self.frame_length = frame_length * 1000
        self.frame_stride = frame_stride * 1000
        self.sampling_rate = sampling_rate
        self.n_mel = n_mel
        import torchaudio

    def __call__(self, sound: np.ndarray, normalize: bool):
        filter_bank = torchaudio.compliance.kaldi.fbank(
            Tensor(sound).unsqueeze(0),
            num_mel_bins=self.n_mel,
            frame_length=self.frame_length,
            frame_shift=self.frame_stride,
            sample_frequency=float(self.sampling_rate),
        )

        filter_bank = filter_bank.transpose(0, 1)

        if normalize:
            filter_bank -= filter_bank.mean()
            filter_bank /= np.std(filter_bank)

        return filter_bank