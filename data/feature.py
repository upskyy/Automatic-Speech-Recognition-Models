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


class Spectrogram(object):
    def __init__(
            self,
            n_fft: int,
            hop_length: int,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, sound: np.ndarray) -> torch.FloatTensor:
        stft = librosa.stft(sound, n_fft=self.n_fft, hop_length=self.hop_length, window=signal.windows.hamming)
        spectrogram, _ = librosa.magphase(stft)

        spectrogram = np.log1p(spectrogram)
        spectrogram = torch.FloatTensor(spectrogram)

        return spectrogram


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

    def __call__(self, sound: np.ndarray) -> torch.FloatTensor:
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

    def __call__(self, sound: np.ndarray) -> torch.FloatTensor:
        mfcc = librosa.feature.mfcc(
            sound,
            sr=self.sampling_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mfcc = torch.FloatTensor(mfcc)

        return mfcc


# class FilterBank:
#     def __init__(
#             self,
#             frame_length: float = 0.020,
#             frame_stride: float = 0.010,
#             sampling_rate: int = 16000,
#             n_mel: int = 80,
#     ) -> None:
#         self.frame_length = frame_length
#         self.frame_stride = frame_stride
#         self.sampling_rate = sampling_rate
#         self.n_mel = n_mel
#
#     def __call__(self, sound: np.ndarray):
