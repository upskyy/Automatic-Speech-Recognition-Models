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

from dataclasses import dataclass


@dataclass
class FeatureConfig:
    extension: str = "pcm"
    sampling_rate: int = 16000
    frame_length: float = 0.020
    frame_stride: float = 0.010
    normalize: bool = False
    spec_augment: bool = False
    num_time_mask: int = 1
    num_freq_mask: int = 1


@dataclass
class MelSpectrogramConfig(FeatureConfig):
    feature_extraction: str = 'melspectrogram'
    n_mel: int = 80
    freq_mask_parameter: int = 18


@dataclass
class SpectrogramConfig(FeatureConfig):
    feature_extraction: str = 'spectrogram'
    n_mel = 0  # not used
    freq_mask_parameter: int = 24


@dataclass
class MFCCConfig(FeatureConfig):
    feature_extraction: str = 'mfcc'
    n_mfcc: int = 40
    freq_mask_parameter: int = 8


@dataclass
class FilterBankConfig(FeatureConfig):
    feature_extraction: str = 'filterbank'
    n_mel: int = 80
