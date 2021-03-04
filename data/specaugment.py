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
import numpy as np
import random


class SpecAugment(object):
    def __init__(
            self,
            freq_mask_parameter: int,
            num_freq_mask: int,
            num_time_mask: int,
    ) -> None:
        self.freq_mask_parameter = freq_mask_parameter
        self.num_freq_mask = num_freq_mask
        self.num_time_mask = num_time_mask

    def __call__(self, feature: torch.FloatTensor) -> torch.FloatTensor:
        freq_length = feature.size(0)
        time_length = feature.size(1)
        time_mask_parameter = time_length / 20

        for _ in range(self.num_time_mask):
            t = int(np.random.uniform(low=0, high=time_mask_parameter))
            t0 = random.randint(0, time_length - t)
            feature[:, t0: t0 + t] = 0

        for _ in range(self.num_freq_mask):
            f = int(np.random.uniform(low=0, high=self.freq_mask_parameter))
            f0 = random.randint(0, freq_length - f)
            feature[f0: f0 + f, :] = 0

        return feature
