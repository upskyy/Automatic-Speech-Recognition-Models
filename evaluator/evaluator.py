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
import torch.nn as nn
import pandas as pd
from models.search import GreedySearch
from data.data_loader import AudioDataLoader
from vocabulary import label_to_string, get_distance
from omegaconf import DictConfig


class Evaluator(object):
    def __init__(
            self,
            config: DictConfig,
            device: torch.device,
            test_loader: AudioDataLoader,
            id2char: dict,
    ) -> None:
        self.config = config
        self.device = device
        self.test_loader = test_loader
        self.id2char = id2char
        self.decoder = GreedySearch(device)

    def evaluate(self, model: nn.Module) -> None:
        target_list = list()
        prediction_list = list()
        cer_list = list()
        inference_result = {'target': target_list, 'prediction': prediction_list, 'cer': cer_list}
        total_distance = 0
        total_length = 0

        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                feature, target, feature_lengths, target_lengths = data

                feature = feature.to(self.device)
                feature_lengths = feature_lengths.to(self.device)

                result = target[:, 1:]

                y_hat = self.decoder.search(model, feature, feature_lengths, result)

                result = label_to_string(self.config.eval.eos_id, self.config.eval.blank_id, result[idx], self.id2char)
                y_hat = label_to_string(self.config.eval.eos_id, self.config.eval.blank_id, y_hat[idx], self.id2char)

                distance, length = get_distance(result, y_hat)

                total_distance += distance
                total_length += length

                cer = total_distance / total_length
                cer_list.append(cer)

                for idx in range(result.size(0)):
                    target_list.append(result[idx])
                    prediction_list.append(y_hat[idx])

                if batch_idx % self.config.train.print_interval == 0:
                    print('cer: {:.2f}'.format(cer))

        inference_result = pd.DataFrame(inference_result)
        inference_result.to_csv(self.config.eval.save_transcripts_path, index=False, encoding='cp949')


