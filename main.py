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
import torch.optim as optim
import numpy as np
import random
import os
import hydra

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from trainer.trainer import train
from model_builder import build_model
from data.data_loader import (
    SpectrogramDataset,
    BucketingSampler,
    AudioDataLoader,
)
from vocabulary import (
    load_label,
    load_dataset,
)
from data import MelSpectrogramConfig
from models.las import (
    ListenAttendSpellConfig,
    JointCTCAttentionLASConfig,
)
from models.deepspeech2 import DeepSpeech2Config
from trainer import (
    ListenAttendSpellTrainConfig,
    DeepSpeech2TrainConfig,
)


cs = ConfigStore.instance()
cs.store(group="audio", name="melspectrogram", node=MelSpectrogramConfig, package="audio")
cs.store(group="model", name="las", node=ListenAttendSpellConfig, package="model")
cs.store(group="model", name="joint_ctc_attention_las", node=JointCTCAttentionLASConfig, package="model")
cs.store(group="model", name="deepspeech2", node=DeepSpeech2Config, package="model")
cs.store(group="train", name="las_train", node=ListenAttendSpellTrainConfig, package="train")
cs.store(group="train", name="deepspeech2_train", node=DeepSpeech2TrainConfig, package="train")


@hydra.main(config_path='configs', config_name='train')
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)

    use_cuda = config.train.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    char2id, id2char = load_label(config.train.label_path, config.train.blank_id)
    train_audio_paths, train_transcripts, valid_audio_paths, valid_transcripts = load_dataset(config.train.dataset_path, config.train.mode)

    train_dataset = SpectrogramDataset(
        config.train.audio_path,
        train_audio_paths,
        train_transcripts,
        config.audio.sampling_rate,
        config.audio.n_mfcc if config.audio.feature_extraction == 'mfcc' else config.audio.n_mel,
        config.audio.frame_length,
        config.audio.frame_stride,
        config.audio.extension,
        config.audio.feature_extraction,
        config.audio.normalize,
        config.audio.spec_augment,
        config.audio.freq_mask_parameter,
        config.audio.num_time_mask,
        config.audio.num_freq_mask,
        config.train.sos_id,
        config.train.eos_id,
    )

    train_sampler = BucketingSampler(train_dataset, batch_size=config.train.batch_size)
    train_loader = AudioDataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.train.num_workers,
    )

    valid_dataset = SpectrogramDataset(
        config.train.audio_path,
        valid_audio_paths,
        valid_transcripts,
        config.audio.sampling_rate,
        config.audio.n_mfcc if config.audio.feature_extraction == 'mfcc' else config.audio.n_mel,
        config.audio.frame_length,
        config.audio.frame_stride,
        config.audio.extension,
        config.audio.feature_extraction,
        config.audio.normalize,
        config.audio.spec_augment,
        config.audio.freq_mask_parameter,
        config.audio.num_time_mask,
        config.audio.num_freq_mask,
        config.train.sos_id,
        config.train.eos_id,
    )
    valid_sampler = BucketingSampler(valid_dataset, batch_size=config.train.batch_size)
    valid_loader = AudioDataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        num_workers=config.train.num_workers,
    )

    model = build_model(config, device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.train.lr)

    print('Start Train !!!')
    for epoch in range(0, config.train.epochs):
        train(config, model, device, train_loader, valid_loader, train_sampler, optimizer, epoch, id2char)

    torch.save(model, os.path.join(os.getcwd(), config.train.model_save_path))


if __name__ == "__main__":
    main()
