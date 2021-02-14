from torch import Tensor
from typing import Tuple, Optional, Any
import numpy as np
import pandas as pd
import Levenshtein as Lev


def load_dataset(dataset_path: str) -> Tuple[list, list, list, list]:
    audio_paths = list()
    transcripts = list()

    TRAIN_NUM = 410000
    VALID_NUM = 5279

    with open(dataset_path) as f:
        for line in f.readlines():
            audio_path, _, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    temp = list(zip(audio_paths, transcripts))
    np.random.shuffle(temp)
    audio_paths, transcripts = zip(*temp)

    train_audio_paths = audio_paths[:TRAIN_NUM]
    train_transcripts = transcripts[:TRAIN_NUM]

    valid_audio_paths = audio_paths[TRAIN_NUM:]
    valid_transcripts = transcripts[TRAIN_NUM:]

    return train_audio_paths, train_transcripts, valid_audio_paths, valid_transcripts


def load_label(label_path: str, blank_id: int) -> Tuple[dict, dict]:
    char2id = dict()
    id2char = dict()

    file = pd.read_csv(label_path, encoding='utf-8')

    id_list = file['id']
    char_list = file['char']

    for id, char in zip(id_list, char_list):
        char2id[char] = id
        id2char[id] = char

    char2id['<blank>'] = blank_id
    id2char[blank_id] = '<blank>'

    return char2id, id2char


def label_to_string(
        eos_id: int,
        blank_id: int,
        labels: Tensor,
        id2char: dict,
) -> str:
    sentence = str()

    for label in labels:
        if label.item() == eos_id:
            break
        if label.item() == blank_id:
            continue
        sentence += id2char[label.item()]

    return sentence

    # elif len(labels.shape) == 2:
    #     sentences = list()
    #     for label in labels:
    #         sentence = str()
    #         for id in label:
    #             if id.item() == eos_id:
    #                 break
    #             if id.item() == blank_id:
    #                 continue
    #             sentence += id2char[id.item()]
    #         sentences.append(sentence)
    #
    #     return sentences


def get_distance(
        eos_id: int,
        blank_id: int,
        targets: Tensor,
        y_hats: Tensor,
        id2char: dict,
) -> Tuple[int, int, dict]:
    transcripts = dict(target=list(), y_hat=list())
    total_distance = 0
    total_length = 0

    for (target, y_hat) in zip(targets, y_hats):
        target = label_to_string(eos_id, blank_id, target, id2char)
        y_hat = label_to_string(eos_id, blank_id, y_hat, id2char)

        transcripts['target'].append(target)
        transcripts['y_hat'].append(y_hat)

        target = target.replace(' ', '')
        y_hat = y_hat.replace(' ', '')

        distance = Lev.distance(y_hat, target)
        length = len(target.replace(' ', ''))

        total_distance += distance
        total_length += length

    return total_distance, total_length, transcripts

