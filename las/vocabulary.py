import pandas as pd
import Levenshtein as Lev


def load_label(args):
    char2id = dict()
    id2char = dict()
    file = pd.read_csv(args.label_path, encoding='utf-8')
    id_list = file['id']
    char_list = file['char']
    for id, char in zip(id_list, char_list):
        char2id[char] = id
        id2char[id] = char

    char2id['<blank>'] = args.blank_id
    id2char[args.blank_id] = '<blank>'

    return char2id, id2char


def label_to_string(args, labels, id2char):
    sentence = str()
    for label in labels:
        if label == args.eos_id:
            break
        if label == args.blank_id:
            continue
        sentence += id2char[label.item()]

    return sentence


def get_distance(args, target, y_hat, id2char):
    transcripts = dict(target=list, y_hat=list())
    total_distance = 0
    total_length = 0

    for idx in range(len(target)):
        target = label_to_string(args, target[idx], id2char)
        y_hat = label_to_string(args, y_hat[idx], id2char)

        transcripts['target'].append(target)
        transcripts['y_hat'].append(y_hat)

        target = target.replace(' ', '')
        y_hat = y_hat.replace(' ', '')

        length = len(target)
        distance = Lev.distance(target, y_hat)

        total_distance += distance
        total_length += length

    return total_distance, total_length, transcripts

