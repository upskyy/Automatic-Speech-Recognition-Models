import argparse
import torch
import torch.nn as nn
import librosa
from torch import Tensor
from data.data_loader import load_audio_
from models.model import ListenAttendSpell
from vocabulary import label_to_string, load_label
from models.search import GreedySearch


def parse_audio(audio_path: str, audio_extension: str = 'pcm') -> Tensor:
    sound = load_audio_(audio_path, extension=audio_extension)

    melspectrogram = librosa.feature.melspectrogram(
        sound,
        sr=16000,
        n_mels=80,
        n_fft=320,
        hop_length=160
    )
    log_melspectrogram = librosa.amplitude_to_db(melspectrogram)
    log_melspectrogram = torch.FloatTensor(log_melspectrogram)

    return log_melspectrogram


parser = argparse.ArgumentParser(description='inference')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--audio_path', type=str, default='')
parser.add_argument('--label_path', type=str, default='')
parser.add_argument('--eos_id', type=int, default=2)
parser.add_argument('--blank_id', type=int, default=1999)
parser.add_argument('--device', type=bool, default=False)
args = parser.parse_args()

use_cuda = args.device and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

feature = parse_audio(args.audio_path)
input_length = torch.LongTensor([feature.size(1)])
target = torch.LongTensor(1, 120)
y_hats = None
char2id, id2char = load_label(args.label_path, args.blank_id)
greedy_search = GreedySearch(device)

model = torch.load(args.model_path, map_location=lambda storage, loc: storage).to(device)

if isinstance(model, nn.DataParallel):
    model = model.module

model.eval()

y_hats = greedy_search(model, feature.unsqueeze(0).to(device), input_length.to(device), target)
y_hats = y_hats.squeeze(0)

sentence = label_to_string(args.eos_id, args.blank_id, y_hats, id2char)

print(sentence)
