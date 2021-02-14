from model_builder import build_encoder, build_decoder, build_model
from trainer.trainer import train
from evaluator.evaluator import evaluate
from data.data_loader import SpectrogramDataset, BucketingSampler, AudioDataLoader
from vocabulary import load_label, load_dataset

import torch
import torch.optim as optim
import numpy as np
import argparse
import random


def main() -> None:
    parser = argparse.ArgumentParser(description='Listen, Attend and Spell')

    # Dataset
    parser.add_argument('--dataset_path', type=str, default='D:/dataset/transcripts.txt', help='path of dataset')
    parser.add_argument('--label_path', type=str, default='D:/label/aihub_labels.csv', help='path of character labels')
    parser.add_argument('--default_audio_path', type=str, default='E:/KsponSpeech', help='setting default audio path')
    parser.add_argument('--eval_audio_path', type=str, default='', help='path of inference audio')
    parser.add_argument('--eval_transcript_path', type=str, default='', help='path of inference transcripts')

    # Audio Config
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate (default: 16000)')
    parser.add_argument('--frame_length', type=int, default=.02, help='frame size for spectrogram (default: 0.020s)')
    parser.add_argument('--frame_stride', type=int, default=.01, help='frame stride for spectrogram (default: 0.010s)')
    parser.add_argument('--n_mel', type=int, default=80, help='number of mel filter (default: 80)')
    parser.add_argument('--extension', type=str, default='pcm', help='audio extension (default: pcm)')

    # Hyperparameters
    parser.add_argument('--bidirectional', action='store_true', default=True, help='if True, becomes a bidirectional encoder')
    parser.add_argument('--input_size', type=int, default=80, help='encoder input size (default: 80)')
    parser.add_argument('--num_vocabs', type=int, default=2000, help='the number of characters')
    parser.add_argument('--rnn_type', default='lstm', help='type of rnn cell: [gru, lstm, rnn] (default: lstm)')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of encoder layers (default: 3)')
    parser.add_argument('--encoder_hidden', type=int, default=256, help='hidden size of encoder (default: 256)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='number of decoder layers (default: 2)')
    parser.add_argument('--decoder_hidden', type=int, default=512, help='hidden size of decoder (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio in training (default: 0.3)')
    parser.add_argument('--ctc_weight', type=float, default=0.2, help='ctc weight ratio (default: 0.2)')
    parser.add_argument('--cross_entropy_weight', type=float, default=0.8, help='cross entropy weight ratio (default: 0.8)')
    parser.add_argument('--smoothing', type=bool, default=False, help='flag indication smoothing or not (default: False)')
    parser.add_argument('--attn_mechanism', type=str, default='location', help='option to specify the attention mechanism method: [location, scaled_dot]')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help='teacher forcing ratio in decoder (default: 1.0)')
    parser.add_argument('--max_len', type=int, default=110, help='maximum characters of sentence (default: 110)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training (default: 4)')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size in inference (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--epochs', type=int, default=11, help='number of max epochs in training (default: 11)')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate (default: 1e-04)')
    parser.add_argument('--pad_id', type=int, default=0, help='pad_id (default: 0)')
    parser.add_argument('--sos_id', type=int, default=1, help='sos_id (default: 1)')
    parser.add_argument('--eos_id', type=int, default=2, help='eos_id (default: 2)')
    parser.add_argument('--blank_id', type=int, default=1999, help='blank (default: 1999)')
    parser.add_argument('--use_joint_ctc_attention', type=bool, default=False, help='whether to use joint ctc attention (default: False)')
    parser.add_argument('--transcripts_print', type=bool, default=True, help='whether to print transcripts in inference (default: False)')

    # System
    parser.add_argument('--cuda', action='store_true', default=True, help='flag whether to be possible cuda training')
    parser.add_argument('--seed', type=int, default=22, help='random seed (default: 22)')
    parser.add_argument('--model_save_path', default='las_model', help='location to save model')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    char2id, id2char = load_label(args.label_path, args.blank_id)
    train_audio_paths, train_transcripts, valid_audio_paths, valid_transcripts = load_dataset(args.dataset_path)

    train_dataset = SpectrogramDataset(
        args.default_audio_path,
        train_audio_paths,
        train_transcripts,
        args.sampling_rate,
        args.n_mel,
        args.frame_length,
        args.frame_stride,
        args.extension,
        args.sos_id,
        args.eos_id
    )
    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    train_loader = AudioDataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers
    )

    valid_dataset = SpectrogramDataset(
        args.default_audio_path,
        valid_audio_paths,
        valid_transcripts,
        args.sampling_rate,
        args.n_mel,
        args.frame_length,
        args.frame_stride,
        args.extension,
        args.sos_id,
        args.eos_id
    )
    valid_sampler = BucketingSampler(valid_dataset, batch_size=args.batch_size)
    valid_loader = AudioDataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        num_workers=args.num_workers
    )

    encoder = build_encoder(args, device)
    decoder = build_decoder(args, device)

    model = build_model(encoder, decoder)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Start Train !!!')
    for epoch in range(1, args.epochs):
        train(args, model, device, train_loader, train_sampler, optimizer, epoch, id2char)
        evaluate(args, model, device, valid_loader, id2char)

        if epoch % 2 == 0:
            torch.save(model, args.model_save_path + str(epoch) + '.pt')


if __name__ == "__main__":
    main()
