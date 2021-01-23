import argparse


def option():
    parser = argparse.ArgumentParser(description='LAS')

    # Dataset
    parser.add_argument('--label_path', type=str, default='./aihub_labels.csv', help='path of character labels')
    parser.add_argument('--audio_path', type=str, default='', help='path of audio')
    parser.add_argument('--transcript_path', type=str, default='', help='path of transcripts')
    parser.add_argument('--eval_audio_path', type=str, default='', help='path of inference audio')
    parser.add_argument('--eval_transcript_path', type=str, default='', help='path of inference transcripts')
    parser.add_argument('--num_dataset', type=int, default=610094, help='the number of train dataset')
    parser.add_argument('--eval_num_dataset', type=int, default=12451, help='the number of inference dataset')

    # Audio Config
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate (default: 16000)')
    parser.add_argument('--frame_length', type=int, default=.02, help='frame size for spectrogram (default: 0.020s)')
    parser.add_argument('--frame_stride', type=int, default=.01, help='frame stride for spectrogram (default: 0.010s)')
    parser.add_argument('--n_mel', type=int, default=80, help='number of mel filter (default: 80)')
    parser.add_argument('--extension', type=str, default='pcm', help='audio extension (default: pcm)')

    # Hyperparameters
    parser.add_argument('--bidirectional', action='store_true', default=True, help='if True, becomes a bidirectional encoder')
    parser.add_argument('--num_vocabs', type=int, default=2001, help='the number of characters')
    parser.add_argument('--rnn_type', default='lstm', help='type of rnn cell: [gru, lstm, rnn] (default: lstm)')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of encoder layers (default: 3)')
    parser.add_argument('--encoder_hidden', type=int, default=256, help='hidden size of encoder (default: 256)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='number of decoder layers (default: 2)')
    parser.add_argument('--decoder_hidden', type=int, default=512, help='hidden size of decoder (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio in training (default: 0.3)')
    parser.add_argument('--ctc_weight', type=float, default=0.2, help='ctc_weight ratio in training (default: 0.2)')
    parser.add_argument('--smoothing', type=float, default=0.1, help='ratio of label smoothing (default: 0.1)')
    parser.add_argument('--attn_mechanism', type=str, default='location', help='option to specify the attention mechanism method: [location, scaled_dot]')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help='teacher forcing ratio in decoder (default: 1.0)')
    parser.add_argument('--max_len', type=int, default=120, help='maximum characters of sentence (default: 120)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 32)')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size in inference (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--epochs', type=int, default=20, help='number of max epochs in training (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-06, help='learning rate (default: 1e-06)')
    parser.add_argument('--pad_id', type=int, default=0, help='pad_id (default: 0)')
    parser.add_argument('--sos_id', type=int, default=1, help='sos_id (default: 1)')
    parser.add_argument('--eos_id', type=int, default=2, help='eos_id (default: 2)')
    parser.add_argument('--blank_id', type=int, default=2000, help='blank (default: 2000)')
    parser.add_argument('--transcripts_print', type=bool, default=False, help='whether to print transcripts in inference (default: False)')

    # System
    parser.add_argument('--cuda', action='store_true', default=False, help='flag whether to be possible cuda training')
    parser.add_argument('--seed', type=int, default=22, help='random seed (default: 22)')
    parser.add_argument('--model_path', default='models/las_model.pt', help='location to save model')

    args = parser.parse_args()

    return args


