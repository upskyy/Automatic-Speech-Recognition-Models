from las.option import option
from las.model_builder import build_encoder, build_decoder, build_model
from las.trainer.trainer import train
from las.evaluator.evaluator import evaluate
from las.data.data_loader import SpectrogramDataset, BucketingSampler, AudioDataLoader
from las.vocabulary import load_label
import torch
import torch.optim as optim
import numpy as np
import random


def main():
    args = option()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    char2id, id2char = load_label(args)

    train_dataset = SpectrogramDataset(
        args.audio_path,
        args.transcript_path,
        args.num_dataset,
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

    test_dataset = SpectrogramDataset(
        args.eval_audio_path,
        args.eval_transcript_path,
        args.eval_num_dataset,
        args.sampling_rate,
        args.n_mel,
        args.frame_length,
        args.frame_stride,
        args.extension,
        args.sos_id,
        args.eos_id
    )
    test_sampler = BucketingSampler(test_dataset, batch_size=args.eval_batch_size)
    test_loader = AudioDataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=args.num_workers
    )

    encoder = build_encoder(args)
    decoder = build_decoder(args, device)

    model = build_model(encoder, decoder)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(0, args.epochs):
        train(args, model, device, train_loader, train_sampler, optimizer, epoch, id2char)
        evaluate(args, model, device, test_loader, id2char)

    torch.save(model.state_dict(), args.model_save_path)


if __name__ == "__main__":
    main()
