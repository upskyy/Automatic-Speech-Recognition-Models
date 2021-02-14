import torch
import torch.nn as nn
import torch.optim as optim
from vocabulary import get_distance
from data.data_loader import BucketingSampler, AudioDataLoader


def train(
        args,
        model: nn.Module,
        device: torch.device,
        train_loader: AudioDataLoader,
        train_sampler: BucketingSampler,
        optimizer: optim,
        epoch: int,
        id2char: dict,
) -> None:
    total_distance = 0
    total_length = 0
    ctcloss = nn.CTCLoss(blank=args.blank_id, reduction='mean', zero_infinity=True)
    crossentropyloss = nn.CrossEntropyLoss(ignore_index=args.pad_id, reduction='mean')

    model.train()

    for batch_idx, data in enumerate(train_loader):
        feature, target, feature_lengths, target_lengths = data

        feature = feature.to(device)
        target = target.to(device)
        feature_lengths = feature_lengths.to(device)
        target_lengths = target_lengths.to(device)

        result = target[:, 1:]

        optimizer.zero_grad()

        encoder_output_prob, encoder_output_lens, decoder_output_prob = model(feature, feature_lengths, target, args.teacher_forcing_ratio)

        decoder_output_prob = decoder_output_prob.to(device)
        decoder_output_prob = decoder_output_prob[:, :result.size(1), :]
        y_hat = decoder_output_prob.max(2)[1]  # (B, T)

        if args.use_joint_ctc_attention:
            encoder_output_prob = encoder_output_prob.transpose(0, 1)
            ctc_loss = ctcloss(encoder_output_prob, target, encoder_output_lens, target_lengths)

            cross_entropy_loss = crossentropyloss(
                decoder_output_prob.contiguous().view(-1, decoder_output_prob.size(2)),
                result.contiguous().view(-1)
            )

            loss = args.ctc_weight * ctc_loss + args.cross_entropy_weight * cross_entropy_loss
        else:
            loss = crossentropyloss(
                decoder_output_prob.contiguous().view(-1, decoder_output_prob.size(2)),
                result.contiguous().view(-1)
            )
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        distance, length, _ = get_distance(args.eos_id, args.blank_id, result, y_hat, id2char)
        total_distance += distance
        total_length += length
        cer = total_distance / total_length

        if args.use_joint_ctc_attention:
            if batch_idx % 10 == 0:
                print('Epoch {epoch} : {batch_idx} / {total_idx}\t\t'
                      'CTC loss : {ctc_loss:.4f}\t'
                      'Cross entropy loss : {cross_entropy_loss:.4f}\t\t'
                      'loss : {loss:.4f}\t'
                      'cer : {cer:.2f}\t'.format(epoch=epoch, batch_idx=batch_idx, total_idx=len(train_sampler), ctc_loss=ctc_loss, cross_entropy_loss=cross_entropy_loss, loss=loss, cer=cer))

        else:
            if batch_idx % 10 == 0:
                print('Epoch {epoch} : {batch_idx} / {total_idx}\t'
                      'loss : {loss:.4f}\t'
                      'cer : {cer:.2f}\t'.format(epoch=epoch, batch_idx=batch_idx, total_idx=len(train_sampler), loss=loss, cer=cer))
