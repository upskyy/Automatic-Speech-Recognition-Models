import torch
import torch.nn as nn
from las.vocabulary import get_distance


def train(args, model, device, train_loader, train_sampler, optimizer, epoch, id2char):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        feature, target, feature_lengths, target_lengths = data

        result = target[:, 1:]

        feature = feature.to(device)
        target = target.to(device)
        feature_lengths = feature_lengths.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        encoder_output_prob, encoder_output_lens, decoder_output_prob = model(feature, feature_lengths, target, args.teacher_forcing_ratio)

        decoder_output_prob = decoder_output_prob.to(device)
        decoder_output_prob = decoder_output_prob[:, :result.size(1), :]
        y_hat = decoder_output_prob.max(2)[1]

        encoder_output_prob = encoder_output_prob.transpose(0, 1)
        ctc_loss = nn.CTCLoss(blank=args.blank_id)
        ctc_loss = ctc_loss(encoder_output_prob, target, feature_lengths, target_lengths)

        cross_entropy_loss = nn.CrossEntropyLoss()
        cross_entropy_loss = cross_entropy_loss(
            decoder_output_prob.view(-1, decoder_output_prob.size(2)).contiguous(),
            result.view(-1).contiguous()
        )

        loss = args.ctc_weight * ctc_loss + (1 - args.ctc_weight) * cross_entropy_loss

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        total_distance, total_length, _ = get_distance(args, result, y_hat, id2char)

        cer = (total_distance / total_length) * 100

        print('Epoch {epoch} : {batch_idx} / {total_idx}\t'
              'Loss : {loss:.4f}\t'
              'Cer : {cer:.2f}\t'.format(epoch=epoch, batch_idx=batch_idx, total_idx=len(train_sampler), loss=loss, cer=cer))

