import torch
import torch.nn as nn
from vocabulary import get_distance
from data.data_loader import AudioDataLoader


def evaluate(
        args,
        model: nn.Module,
        device: torch.device,
        test_loader: AudioDataLoader,
        id2char: dict,
) -> list:
    total_distance = 0
    total_length = 0
    total_loss = 0
    total_feature_length = 0
    transcripts_list = list()
    ctcloss = nn.CTCLoss(blank=args.blank_id, reduction='mean', zero_infinity=True)
    crossentropyloss = nn.CrossEntropyLoss(ignore_index=args.pad_id, reduction='mean')

    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            feature, target, feature_lengths, target_lengths = data

            feature = feature.to(device)
            target = target.to(device)
            feature_lengths = feature_lengths.to(device)
            target_lengths = target_lengths.to(device)

            result = target[:, 1:]

            encoder_output_prob, encoder_output_lens, decoder_output_prob = model(feature, feature_lengths, None, 0.0)

            decoder_output_prob = decoder_output_prob.to(device)
            decoder_output_prob = decoder_output_prob[:, :result.size(1), :]
            y_hat = decoder_output_prob.max(2)[1]

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

            distance, length, transcripts = get_distance(args.eos_id, args.blank_id, result, y_hat, id2char)
            if args.transcripts_print and batch_idx % 10000 == 0:
                transcripts_list.append(transcripts)

            total_loss += loss.item()
            total_feature_length += sum(feature_lengths).item()

            total_distance += distance
            total_length += length

    average_loss = total_loss / total_feature_length
    average_cer = total_distance / total_length
    print('Loss : {loss:.4f}\t'
          'Cer : {cer:.2f}\t'.format(loss=average_loss, cer=average_cer))
    print(transcripts_list)

    return transcripts_list

