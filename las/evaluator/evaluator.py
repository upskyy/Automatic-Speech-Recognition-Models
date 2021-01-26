import torch
import torch.nn as nn
from las.vocabulary import get_distance


def evaluate(args, model, device, test_loader, id2char):
    total_loss = 0
    total_length = 0
    total_distance = 0
    total_seq_length = 0
    transcripts_list = list()
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            feature, target, feature_lengths, target_lengths = data

            result = target[:, 1:]

            feature = feature.to(device)
            target = target.to(device)
            feature_lengths = feature_lengths.to(device)
            target_lengths = target_lengths.to(device)

            encoder_output_prob, encoder_output_lens, decoder_output_prob = model(feature, feature_lengths, None, 0.0)

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

            distance, length, transcripts = get_distance(args, result, y_hat, id2char)
            if args.transcripts_print:
                transcripts_list.append(transcripts)

            total_loss += loss.item()
            total_length += sum(feature_lengths).item()

            total_distance += distance
            total_seq_length += length

    average_loss = total_loss / total_length
    average_cer = (total_distance / total_seq_length) * 100
    print('Loss : {loss:.4f}\t'
          'Cer : {cer:.2f}\t'.format(loss=average_loss, cer=average_cer))

    return transcripts_list

