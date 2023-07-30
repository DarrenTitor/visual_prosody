import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet, VisualEncoder
from .modules import VarianceAdaptor, VarianceAdaptorWithSpeaker
from utils.tools import get_mask_from_lengths


def compute_deltas_with_last_zero(tensor):
    shifted_tensor = torch.roll(tensor, shifts=-1, dims=-1)
    deltas = shifted_tensor - tensor
    # Set the last element in each row to 0
    deltas[..., -1] = 0
    return deltas


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)

        self.using_speaker_embeddings = model_config["transformer"]["using_speaker_embeddings"]
        if self.using_speaker_embeddings:
            print("=> Using speaker embeddings.")
            print("=> Using VarianceAdaptorWithSpeaker.")
            self.variance_adaptor = VarianceAdaptorWithSpeaker(preprocess_config, model_config)
        else:
            print("=> Not using speaker embeddings.")
            self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)

        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )


        self.speaker_layernorm = nn.LayerNorm(192)
        self.using_video_embeddings = model_config["transformer"]["using_video_embeddings"]


        if self.using_video_embeddings:
            print('=> Using VisualEncoder.')

            # self.ada_avg_pool_pitch = nn.AdaptiveAvgPool1d(model_config["transformer"]["prosody_vector_dim"] // 4)
            # self.ada_avg_pool_energy = nn.AdaptiveAvgPool1d(model_config["transformer"]["prosody_vector_dim"] // 4)
            # self.ada_max_pool_pitch = nn.AdaptiveMaxPool1d(model_config["transformer"]["prosody_vector_dim"] // 4)
            # self.ada_max_pool_energy = nn.AdaptiveMaxPool1d(model_config["transformer"]["prosody_vector_dim"] // 4)
            self.p_lstm = nn.LSTM(input_size=1, hidden_size=model_config["transformer"]["prosody_vector_dim"]//2, batch_first=True)
            self.e_lstm = nn.LSTM(input_size=1, hidden_size=model_config["transformer"]["prosody_vector_dim"]//2, batch_first=True)

            self.prosody_using_delta = model_config["transformer"]["prosody_using_delta"]

            self.visual_encoder = VisualEncoder(model_config)
            self.avgmax_layernorm = nn.LayerNorm(2*model_config["transformer"]["visual_encoder_hidden"])
            self.visual_fc = nn.Linear(2*model_config["transformer"]["visual_encoder_hidden"], model_config["transformer"]["decoder_hidden"])
            

        

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        ### new
        speaker_embeddings=None,
        video_embeddings=None,
        vid_lens=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks)
        ### use speechbrain speaker embedding, project from 192dim to 256dim
        # if speaker_embeddings is not None:
        # print('x.shape before variance_adaptor: ', output.shape)
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        if self.using_speaker_embeddings:
            (
                output,
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                mel_lens,
                mel_masks,
            ) = self.variance_adaptor(
                output,
                src_masks,
                mel_masks,
                max_mel_len,
                p_targets,
                e_targets,
                d_targets,
                p_control,
                e_control,
                d_control,
                speaker_embeddings=speaker_embeddings,
            )
        else:
            (
                output,
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                mel_lens,
                mel_masks,
            ) = self.variance_adaptor(
                output,
                src_masks,
                mel_masks,
                max_mel_len,
                p_targets,
                e_targets,
                d_targets,
                p_control,
                e_control,
                d_control,
            )
        # print('p_targets.shape: ', p_targets.shape)
        # print('p_predictions.shape: ', p_predictions.shape)
        # print('e_targets.shape: ', e_targets.shape)
        # print('e_predictions.shape: ', e_predictions.shape)
        # print('src_lens.shape: ', src_lens.shape)
        # print('src_lens: ', src_lens)


        if self.using_video_embeddings:
            assert video_embeddings is not None
            assert vid_lens is not None
            vid_max_len = vid_lens.max().item()
            vid_mask = get_mask_from_lengths(vid_lens, max_len=vid_max_len)

            if self.training:
                p_seqs = p_targets
                e_seqs = e_targets
            else:
                p_seqs = p_predictions
                e_seqs = e_predictions

            if self.prosody_using_delta:
                # print('delta')
                p_seqs = compute_deltas_with_last_zero(p_seqs)
                e_seqs = compute_deltas_with_last_zero(e_seqs)

            lstm_p_input = nn.utils.rnn.pack_padded_sequence(
                torch.unsqueeze(p_seqs, dim=2), 
                lengths=src_lens.cpu(), 
                batch_first=True,
                enforce_sorted=False,
            )
            lstm_e_input = nn.utils.rnn.pack_padded_sequence(
                torch.unsqueeze(e_seqs, dim=2), 
                lengths=src_lens.cpu(), 
                batch_first=True,
                enforce_sorted=False,
            )


            _, (lstm_p_hn, _) = self.p_lstm(lstm_p_input)
            _, (lstm_e_hn, _) = self.e_lstm(lstm_e_input)
            lstm_p_hn = torch.squeeze(lstm_p_hn)
            lstm_e_hn = torch.squeeze(lstm_e_hn)
            # print('lstm_p_hn.shape', lstm_p_hn.shape)
            # print('lstm_e_hn.shape', lstm_e_hn.shape)
            visual_query_vec = torch.cat([
                lstm_p_hn,
                lstm_e_hn,
            ], dim=1)




            # visual_query_vec = torch.cat([
            #     self.ada_avg_pool_energy(e_predictions),
            #     self.ada_avg_pool_pitch(p_predictions),
            #     self.ada_max_pool_energy(e_predictions),
            #     self.ada_max_pool_pitch(p_predictions),
            # ], dim=1)
            visual_out = self.visual_encoder(   
                src_seq=video_embeddings, 
                mask=vid_mask, 
                q_vec=visual_query_vec,
            )


            avg_pool_out = visual_out.sum(dim=1) / vid_lens.reshape(-1, 1)
            max_pool_out, _ = visual_out.max(dim=1)
            visual_out = torch.cat([avg_pool_out, max_pool_out], dim=1)
            visual_out = self.avgmax_layernorm(visual_out)

            visual_out = self.visual_fc(visual_out)

            # [batch_size, mel_len, hidden] broadcast+ [batch_size, 1, hidden]
            output = output + torch.unsqueeze(visual_out, dim=1)

            


        # print('shape of decoder input : ', output.shape)
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )