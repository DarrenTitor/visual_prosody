import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet, VisualEncoder
from .modules import VarianceAdaptor, VarianceAdaptorWithSpeaker
from utils.tools import get_mask_from_lengths


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

            self.ada_avg_pool_pitch = nn.AdaptiveAvgPool1d(model_config["transformer"]["prosody_vector_dim"] // 4)
            self.ada_avg_pool_energy = nn.AdaptiveAvgPool1d(model_config["transformer"]["prosody_vector_dim"] // 4)
            self.ada_max_pool_pitch = nn.AdaptiveMaxPool1d(model_config["transformer"]["prosody_vector_dim"] // 4)
            self.ada_max_pool_energy = nn.AdaptiveMaxPool1d(model_config["transformer"]["prosody_vector_dim"] // 4)
            
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
        if self.using_video_embeddings:
            assert video_embeddings is not None
            assert vid_lens is not None
            vid_max_len = vid_lens.max().item()
            vid_mask = get_mask_from_lengths(vid_lens, max_len=vid_max_len)

            visual_query_vec = torch.cat([
                self.ada_avg_pool_energy(e_predictions),
                self.ada_avg_pool_pitch(p_predictions),
                self.ada_max_pool_energy(e_predictions),
                self.ada_max_pool_pitch(p_predictions),
            ], dim=1)
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