import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor, VarianceAdaptorWithSpeaker
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        # self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        ### new
        self.using_speaker_embeddings = model_config["transformer"]["using_speaker_embeddings"]
        if self.using_speaker_embeddings:
            print("=> Using speaker embeddings.")
            self.speaker_embeddings_method = model_config["transformer"]["speaker_embeddings_method"]
        else:
            print("=> Not using speaker embeddings.")
            self.speaker_embeddings_method = None
        print(self.using_speaker_embeddings)
        print(self.speaker_embeddings_method)
        if self.using_speaker_embeddings and self.speaker_embeddings_method == 2:
            self.variance_adaptor = VarianceAdaptorWithSpeaker(preprocess_config, model_config)
            print("=> Using VarianceAdaptorWithSpeaker.")
        else:
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
        ### new
        # self.speaker_embedding_projector = nn.Linear(192, model_config["transformer"]["encoder_hidden"], bias=False)
        # self.speaker_embedding_padding_size = (0, model_config["transformer"]["encoder_hidden"] - 192)
        self.speaker_layernorm = nn.LayerNorm(192)
        

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

        # default speaker embedding, useless
        # if self.speaker_emb is not None:
        #     output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
        #         -1, max_src_len, -1
            # )

        # print('=====')
        # print('output.shape: ', output.shape)
        # print('speaker_embeddings.shape:', speaker_embeddings.shape)
        # print('max_src_len:', max_src_len)
        # print('=====')
        # =====
        # output.shape:  torch.Size([24, 96, 256])
        # speaker_embeddings.shape: torch.Size([24, 192])
        # max_src_len: 96
        # =====

        # =====
        # print('output range: ', output.min(), output.max())
        # print('speaker_embeddings range: ', speaker_embeddings.min(), speaker_embeddings.max())
        # =====




        ### use speechbrain speaker embedding, project from 192dim to 256dim
        # if speaker_embeddings is not None:
        if self.using_speaker_embeddings:
            # print('=> Using speaker embedding')
            # print('speaker_embeddings.dtype: ', speaker_embeddings.dtype)

            ### try1: 
            # projected_embeddings = self.speaker_embedding_projector(speaker_embeddings)
            # expanded_embeddings = projected_embeddings.unsqueeze(1).expand(-1, max_src_len, -1)
            # output = output + expanded_embeddings

            ### try2:
            # padded_embeddings = F.pad(speaker_embeddings, self.speaker_embedding_padding_size, value=0)
            # expanded_embeddings = padded_embeddings.unsqueeze(1).expand(-1, max_src_len, -1) 
            # output = output + expanded_embeddings

            ### try3: 
            # expanded_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, max_src_len, -1) 
            # output = torch.cat((output, expanded_embeddings), dim=2)
            # print('output.shape after concat : ', output.shape)

            ### try4: for debugging
            # expanded_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, max_src_len, -1) 
            # zero_embeddings = torch.zeros_like(expanded_embeddings)
            # output = torch.cat((output, zero_embeddings), dim=2)

            if self.speaker_embeddings_method == 0:
            ### try5: layernorm + concat
            # print('=====')
            # print('ori_embeddings_shape', speaker_embeddings.shape)
            # print('ori_embeddings range: ', speaker_embeddings.min(), speaker_embeddings.max())

                speaker_embeddings = self.speaker_layernorm(speaker_embeddings)

            # print('normalized_embeddings_shape', speaker_embeddings.shape)
            # print('normalized_embeddings_shape range: ', speaker_embeddings.min(), speaker_embeddings.max())
            # print('=====')

                expanded_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, max_src_len, -1) 
                output = torch.cat((output, expanded_embeddings), dim=2)

            # ====
            # ori_embeddings_shape torch.Size([4, 192])
            # ori_embeddings range:  tensor(-68.2364, device='cuda:0') tensor(71.6695, device='cuda:0')
            # normalized_embeddings_shape torch.Size([4, 192])
            # normalized_embeddings_shape range:  tensor(-3.0481, device='cuda:0', grad_fn=<MinBackward1>) tensor(2.6331, device='cuda:0', grad_fn=<MaxBackward1>)
            # =====

            elif self.speaker_embeddings_method == 1:
            ### try6: layernorm + add
                speaker_embeddings = self.speaker_layernorm(speaker_embeddings)
                expanded_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, max_src_len, -1) 
                output = output + expanded_embeddings
            elif self.speaker_embeddings_method == 2:
                pass
            
            else:
                print('Actually not using speaker embedding. Double check model_config["transformer"]["speaker_embeddings_method"].')



            # print('expanded_embeddings.shape: ', expanded_embeddings.shape)
            # output = output + expanded_embeddings
        if self.using_speaker_embeddings and self.speaker_embeddings_method == 2:
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