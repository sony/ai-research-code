# Copyright 2023 Sony Group Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
######################################################################
#
# Implementation derived from
#   https://github.com/ming024/FastSpeech2/tree/d4e79e/model/fastspeech2.py
# available under MIT License.

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Encoder, Decoder
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)

        if model_config["tvcgmm"]["enabled"]:
            self.mel_linear = nn.Linear(
                model_config["transformer"]["decoder_hidden"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"] * model_config["tvcgmm"]["mixture_model"]["k"] * 10,
            )
        else:
            self.mel_linear = nn.Linear(
                model_config["transformer"]["decoder_hidden"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            )

        if model_config["multi_speaker"]:
            self.speaker_emb = nn.Embedding(
                preprocess_config["stats"]["n_speakers"],
                model_config["transformer"]["encoder_hidden"],
            )
        else:
            self.speaker_emb = None

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
        p_control=None,
        e_control=None,
        d_control=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        encoder_output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            encoder_output = encoder_output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            adaptor_output,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            encoder_output,
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

        decoder_output, mel_masks = self.decoder(adaptor_output, mel_masks)
        value_predictions = self.mel_linear(decoder_output) # mel or mixture parameters prediction

        return (
            value_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )