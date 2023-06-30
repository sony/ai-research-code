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
#   https://github.com/ming024/FastSpeech2/tree/d4e79e/model/loss.py
# available under MIT License.

import torch
import torch.nn as nn
import torch.distributions as D

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        if model_config["tvcgmm"]["enabled"]:
            self.tvcgmm_enabled = True
            self.k = model_config["tvcgmm"]["mixture_model"]["k"]
            self.min_var = model_config["tvcgmm"]["mixture_model"]["min_var"]
        else:
            self.tvcgmm_enabled = False
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.SmoothL1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            value_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        elif self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = torch.log(duration_targets.float() + 1).masked_select(src_masks)

        # mel spectrogram value loss
        if self.tvcgmm_enabled:
            param_predictions = value_predictions.reshape(*mel_targets.shape, self.k, 10)[:, :mel_masks.shape[1]]
            # in practice we predict the scale_tril (lower triangular factor of the covariance matrix)
            # we predict the parameters for every t,f bin
            # at every bin we predict the joint distribution of t,f t+1,f and t,f+1
            # --> later in sampling we have overlap of one bin with the next time and the next freq bin
            scale_tril = torch.diag_embed(nn.functional.softplus(param_predictions[..., 4:7]) + self.min_var, offset=0)
            scale_tril += torch.diag_embed(param_predictions[..., 7:9], offset=-1)
            scale_tril += torch.diag_embed(param_predictions[..., 9:10], offset=-2)

            mix = D.Categorical(nn.functional.softmax(param_predictions[..., 0], dim=-1))
            comp = D.MultivariateNormal(param_predictions[..., 1:4], scale_tril=scale_tril)
            mixture = D.MixtureSameFamily(mix, comp)

            mel_multivariate_targets = torch.zeros([*mel_targets.shape, 3], device=mel_targets.device)
            mel_multivariate_targets[..., 0] = mel_targets # spectrogram
            mel_multivariate_targets[..., :-1, :, 1] = mel_targets[..., 1:, :] # t shifted spectrogram
            mel_multivariate_targets[..., :, :-1, 2] = mel_targets[..., :, 1:] # f shifted spectrogram
            
            value_loss = -mixture.log_prob(mel_multivariate_targets).masked_select(mel_masks.unsqueeze(-1)).mean()
        else:
            mel_predictions = value_predictions.masked_select(mel_masks.unsqueeze(-1))
            mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
            value_loss = self.mae_loss(mel_predictions, mel_targets)

        # variance predictor losses
        pitch_loss = self.mae_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            value_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            value_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
