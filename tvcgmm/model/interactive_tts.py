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

import os
import argparse
import yaml
import json
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributions as D
from torchvision.transforms.functional import gaussian_blur

import utils.audio as Audio
from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device, expand, plot_mel
from utils.text import sequence_to_text, text_to_sequence
from synthesize import read_lexicon, preprocess_english

BASE_DIR = os.path.dirname(__file__)

class InteractiveTTS:
    
    def __init__(self, model_ckpt='libritts_tvcgmm_k5', vocoder_ckpt=None, step=40000, device='cpu'):

        self.device = device
        
        # load configs
        config = yaml.load(open(f'{BASE_DIR}/output/ckpt/{model_ckpt}/used_config.yaml', 'r'), Loader=yaml.FullLoader)
        self.preprocess_config = config['preprocess']
        self.model_config = config['model']
        self.train_config = config['train']

        path = self.train_config['path']
        path['ckpt_path'] = f'{BASE_DIR}/output/ckpt/{model_ckpt}'
        path['log_path'] = f'{BASE_DIR}/output/log/{model_ckpt}'
        path['result_path'] = f'{BASE_DIR}/output/result/{model_ckpt}'
        
        path = self.preprocess_config['path']
        path['lexicon_path'] = f'{BASE_DIR}/{path["lexicon_path"]}'
        path['preprocessed_path'] = f'{BASE_DIR}/{path["preprocessed_path"]}'
        path['raw_path'] = f'{BASE_DIR}/{path["raw_path"]}'
        
        self.model_config['vocoder']['path'] = f'{BASE_DIR}/hifigan/'
            
        # overwrite vocoder from config
        if vocoder_ckpt:
            self.model_config['vocoder']['speaker'] = vocoder_ckpt

        # init config
        configs = (self.preprocess_config, self.model_config, self.train_config)
        args = argparse.Namespace(restore_step=step, mode='single')

        # load lexicon
        self.lexicon = read_lexicon(self.preprocess_config['path']['lexicon_path'])
        # load model
        self.model = get_model(args, configs, self.device, train=False)
        print(f'loaded model checkpoint {model_ckpt}')
        # load vocoder
        self.vocoder = get_vocoder(self.model_config, self.device)
        print(f'loaded vocoder checkpoint {self.model_config["vocoder"]["speaker"]}')

    def synthesize(self, text, speaker_id=0, pitch=0, energy=0, duration=1.0, sharpen=False, blur=False, plot=False, conditional=False, alignment=None):
        ids = raw_texts = [text[:100]]
        speakers = np.array([speaker_id])

        if text[0] == '{' and text[-1] == '}':
            texts = np.array(text_to_sequence(text, self.preprocess_config["preprocessing"]["text"]["text_cleaners"]))
            word_pos = [0]
        else:
            texts, word_pos = preprocess_english(text, self.preprocess_config)
        texts = np.array([texts])
        phonemes = texts[0].tolist()
        text_lens = np.array([len(texts[0])])
        batch = to_device((ids, raw_texts, speakers, texts, text_lens, max(text_lens)), self.device)

        if type(pitch) is not list:
            pitch = [float(pitch)]*len(phonemes)
        if type(energy) is not list:
            energy = [float(energy)]*len(phonemes)
        if type(duration) is not list:
            duration = [float(duration)]*len(phonemes)

        pitch = torch.tensor(pitch, device=self.device)
        energy = torch.tensor(energy, device=self.device)
        duration = torch.tensor(duration, device=self.device)

        with torch.no_grad():
            predictions = self.model(
                *(batch[2:]),
                p_control=pitch[None, ...],
                e_control=energy[None, ...],
                d_control=duration,
                d_targets=alignment
            )

            src_len = predictions[7].item()
            mel_len = predictions[8].item()

            if self.model_config["tvcgmm"]["enabled"]:
                param_predictions = predictions[0].reshape(1, mel_len, self.preprocess_config["preprocessing"]["mel"]["n_mel_channels"], self.model_config["tvcgmm"]["mixture_model"]["k"], 10)[0, :mel_len]
                
                pis = F.softmax(param_predictions[..., 0], dim=-1)
                mus = param_predictions[..., 1:4]
                scale_trils = torch.diag_embed(F.softplus(param_predictions[..., 4:7]) + self.model_config["tvcgmm"]["mixture_model"]["min_var"], offset=0)
                scale_trils += torch.diag_embed(param_predictions[..., 7:9], offset=-1)
                scale_trils += torch.diag_embed(param_predictions[..., 9:10], offset=-2)
                sigmas = scale_trils @ scale_trils.transpose(-1, -2)
                sigmas[:, -1] = sigmas[:, -2] # last frequency bin is erroneous in conditional sampling due to missing training targets

                if not conditional: # naive sampling

                    mix = D.Categorical(pis)
                    comp = D.MultivariateNormal(mus, scale_tril=scale_trils)
                    mixture = D.MixtureSameFamily(mix, comp)

                    # picking max mode instead of sampling
                    # mode = pis.argmax(dim=-1, keepdims=True)[..., None].repeat(1, 1, 3, 1)
                    # mel_predictions = comp.sample().gather(index=mode, dim=-2).transpose(1, 2).reshape(-1,240).transpose(0, 1).unsqueeze(0)

                    mel_predictions = mixture.sample().transpose(1, 2).reshape(-1,240).transpose(0, 1).unsqueeze(0)
                    orig = mel_predictions.clone()
                    mel_predictions[:, :80, 1:] += mel_predictions[:, 80:160, :-1]
                    mel_predictions[:, 1:80, :] += mel_predictions[:, 160:-1, :]
                    mel_predictions[:, 1:80, 1:] /= 3
                    mel_predictions[:, 0, 1:] /= 2
                    mel_predictions[:, 1:, 0] /= 2
                    mel_predictions = mel_predictions[:, :80]

                else:
                    # need to correct numerical instability of LL^T ?
                    sigmas[(torch.linalg.eigvals(sigmas).abs() < 1e-3).any(dim=-1)] += torch.diag(torch.tensor([self.model_config["tvcgmm"]["mixture_model"]["min_var"]]*scale_trils.shape[-1], device=scale_trils.device))

                    mel_predictions = torch.zeros(param_predictions.shape[:2], device=param_predictions.device)
                    # we sample the first bivariate and then condition all other on their predecessor
                    mix = D.Categorical(pis[0])
                    comp = D.MultivariateNormal(mus[0], covariance_matrix=sigmas[0])
                    sample = D.MixtureSameFamily(mix, comp).sample().transpose(-1, -2)
                    mel_predictions[:2] = sample[:2] # time dimension
                    mel_predictions[0, 1:] = (mel_predictions[0, 1:] + sample[-1, :-1]) / 2 # freq dimension
                    for t in range(1, mel_len-1):
                        mu = mus[t, ..., 1:] + sigmas[t, ..., 1:, 0] * (1/sigmas[t, ..., 0, 0] * (mel_predictions[t].unsqueeze(-1) - mus[t, ..., 0])).unsqueeze(-1)
                        cov = sigmas[t, ..., 1:, 1:] - (sigmas[t, ..., 1:, :1] * 1/sigmas[t, ..., :1, :1]) @ sigmas[t, ..., :1, 1:]
                        marginal = D.Normal(mus[t, ..., 0], sigmas[t, ..., 0, 0])

                        pi = pis[t] * (torch.exp(marginal.log_prob(mel_predictions[t, :, None]))+1e-14)
                        pi = pi / pi.sum(dim=-1, keepdims=True)

                        mix = D.Categorical(pi)
                        comp = D.MultivariateNormal(mu, covariance_matrix=cov)
                        sample = D.MixtureSameFamily(mix, comp).sample().transpose(-1, -2)
                        mel_predictions[t+1] = sample[0] # cond. sampling time dimension
                        mel_predictions[t, 1:] = (mel_predictions[t, 1:] + sample[-1, :-1]) / 2 # naive sampling in freq dimension
                    mel_predictions = mel_predictions.unsqueeze(0).transpose(-1,-2)
                    orig = mel_predictions.clone()

            else:
                mel_predictions = predictions[0][:, :mel_len].detach().transpose(1, 2)
                orig = mel_predictions.clone()

            if plot:
                if self.preprocess_config["preprocessing"]["pitch"]["feature"] == 'phoneme_level':
                    duration = predictions[4][0, :src_len].detach().cpu().numpy()
                    pitch = predictions[1][0, :src_len].detach().cpu().numpy()
                    pitch = expand(pitch, duration)
                else:
                    pitch = predictions[1][0, :mel_len].detach().cpu().numpy()

                if self.preprocess_config["preprocessing"]["energy"]["feature"] == 'phoneme_level':
                    duration = predictions[4][0, :src_len].detach().cpu().numpy()
                    energy = predictions[2][0, :src_len].detach().cpu().numpy()
                    energy = expand(energy, duration)
                else:
                    energy = predictions[2][0, :mel_len].detach().cpu().numpy()

                stats = self.preprocess_config["stats"]["pitch"] + self.preprocess_config["stats"]["energy"][:2]
            
                fig = plot_mel([(mel_predictions[0].cpu().numpy(), pitch, energy)], stats, ["Synthetized Spectrogram"])
            else:
                fig = None

            # naive postprocessing
            if sharpen > 0:
                strength = sharpen
                kernel = torch.tensor([-strength/2, strength, -strength/2], device=self.device)[None, None, ...]
                details = F.conv1d(mel_predictions.permute(1,0,2), weight=kernel, padding=1).permute(1,0,2) # sharpen time dimension
                mel_sharpened = mel_predictions + details
                mel_predictions = mel_sharpened/(mel_sharpened.sum(dim=-1, keepdims=True)/mel_predictions.sum(dim=-1, keepdims=True))
            
            if blur:
                mel_predictions = gaussian_blur(mel_predictions, kernel_size=blur[0], sigma=blur[1])

            lengths = predictions[8] * self.preprocess_config['preprocessing']['stft']['hop_length']
            wav_predictions = vocoder_infer(mel_predictions, self.vocoder, self.model_config, self.preprocess_config, lengths=lengths)

            if self.model_config["tvcgmm"]["enabled"]:
                return wav_predictions[0], mel_predictions[0].cpu().numpy(), fig, (pis, mus, sigmas), orig, predictions[4]
            else:
                return wav_predictions[0], mel_predictions[0].cpu().numpy(), fig, orig, predictions[4]
