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

import argparse
from pathlib import Path
import random
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa.feature as lf
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

import sys
sys.path.append('model/')

UNIV_HIFIGAN_CKPT_PATH = 'model/hifigan/generator_universal.pth.tar'
FT_HIFIGAN_CKPT_PATH = 'model/hifigan/generator_LJSpeech_blur_finetuned.pth.tar' #smoothness-aware finetuned 
CONFIG_PATH = 'model/hifigan/config_v1.json' 

def wav_to_mel(wav, sr):
    return np.log(lf.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000, power=1.0).clip(min=1e-5))

def mel_to_wav(mel, sr):
    return lf.inverse.mel_to_audio(M=np.exp(mel), sr=sr, n_fft=1024, hop_length=256, win_length=1024, fmin=0.0, fmax=8000, power=1.0, n_iter=60)

def blur(mel, sigma=1.0, size=3, split_in=0, split_out=80):
    mel_blurred = mel.clone()
    mel_blurred[:,split_in:split_out] = gaussian_blur(mel, kernel_size=(size,size), sigma=sigma)[:,split_in:split_out]
    return mel_blurred

def sharpen(mel, strength=1.0, split_in=0, split_out=80):
    kernel = torch.tensor([-strength/2, strength, -strength/2], device=mel.device)[None, None, ...]
    details = F.conv1d(mel.permute(1,0,2), weight=kernel, padding=1).permute(1,0,2)
    mel_sharpened = mel.clone()
    mel_sharpened[:,split_in:split_out] = mel[:,split_in:split_out] + details[:,split_in:split_out]
    return mel_sharpened/(mel_sharpened.sum(dim=-1, keepdims=True)/mel.sum(dim=-1, keepdims=True))

def noise(mel, var=1.0, scale=0.5):
    mel_noisy = mel.clone() + scale*torch.distributions.Normal(0, var).sample(mel.shape).to(device=mel.device)
    return mel_noisy

def load_vocoder(vocoder_name, args):
    if vocoder_name == 'griffinlim':
        preprocess = lambda wav, sr: torch.from_numpy(wav_to_mel(wav, sr)).to(args.device).float()[None, ...]
        vocoder = lambda mel, sr: mel_to_wav(mel.cpu().numpy(), sr=sr)[0]
        return vocoder, preprocess
    
    if vocoder_name == 'hifigan':
        import hifigan
        
        HIFIGAN_CONFIG = hifigan.AttrDict(json.load(open(CONFIG_PATH)))
        vocoder_hifigan = hifigan.Generator(HIFIGAN_CONFIG)
        ckpt = torch.load(UNIV_HIFIGAN_CKPT_PATH, map_location=args.device)
        vocoder_hifigan.load_state_dict(ckpt['generator'])
        vocoder_hifigan.remove_weight_norm()
        vocoder_hifigan.eval().to(args.device)
        preprocess = lambda wav, sr: torch.from_numpy(wav_to_mel(wav, sr)).to(args.device).float().unsqueeze(0)
        vocoder = lambda mel, sr: vocoder_hifigan(mel).detach()[0, 0].cpu().numpy()
        return vocoder, preprocess
    
    if vocoder_name == 'hifigan-ft':
        import hifigan

        HIFIGAN_CONFIG = hifigan.AttrDict(json.load(open(CONFIG_PATH)))
        vocoder_hifigan = hifigan.Generator(HIFIGAN_CONFIG)
        ckpt = torch.load(FT_HIFIGAN_CKPT_PATH, map_location=args.device)
        vocoder_hifigan.load_state_dict(ckpt['generator'])
        vocoder_hifigan.remove_weight_norm()
        vocoder_hifigan.eval().to(args.device)
        preprocess = lambda wav, sr: torch.from_numpy(wav_to_mel(wav, sr)).to(args.device).float().unsqueeze(0)
        vocoder = lambda mel, sr: vocoder_hifigan(mel).detach()[0, 0].cpu().numpy()
        return vocoder, preprocess
    
    if vocoder_name == 'melgan':
        # sampling rate is 22050
        melgan = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan', 'multi_speaker', trust_repo=True)
        preprocess = lambda wav, sr: melgan(torch.from_numpy(wav).to(args.device).float().unsqueeze(0))
        vocoder = lambda mel, sr: melgan.inverse(mel)[0].cpu().numpy()
        return vocoder, preprocess
        
    if vocoder_name == 'cargan':
        import cargan
        gpu = None if args.device == 'cpu' else 0
        preprocess = lambda wav, sr: cargan.preprocess.mels.from_audio(torch.from_numpy(wav).float().unsqueeze(0), sample_rate=sr).to(args.device)
        vocoder = lambda mel, sr: cargan.from_features(mel, gpu=gpu)[0].cpu().numpy()
        return vocoder, preprocess
        
    if vocoder_name == 'wavernn':
        from torchaudio.pipelines import TACOTRON2_WAVERNN_PHONE_LJSPEECH
        wavernn = TACOTRON2_WAVERNN_PHONE_LJSPEECH.get_vocoder()
        wavernn.to(args.device)
        preprocess = lambda wav, sr: torch.from_numpy(np.log(lf.melspectrogram(y=wav, sr=sr, n_fft=2048, hop_length=275, win_length=1100, n_mels=80, fmin=40.0, fmax=11025, power=1.0).clip(min=1e-5))).to(args.device).float().unsqueeze(0)
        vocoder = lambda mel, sr: wavernn(mel)[0][0].cpu().numpy()
        return vocoder, preprocess
        
    if vocoder_name == 'waveglow':
        # model is trained with 22050 sampling rate
        waveglow = torch.hub.load('nvidia/DeepLearningExamples', 'nvidia_waveglow', pretrained=True, trust_repo=True, device=args.device)
        waveglow.eval().to(args.device)
        preprocess = lambda wav, sr: torch.from_numpy(wav_to_mel(wav, sr)).to(args.device).float().unsqueeze(0)
        vocoder = lambda mel, sr: waveglow.infer(mel)[0].cpu().numpy()
        return vocoder, preprocess

def main(args):
    
    random.seed(args.seed)
    
    gt_index = {path.stem:path for path in args.input.rglob('*.wav')}
    samples = random.sample([line.split('|')[0] for line in open(args.split, 'r')], args.n_samples)
    assert len(gt_index) != 0, 'Did not find speaker dir with wavs in input path'
    print(f'Found {len(gt_index)} wavs with ids like {next(iter(gt_index))}.')
    
    for vocoder_name in tqdm(args.vocoder, desc='vocoder', dynamic_ncols=True):
        vocoder, preprocess = load_vocoder(vocoder_name, args)
        
        gt_output = args.output / vocoder_name / 'gt'
        smooth_output = args.output / vocoder_name / 'smooth'
        sharp_output = args.output / vocoder_name / 'sharp'
        noisy_output = args.output / vocoder_name / 'noisy'
        gt_output.mkdir(parents=True, exist_ok=True)
        smooth_output.mkdir(parents=True, exist_ok=True)
        sharp_output.mkdir(parents=True, exist_ok=True)
        noisy_output.mkdir(parents=True, exist_ok=True)
        
        if args.overwrite:
            remaining_samples = samples
        else:
            remaining_samples = list(filter(lambda sample: not all((path / f'{sample}.wav').exists() for path in [gt_output, smooth_output, sharp_output, noisy_output]), samples))
        
        with (
            open(args.output / vocoder_name / 'timings.txt', 'a') as timing_file,
            torch.no_grad()
        ):   
            for sample in tqdm(remaining_samples, desc='sample', leave=False, dynamic_ncols=True):
                timings = [sample]
                gt_wav, gt_sr = sf.read(gt_index[sample])#, samplerate=22050) #all models here are fit to 22050

                start_time = time.time()
                gt_mel = preprocess(gt_wav, gt_sr)
                timings.append(str(time.time()-start_time))

                start_time = time.time()
                gt_reconstruction = vocoder(gt_mel, gt_sr)
                timings.append(str(time.time()-start_time))
                sf.write(file=(gt_output / f'{sample}.wav'), data=gt_reconstruction, samplerate=gt_sr, format='WAV')
                np.save(gt_output / f'{sample}_mel.npy', gt_mel)

                start_time = time.time()
                smooth_mel = blur(gt_mel)
                smooth_reconstruction = vocoder(smooth_mel, gt_sr)
                timings.append(str(time.time()-start_time))
                sf.write(file=(smooth_output / f'{sample}.wav'), data=smooth_reconstruction, samplerate=gt_sr, format='WAV')
                np.save(smooth_output / f'{sample}_mel.npy', smooth_mel)

                start_time = time.time()
                sharp_mel = sharpen(gt_mel)
                sharp_reconstruction = vocoder(sharp_mel, gt_sr)
                timings.append(str(time.time()-start_time))
                sf.write(file=(sharp_output / f'{sample}.wav'), data=sharp_reconstruction, samplerate=gt_sr, format='WAV')
                np.save(sharp_output / f'{sample}_mel.npy', sharp_mel)

                noisy_mel = noise(gt_mel)
                noisy_reconstruction = vocoder(noisy_mel, gt_sr)
                sf.write(file=(noisy_output / f'{sample}.wav'), data=noisy_reconstruction, samplerate=gt_sr, format='WAV')
                np.save(noisy_output / f'{sample}_mel.npy', noisy_mel.cpu())
                
                timing_file.write(','.join(timings)+'\n')
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export vocoder-reconstructed audio. Uses ground-truth audio, converts to vocoder-specific acoustic features, corrupts the features and saves vocoder reconstruction output.')
    parser.add_argument('--vocoder', default=['hifigan', 'hifigan-ft', 'melgan', 'cargan', 'griffinlim', 'waveglow', 'wavernn'], type=str, help='vocoder to use', choices=['hifigan', 'hifigan-ft', 'melgan', 'cargan', 'griffinlim', 'wavernn', 'waveglow'], nargs='+')
    parser.add_argument('--input', required=True, type=Path, help='path to gt audio data')
    parser.add_argument('--output', required=True, type=Path, help='output path')
    parser.add_argument('--split', required=True, type=Path, help='split file to select from')
    parser.add_argument('--overwrite', action='store_const', const=True, help='overwrite existing output files')
    parser.add_argument('--n_samples', type=int, default=512, help='how many samples to export')
    parser.add_argument('--seed', type=int, default=0, help='seed for random selection from dataset')
    parser.add_argument('--device', type=str, default='cuda', help='device to use', choices=['cpu', 'cuda'])

    args = parser.parse_args()
    main(args)