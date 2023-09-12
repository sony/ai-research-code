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
import random
import torch
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm

import sys
sys.path.append('model/')
import interactive_tts

def main(args):
    
    print('ensure you have checked out the correct git branch!')
    args.output.mkdir(parents=True, exist_ok=True)
    
    random.seed(args.seed)
    
    samples = random.sample([line.strip().split('|') for line in open(args.input / f'{args.split}.txt', 'r')], args.n_samples)
    speakers = json.load(open(args.input / 'speakers.json', 'r'))
    
    for checkpoint_name in tqdm(args.checkpoint, desc='checkpoint', dynamic_ncols=True):
        model = interactive_tts.InteractiveTTS(model_ckpt=checkpoint_name, vocoder_ckpt=args.vocoder_ckpt, step=args.step, device=args.device)
        output = args.output / checkpoint_name
        output.mkdir(exist_ok=True)
        
        for sample in tqdm(samples, desc='sample', leave=False, dynamic_ncols=True):
            sample, speaker, phonemes, raw_text = sample
            
            if args.gt_alignment:
                gt_alignment = torch.tensor(np.load(args.input / 'duration' / f'{speaker}-duration-{sample}.npy'), device=args.device).unsqueeze(0)
                text = phonemes # durations only work with exactly the same phoneme sequence as MFA used for alignment
            else:
                gt_alignment = None
                text = raw_text
            extra_args = {}
            if args.conditional:
                extra_args['conditional'] = True
            wav, mel, *rest = model.synthesize(text, pitch=args.pitch, energy=args.energy, duration=args.duration, alignment=gt_alignment, speaker_id=speakers[speaker], **extra_args)
            
            sf.write(file=(output / f'{sample}.wav'), data=wav, samplerate=22050, format='WAV')
            np.save(output / f'{sample}_mel.npy', mel)

            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export synthesized audio from TTS.')
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint to use', nargs='+')
    parser.add_argument('--vocoder_ckpt', type=str, required=True, help='vocoder checkpoint to use', choices=['generator_ljspeech.pth.tar', 'generator_universal.pth.tar'])
    parser.add_argument('--step', type=int, default=40000, help='checkpoint training step to use')
    parser.add_argument('--input', required=True, type=Path, help='path to fs2 preprocessed dataset')
    parser.add_argument('--output', required=True, type=Path, help='output path')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'train'], help='split to select samples from')
    parser.add_argument('--gt_alignment', action='store_const', const=True, help='use ground-truth alignment from input dir')
    parser.add_argument('--conditional', action='store_const', const=True, help='use conditional sampling (if TVC-GMM checkpoint)')
    parser.add_argument('--pitch', type=float, default=0, help='pitch control baseline for full utterance')
    parser.add_argument('--energy', type=float, default=0, help='energy control baseline for full utterance')
    parser.add_argument('--duration', type=float, default=1.0, help='duration control factor for full utterance')
    parser.add_argument('--n_samples', type=int, default=512, help='how many samples to export')
    parser.add_argument('--seed', type=int, default=0, help='seed for random selection from dataset')
    parser.add_argument('--device', type=str, default='cuda', help='device to use', choices=['cpu', 'cuda'])

    args = parser.parse_args()
    main(args)
