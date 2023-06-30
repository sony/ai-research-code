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
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tgt

def get_alignment(tier, sr=16000):
    sil_phones = ["sil", "sp", "spn", ""]
    
    start_time = None
    end_time = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if start_time is None:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            end_time = e

    return int(start_time*sr), int(end_time*sr)

def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    
    ref_index = {path.name:path for path in args.reference.rglob('*.wav')}
    if args.alignment:
        assert args.alignment.exists(), 'alignment path does not exist'
        align_index = {path.stem:path for path in args.alignment.rglob('*.TextGrid')}
    
    for metric in args.metric:
        if (args.output / f'{metric}.npy').exists() and not args.overwrite:
            print('output file exists, use --overwrite if intended')
            continue
        
        scores = []
        if metric == 'pesq':
            from pesq import pesq
            for deg_sample in tqdm(list(args.degraded.glob('*.wav')), dynamic_ncols=True):
                ref, sr = librosa.load(ref_index[deg_sample.name], sr=16000)
                deg, sr = librosa.load(deg_sample, sr=16000)
                if args.alignment:
                    start, end = get_alignment(tgt.io.read_textgrid(align_index[deg_sample.stem], include_empty_intervals=True).get_tier_by_name("phones"))
                    ref = ref[start:min(end, deg.shape[0]+start)]
                scores.append(pesq(ref=ref, deg=deg, fs=16000, mode='wb'))
        
        elif metric == 'stoi':
            from pystoi import stoi
            for deg_sample in tqdm(list(args.degraded.glob('*.wav')), dynamic_ncols=True):
                ref, sr = librosa.load(ref_index[deg_sample.name], sr=16000)
                deg, sr = librosa.load(deg_sample, sr=16000)
                scores.append(stoi(ref, deg, 16000))
                
        elif metric == 'visqol':
            from visqol import visqol_lib_py
            from visqol.pb2 import visqol_config_pb2
            from visqol.pb2 import similarity_result_pb2

            config = visqol_config_pb2.VisqolConfig()
            config.audio.sample_rate = 16000
            config.options.use_speech_scoring = True
            config.options.svr_model_path = str(Path(visqol_lib_py.__file__) / "model" / "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite")
            api = visqol_lib_py.VisqolApi()
            api.Create(config)
            for deg_sample in tqdm(list(args.degraded.glob('*.wav')), dynamic_ncols=True):
                ref, sr = librosa.load(ref_index[deg_sample.name], sr=16000)
                deg, sr = librosa.load(deg_sample, sr=16000)
                if args.alignment:
                    start, end = get_alignment(tgt.io.read_textgrid(align_index[deg_sample.stem], include_empty_intervals=True).get_tier_by_name("phones"))
                    ref = ref[start:min(end, deg.shape[0]+start)]
                score = api.Measure(ref.astype('float64'), deg.astype('float64'))
                scores.append(score.moslqo)
        
        elif metric == 'cdpam':
            import cdpam
            import torch
            with torch.no_grad():
                loss_fn = cdpam.CDPAM(dev=args.device)
                for deg_sample in tqdm(list(args.degraded.glob('*.wav')), dynamic_ncols=True):
                    ref = cdpam.load_audio(ref_index[deg_sample.name])
                    deg = cdpam.load_audio(deg_sample)
                    if args.alignment:
                        start, end = get_alignment(tgt.io.read_textgrid(align_index[deg_sample.stem], include_empty_intervals=True).get_tier_by_name("phones"), 22050)
                        ref = ref[:, start:min(end, deg.shape[1]+start)]
                        
                    scores.append(loss_fn.forward(ref, deg).cpu().item())
            
        elif metric == 'varl':
            #import cv2            
            import torch
            kernel = 1/6 * np.array([[0, -1, 0], [-1, 4, -1],[0, -1, 0]])
            kernel = torch.from_numpy(kernel).float()[None, None]
            def laplacian_var(mel):
                filtered = torch.nn.functional.conv2d(torch.from_numpy(mel).float()[None], weight=kernel)[0]
                return filtered.numpy(), filtered.var().item()

            for sample in tqdm(list(args.reference.glob('*.npy')), dynamic_ncols=True):
                sample_mel = np.load(sample)
                if len(sample_mel.shape) == 3:
                    sample_mel = sample_mel[0]
                img = ((sample_mel-sample_mel.min())/(sample_mel.max()-sample_mel.min()))
                
                #vol = cv2.Laplacian(img, cv2.CV_64F).var()
                _, vol = laplacian_var(img)
                scores.append(vol)

        np.save(args.output / f'{metric}.npy', scores)

            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics between reference and target audio.')
    parser.add_argument('--metric', type=str, required=True, help='metrics to calculate', nargs='+', choices=['pesq', 'stoi', 'cdpam', 'varl', 'visqol'])
    parser.add_argument('--reference', required=True, type=Path, help='path to reference wavs')
    parser.add_argument('--degraded', type=Path, help='path to degraded wav to be compared to reference with same name prefix')
    parser.add_argument('--alignment', type=Path, default=None, help='path to alignment textgrids')
    parser.add_argument('--overwrite', action='store_const', const=True, help='overwrite output')
    parser.add_argument('--output', required=True, type=Path, help='output path')
    parser.add_argument('--device', type=str, default='cuda', help='device to use', choices=['cpu', 'cuda'])

    args = parser.parse_args()
    main(args)