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

from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from utils.text import _clean_text


def prepare_align(config):
    in_dir = Path(config["path"]["corpus_path"])
    out_dir = Path(config["path"]["raw_path"])
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    out_dir.mkdir(exist_ok=True)

    for audio_path in tqdm(list(in_dir.glob('**/*.flac'))):

        speaker = audio_path.parts[-2]
        audio_filename = audio_path.stem
        txt_filename = '_'.join(audio_filename.split('_')[:2]) + '.txt'
        txt_path = in_dir / 'txt' / speaker / txt_filename

        if not txt_path.exists():
            print(f'{txt_path} not found')
            continue
            
        with open(txt_path) as f:
            text = f.readline().strip("\n")
        text = _clean_text(text, cleaners)

        speaker_out_dir = (out_dir / speaker)
        speaker_out_dir.mkdir(exist_ok=True)

        wav, _ = librosa.load(audio_path, sr=sampling_rate)
        wav = wav / max(abs(wav)) * max_wav_value
        wavfile.write(
            speaker_out_dir / (audio_filename + '.wav'),
            sampling_rate,
            wav.astype(np.int16),
        )
        with open(speaker_out_dir / (audio_filename + '.lab'), 'w') as f1:
            f1.write(text)