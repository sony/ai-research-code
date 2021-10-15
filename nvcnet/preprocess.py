# Copyright 2021 Sony Group Corporation
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
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import librosa as lr
import numpy as np
from tqdm import tqdm


def process(p, args):
    r"""Read audio waveform and write it to the numpy format."""
    spk_id = args.spk_map[p.name]
    waves = list(p.glob('*.wav'))
    path = Path(args.outpath).joinpath('data')
    for wave in tqdm(waves):
        w, _ = lr.load(wave, sr=args.sr, mono=True)
        w, _ = lr.effects.trim(w, top_db=25, frame_length=256, hop_length=64)
        name = wave.with_suffix('.npz').name
        np.savez(path / name, wave=w, speaker_id=spk_id)
    return waves


def write_to_file(save_path, waves):
    r"""Write list of files to file."""
    with open(save_path, 'w') as writer:
        for n in waves:
            writer.write(f"{n.with_suffix('.npz').name}\n")


def run(args):
    path = Path(args.inpath)
    with open(args.spk_list) as f:
        spk_id = f.read().split('\n')
        assert len(spk_id) == len(set(spk_id)), 'speakers are not unique'
        args.spk_map = dict(zip(spk_id, range(len(spk_id))))

    save_path = Path(args.outpath)
    save_path.joinpath('data').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as executor:
        results = executor.map(
            process,
            [path / p for p in spk_id],
            repeat(args)
        )

    # split dataset
    rng = np.random.RandomState(args.seed)
    train, test = list(), list()
    for result in results:
        rng.permutation(result)
        n = int(len(result) * (0.9 if args.make_test else 1))
        train.extend(result[:n])
        test.extend(result[n:])

    write_to_file(save_path / "metadata_train.csv", train)
    if args.make_test:
        write_to_file(save_path / "metadata_test.csv", test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', '-i', type=str,
                        required=True, help="Path to input database.")
    parser.add_argument('--outpath', '-o', type=str,
                        required=True, help="Path to output database.")
    parser.add_argument('--spk_list', '-s', type=str,
                        required=True, help="File to a list of speakers.")
    parser.add_argument('--make-test', default=False, action="store_true",
                        help="Split for testing.")
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sampling rate.')
    parser.add_argument('--seed', type=int, default=123456,
                        help='Random seed.')

    args = parser.parse_args()
    run(args)
