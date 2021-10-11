# Copyright 2021 Sony Group Corporation.
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

'''
MSS evaluation code on MUSDB18 test dataset.
'''

import os
import argparse
import multiprocessing
import functools
import tqdm
import numpy as np
import yaml
import museval
import musdb
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import librosa
from filter import apply_mwf
from util import stft2time_domain, model_separate


def separate_and_evaluate(
    track,
    model_dir,
    targets,
    output_dir
):

    fft_size, hop_size, n_channels = 4096, 1024, 2
    audio = track.audio
    for i in range(audio.shape[1]):
        stft = librosa.stft(audio[:, i].flatten(),
                            n_fft=fft_size, hop_length=hop_size).transpose()
        if i == 0:
            data = np.ndarray(shape=(stft.shape[0], n_channels, fft_size // 2 + 1),
                              dtype=np.complex64)
        data[:, i, :] = stft

    if n_channels == 2 and audio.shape[1] == 1:
        data[:, 1] = data[:, 0]

    inp_stft = data

    out_stfts = {}
    inp_stft_contiguous = np.abs(np.ascontiguousarray(inp_stft))

    for target in targets:
        # Load the model weights for corresponding target
        nn.load_parameters(f"{os.path.join(model_dir, target)}.h5")
        with open(f"./configs/{target}.yaml") as file:
            # Load target specific Hyper parameters
            hparams = yaml.load(file, Loader=yaml.FullLoader)
        with nn.parameter_scope(target):
            out_sep = model_separate(
                inp_stft_contiguous, hparams, ch_flip_average=True)
            out_stfts[target] = out_sep * np.exp(1j * np.angle(inp_stft))

    out_stfts = apply_mwf(out_stfts, inp_stft)

    estimates = {}
    for target in targets:
        estimates[target] = stft2time_domain(out_stfts[target], hop_size)

    if output_dir:
        mus.save_estimates(estimates, track, output_dir)

    scores = museval.eval_mus_track(
        track, estimates, output_dir=output_dir
    )
    return scores


if __name__ == '__main__':
    # Evaluation settings
    parser = argparse.ArgumentParser(
        description='MUSDB18 Evaluation', add_help=False)
    parser.add_argument('--model-dir', '-m', type=str,
                        default='./d3net-mss', help='path to the directory of pretrained models.')
    parser.add_argument('--targets', nargs='+', default=['vocals', 'drums', 'bass', 'other'],
                        type=str, help='provide targets to be processed. If none, all available targets will be computed')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Path to save musdb estimates and museval results')
    parser.add_argument('--root', type=str, help='Path to MUSDB18')
    parser.add_argument('--subset', type=str, default='test',
                        help='MUSDB subset (`train`/`test`)')
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--is-wav', action='store_true',
                        default=False, help='flag: wav version of the dataset')
    parser.add_argument('--context', default='cudnn',
                        type=str, help='Execution on CUDA')
    args, _ = parser.parse_known_args()

    # Set NNabla context and Dynamic graph execution
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)

    mus = musdb.DB(
        root=args.root,
        download=args.root is None,
        subsets=args.subset,
        is_wav=args.is_wav
    )

    if args.cores > 1:
        pool = multiprocessing.Pool(args.cores)
        results = museval.EvalStore()
        scores_list = list(
            pool.imap_unordered(
                func=functools.partial(
                    separate_and_evaluate,
                    model_dir=args.model_dir,
                    targets=args.targets,
                    output_dir=args.out_dir
                ),
                iterable=mus.tracks,
                chunksize=1
            )
        )
        pool.close()
        pool.join()
        for scores in scores_list:
            results.add_track(scores)
    else:
        results = museval.EvalStore()
        for track in tqdm.tqdm(mus.tracks):
            scores = separate_and_evaluate(
                track=track,
                model_dir=args.model_dir,
                targets=args.targets,
                output_dir=args.out_dir
            )
            results.add_track(scores)

    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, args.model_dir)
    method.save(args.model_dir + '.pandas')
