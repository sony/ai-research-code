# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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
import test
import multiprocessing
import functools
import tqdm
import museval
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import musdb


def separate_and_evaluate(
    track,
    model,
    niter,
    alpha,
    softmask,
    output_dir,
    eval_dir,
):
    estimates = test.separate(
        audio=track.audio,
        model_path=model,
        niter=niter,
        alpha=alpha,
        softmask=softmask
    )

    if output_dir:
        mus.save_estimates(estimates, track, output_dir)

    scores = museval.eval_mus_track(
        track, estimates, output_dir=eval_dir
    )
    return scores


if __name__ == '__main__':
    # Evaluation settings
    parser = argparse.ArgumentParser(description='MUSDB18 Evaluation', add_help=False)
    parser.add_argument('--model', default='models/x-umx.h5', type=str, help='path to pretrained x-umx  model')
    parser.add_argument('--outdir', type=str,  help='Results path where audio evaluation results are stored')
    parser.add_argument('--evaldir', type=str, help='Results path for museval estimates are stored')
    parser.add_argument('--root', type=str, help='Path to MUSDB18')
    parser.add_argument('--subset', type=str, default='test', help='MUSDB subset (`train`/`test`)')
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--is-wav', action='store_true', default=False, help='flags wav version of the dataset')
    parser.add_argument('--context', default='cudnn', type=str, help='Execution on CUDA')
    parser.add_argument('--niter', type=int, default=1, help='number of iterations for refining results.')
    parser.add_argument('--alpha', type=float, default=1.0, help='exponent in case of softmask separation')
    parser.add_argument('--softmask', dest='softmask', action='store_true',
                        help=('if enabled, will initialize separation with softmask.'
                              'otherwise, will use mixture phase with spectrogram'))
    args, _ = parser.parse_known_args()

    # Set NNabla context and Dynamic graph execution
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

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
                    model=args.model,
                    niter=args.niter,
                    alpha=args.alpha,
                    softmask=args.softmask,
                    output_dir=args.outdir,
                    eval_dir=args.evaldir
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
                model=args.model,
                niter=args.niter,
                alpha=args.alpha,
                softmask=args.softmask,
                output_dir=args.outdir,
                eval_dir=args.evaldir
            )
            results.add_track(scores)

    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, 'x-umx')
    method.save('x-umx.pandas')
