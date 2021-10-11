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

'''
MSS evaluation code on MUSDB18 test dataset.
'''

import test
import multiprocessing
import functools
import tqdm
import museval
import nnabla as nn
from nnabla.ext_utils import get_extension_context, import_extension_module
import musdb
from args import get_inference_args


def separate_and_evaluate(track, args, ext):
    estimates = test.separate(track.audio, args)

    if args.out_dir:
        mus.save_estimates(estimates, track, args.out_dir)

    scores = museval.eval_mus_track(
        track, estimates, output_dir=args.out_dir
    )
    # clear cache memory
    ext.clear_memory_cache()
    return scores


if __name__ == '__main__':
    # Get the arguments parser
    args = get_inference_args()

    # Set NNabla context and Dynamic graph execution
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)
    ext = import_extension_module(args.context)

    mus = musdb.DB(
        root=args.root,
        download=args.root is None,
        subsets='test',
        is_wav=args.is_wav
    )

    if args.cores > 1:
        pool = multiprocessing.Pool(args.cores)
        results = museval.EvalStore()
        scores_list = list(
            pool.imap_unordered(
                func=functools.partial(separate_and_evaluate, args, ext),
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
            scores = separate_and_evaluate(track, args, ext)
            results.add_track(scores)

    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, args.out_dir)
    method.save(args.out_dir + '.pandas')
