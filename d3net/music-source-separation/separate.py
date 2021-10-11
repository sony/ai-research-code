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
MSS Inference code using D3Net.
'''

import os
import argparse
import yaml
import numpy as np
import nnabla as nn
from nnabla.ext_utils import get_extension_context
from util import model_separate, save_stft_wav, generate_data
from filter import apply_mwf
from args import get_inference_args


def run_separation(args, fft_size=4096, hop_size=1024, n_channels=2, apply_mwf_flag=True, ch_flip_average=False):
    # Set NNabla extention
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)

    for i, input_file in enumerate(args.inputs):
        sample_rate, inp_stft = generate_data(
            input_file, fft_size, hop_size, n_channels)
        print(f"{i+1} / {len(args.inputs)}  : {input_file}")
        out_stfts = {}
        inp_stft_contiguous = np.abs(np.ascontiguousarray(inp_stft))

        for target in args.targets:
            # Load the model weights for corresponding target
            nn.load_parameters(f"{os.path.join(args.model_dir, target)}.h5")
            with open(f"./configs/{target}.yaml") as file:
                # Load target specific Hyper parameters
                hparams = yaml.load(file, Loader=yaml.FullLoader)
            with nn.parameter_scope(target):
                out_sep = model_separate(
                    inp_stft_contiguous, hparams, ch_flip_average=ch_flip_average)
                out_stfts[target] = out_sep * np.exp(1j * np.angle(inp_stft))

        if apply_mwf_flag:
            out_stfts = apply_mwf(out_stfts, inp_stft)

        sub_dir_name = ''
        output_subdir = args.out_dir + sub_dir_name
        output_subdir = os.path.join(
            output_subdir, os.path.splitext(os.path.basename(input_file))[0])

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        out = {}
        for target in args.targets:
            out[target] = save_stft_wav(out_stfts[target], hop_size, sample_rate, output_subdir + '/' +
                                        target + '.wav', samplewidth=2)


if __name__ == '__main__':

    run_separation(
        get_inference_args(),
        fft_size=4096,
        hop_size=1024,
        n_channels=2,
        apply_mwf_flag=True,
        ch_flip_average=True
    )
