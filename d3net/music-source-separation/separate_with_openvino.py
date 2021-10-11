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

import os
import argparse
import yaml
import numpy as np
from util import model_separate, save_stft_wav, generate_data
from filter import apply_mwf
from model_openvino import D3NetOpenVinoWrapper


def run_separation(args, fft_size=4096, hop_size=1024, n_channels=2, apply_mwf_flag=True, ch_flip_average=False):
    sources = ['vocals', 'bass', 'drums', 'other']

    for i, input_file in enumerate(args.inputs):
        sample_rate, inp_stft = generate_data(
            input_file, fft_size, hop_size, n_channels)
        print("%d / %d  : %s" % (i + 1, len(args.inputs), input_file))
        out_stfts = {}
        inp_stft_contiguous = np.abs(np.ascontiguousarray(inp_stft))
        for source in sources:
            with open('./configs/{}.yaml'.format(source)) as file:
                # Load source specific Hyper parameters
                hparams = yaml.load(file, Loader=yaml.FullLoader)
            d3netwrapper = D3NetOpenVinoWrapper(args, source)
            out_sep = model_separate(
                inp_stft_contiguous, hparams, ch_flip_average=ch_flip_average, openvino_wrapper=d3netwrapper)

            out_stfts[source] = out_sep * np.exp(1j * np.angle(inp_stft))

        if apply_mwf_flag:
            out_stfts = apply_mwf(out_stfts, inp_stft)

        sub_dir_name = ''
        output_subdir = args.out_dir + sub_dir_name
        output_subdir = os.path.join(
            output_subdir, os.path.splitext(os.path.basename(input_file))[0])

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        out = {}
        for source in sources:
            out[source] = save_stft_wav(out_stfts[source], hop_size, sample_rate, output_subdir + '/' +
                                        source + '.wav', samplewidth=2)


def get_args(description=''):
    '''
    Get command line arguments.
    Arguments set the default values of command line arguments.
    '''

    parser = argparse.ArgumentParser(description)

    parser.add_argument('--inputs', '-i', nargs='+', type=str,
                        help='List of input audio files supported by FFMPEG.', required=True)
    parser.add_argument('--out-dir', '-o', type=str,
                        default='output/', help='output directory')
    parser.add_argument('--model-dir', '-m', type=str,
                        default='./openvino_models', help='Path to openvino model folder')
    parser.add_argument('--cpu-number', '-n', type=str, default='4',
                        help='The number of threads that openvino should use')
    return parser.parse_args()


if __name__ == '__main__':

    run_separation(
        get_args(),
        fft_size=4096,
        hop_size=1024,
        n_channels=2,
        apply_mwf_flag=True,
        ch_flip_average=True
    )
