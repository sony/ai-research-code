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
ArgumentParser for X-UMX/UMX Inference/Trainer
'''

import argparse


def get_train_args():
    parser = argparse.ArgumentParser(
        description='OpenUnmix_CrossNet(X-UMX)/OpenUnmix(UMX) Trainer')

    # Dataset paramaters
    parser.add_argument('--root', type=str,
                        help='root path of MUSDB18 dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='provide output path base folder name for saving the checkpoints')

    # which target do we want to train?
    parser.add_argument('--source', type=str, default='vocals',
                        help='target source to be trained for UMX')
    parser.add_argument('--sources', type=str, nargs='+',
                        default=['bass', 'drums', 'vocals', 'other'],
                        help='List of target sources to be trained for X-UMX')
    parser.add_argument('--umx-train', action='store_true', default=False,
                        help='If True, OpenUnmix(UMX) network is used for Training')

    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=140,
                        help='minimum number of bad epochs for EarlyStoping (default: 140 for UMX)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # Model Parameters
    parser.add_argument('--seq-dur', type=float, default=6.0,
                        help='Sequence duration in seconds per trainig batch'
                        'value of <=0.0 will use full/variable length')
    parser.add_argument('--unidirectional', action='store_true', default=False,
                        help='Use unidirectional LSTM instead of bidirectional')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=1024,
                        help='STFT hop size')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='hidden size parameter of dense bottleneck layers')
    parser.add_argument('--bandwidth', type=int, default=16000,
                        help='maximum model bandwidth in herz')
    parser.add_argument('--nb-channels', type=int, default=2,
                        help='set number of channels for model (1, 2)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='model sample rate')

    # Misc Parameters
    parser.add_argument('--mcoef', type=float, default=10.0,
                        help='coefficient for mixing: mfoef*TD-Loss + FD-Loss')
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension modules. ex) 'cpu', 'cudnn'.")

    # The duration of validation sample
    parser.add_argument('--valid_dur', type=float, default=10.0,
                        help='Split duration for validation sample to avoid GPU memory overflow')

    args, _ = parser.parse_known_args()

    return parser, args


def get_inference_args():
    parser = argparse.ArgumentParser(
        description='OpenUnmix_CrossNet(X-UMX)/OpenUnmix(UMX) Inference/Evaluation')

    parser.add_argument('--inputs', type=str, nargs='+',
                        help='List of paths to any audio files supported by FFMPEG.')
    parser.add_argument('--targets', nargs='+', default=['bass', 'drums', 'vocals', 'other'],
                        type=str, help='provide targets to be processed. \
                        If none, all available targets will be computed')
    parser.add_argument('--out-dir', type=str,
                        help='Path to save separated sources')
    parser.add_argument('--start', type=float, default=0.0,
                        help='Audio chunk start in seconds')
    parser.add_argument('--duration', type=float, default=-1.0,
                        help='Audio chunk duration in seconds, negative values load full track')
    parser.add_argument('--model', type=str, required=True,
                        help='path to pretrained weights (weight filename in case of X-UMX Inference or directory of weight files for UMX Inference')
    parser.add_argument('--context', default='cudnn', type=str,
                        help='Execution on CUDA')
    parser.add_argument('--softmask', dest='softmask', action='store_true',
                        help=('if enabled, will initialize separation with softmask.'
                              'otherwise, will use mixture phase with spectrogram'))
    parser.add_argument('--niter', type=int, default=1,
                        help='number of iterations for refining results.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='exponent in case of softmask separation')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='model sample rate')
    parser.add_argument('--residual-model', action='store_true',
                        help='create a model for the residual')
    parser.add_argument('--chunk-dur', type=int, default=None,
                        help='window length in seconds - reduce this if Inference fails with SegFault')
    parser.add_argument('--umx-infer', action='store_true', default=False,
                        help='If True, OpenUnmix(UMX) network is used for Inference/Evaluation')

    # Only Evaluation specific arguments
    parser.add_argument('--root', type=str,
                        help='root path of MUSDB18 dataset')
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--is-wav', action='store_true',
                        default=False, help='flags wav version of the dataset')
    args, _ = parser.parse_known_args()

    return args
