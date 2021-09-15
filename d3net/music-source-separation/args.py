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


def get_train_args():
    '''
    Get command line arguments.
    Arguments set the default values of command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description=f"Music Source Separation Trainer using D3Net")

    # which target do we want to train?
    parser.add_argument('--target', type=str, default='vocals',
                        help='target source (will be passed to the dataset)')

    # Dataset paramaters
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--output', type=str, default="d3net-mss",
                        help='provide output path base folder name')

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay')
    # Model Parameters
    parser.add_argument('--seq-dur', type=float, default=6.0,
                        help='Sequence duration in seconds per trainig batch'
                        'value of <=0.0 will use full/variable length')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=1024,
                        help='STFT hop size')

    # Misc Parameters
    parser.add_argument('--device-id', '-d', type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension modules. ex) 'cpu', 'cudnn'.")

    args, _ = parser.parse_known_args()

    return parser, args


def get_inference_args():
    '''
    Get command line arguments.
    Arguments set the default values of command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description=f"Music Source Separation Inference using D3Net")

    parser.add_argument('--inputs', '-i', nargs='+', type=str,
                        help='List of input audio files supported by FFMPEG.', required=True)
    parser.add_argument('--model-dir', '-m', type=str,
                        default='./d3net-mss/', help='path to the directory of pretrained models.')
    parser.add_argument('--targets', nargs='+', default=['vocals', 'drums', 'bass', 'other'],
                        type=str, help='provide targets to be processed. If none, all available targets will be computed')
    parser.add_argument('--out-dir', '-o', type=str,
                        default='./output/', help='output directory')
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension modules('cpu', 'cudnn')")
    return parser.parse_args()
