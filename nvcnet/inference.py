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
import os

import librosa as lr
import nnabla as nn
import soundfile as sf
from nnabla.ext_utils import get_extension_context

from hparams import hparams as hp
from model.model import NVCNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--context', '-c', type=str, default='cudnn',
        help="Extension module. 'cudnn' is highly recommended.")
    parser.add_argument("--device-id", "-d", type=str, default='-1',
                        help='A list of device ids separated by comma.\
                        This is only valid if you specify `-c cudnn`.\
                        Defaults to use all available GPUs.')
    parser.add_argument("--model", "-m", type=str,
                        help='Path to the pretrained model.')
    parser.add_argument("--input", "-i", type=str, default=None,
                        help='Path to the input audio.')
    parser.add_argument("--reference", "-r", type=str, default=None,
                        help='Path to the reference audio.')
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to the converted audio file.")
    args = parser.parse_args()

    # setup context for nnabla
    if args.device_id != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    # setup the context
    ctx = get_extension_context(args.context, device_id='0')
    nn.set_default_context(ctx)

    hp.batch_size = 1
    model = NVCNet(hp)
    model.training = False
    model.load_parameters(args.model)

    x_audio = lr.load(args.input, sr=hp.sr)[0]  # read input utterance
    y_audio = lr.load(args.reference, sr=hp.sr)[0]  # read reference utterance

    x = nn.Variable.from_numpy_array(x_audio[None, None, ...])
    y = nn.Variable.from_numpy_array(y_audio[None, None, ...])
    out = model(x, y)

    out.forward(clear_buffer=True)
    sf.write(args.output, out.d[0, 0], samplerate=hp.sr)
