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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from utils.audio import log_mel_spectrogram

from .module import Module
from .ops import res_block, wn_conv


class Classifier(Module):
    r"""Speaker identification classifier.

    Args:
        hp (HParams): Hyper-parameters used for the classifier.
    """

    def __init__(self, hp):
        self.hp = hp

    def call(self, x):
        hp = self.hp
        kernel, pad = (3,), (1,)

        with nn.parameter_scope('melspectrogram'):
            out = log_mel_spectrogram(x, hp.sr, 1024)

        with nn.parameter_scope('init_conv'):
            out = wn_conv(out, 32, kernel=kernel, pad=pad)

        for i in range(5):
            dim_out = min(out.shape[1] * 2, 512)
            out = res_block(
                out, dim_out,
                kernel=kernel, pad=pad,
                scope=f'downsample_{i}',
                training=self.training
            )

        with nn.parameter_scope('last_layer'):
            out = F.mean(out, axis=(2,))
            out = F.leaky_relu(out, 0.2)
            out = PF.affine(out, hp.n_speakers)

        return out
