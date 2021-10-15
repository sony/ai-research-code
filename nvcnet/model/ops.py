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
from nnabla.initializer import NormalInitializer

from .module import Module


def wn_conv(*args, **kwargs):
    return PF.convolution(
        *args, **kwargs,
        apply_w=PF.weight_normalization,
        w_init=NormalInitializer(0.02)
    )


def wn_deconv(*args, **kwargs):
    return PF.deconvolution(
        *args, **kwargs,
        apply_w=PF.weight_normalization,
        w_init=NormalInitializer(0.02)
    )


def res_block(x, dim_out, kernel, pad, scope, training):
    r"""Residual block.
    Args:
        x (nn.Variable): Input variable.
        dim_out (int): Number of output channels.
        kernel (tuple of int): Kernel size.
        pad (tuple of int): Padding.
        scope (str): The scope.
    Returns:
        nn.Variable: Output variable.
    """
    dim_in = x.shape[1]
    with nn.parameter_scope(scope):
        with nn.parameter_scope('shortcut'):
            sc = x
            if dim_in != dim_out:
                sc = wn_conv(
                    sc, dim_out, (1,),
                    with_bias=False, name='conv'
                )
            sc = F.average_pooling(sc, kernel=(1, 2))
        with nn.parameter_scope('residual'):
            x = F.leaky_relu(x, 0.2)
            x = wn_conv(x, dim_in, kernel, pad, name='conv1')
            x = F.average_pooling(x, kernel=(1, 2))
            x = F.leaky_relu(x, 0.2, inplace=True)
            x = wn_conv(x, dim_out, (1,), name='conv2')
        return sc + x


class ResnetBlock(Module):
    def call(self, x, spk_emb, dilation):
        dim = x.shape[1]
        with nn.parameter_scope('shortcut'):
            s = wn_conv(x, dim, (1,))
        with nn.parameter_scope('block'):
            b = F.pad(x, (0, 0, dilation, dilation), 'reflect')
            b = wn_conv(b, 2 * dim, (3,), dilation=(dilation,), name='conv_1')
            if spk_emb is not None:
                b = b + wn_conv(spk_emb, 2 * dim, (1,), name="spk_emb")
            b = F.tanh(b[:, :dim, ...]) * F.sigmoid(b[:, dim:, ...])
            b = wn_conv(b, dim, (1,), dilation=(dilation,), name='conv_2')
        return s + b


class UpBlock(Module):
    def __init__(self, hp):
        self.hp = hp
        for i in range(hp.n_residual_layers):
            setattr(self, f"resblock_{i}", ResnetBlock())

    def call(self, x, spk_emb, r, mult):
        hp = self.hp
        with nn.parameter_scope("deconv"):
            x = F.gelu(x)
            x = wn_deconv(
                x, mult * hp.ngf // 2, (r * 2, ),
                stride=(r,), pad=(r // 2 + r % 2,),
            )

        for i in range(hp.n_residual_layers):
            x = getattr(self, f"resblock_{i}")(x, spk_emb, 3 ** i)

        return x


class DownBlock(Module):
    def __init__(self, hp):
        self.hp = hp
        for i in range(hp.n_residual_layers):
            setattr(self, f"resblock_{i}", ResnetBlock())

    def call(self, x, r, mult):
        hp = self.hp
        for i in range(hp.n_residual_layers):
            x = getattr(self, f"resblock_{i}")(x, None, 3 ** i)
        with nn.parameter_scope("conv"):
            x = F.gelu(x)
            x = wn_conv(
                x, mult * hp.ngf, (r * 2,),
                stride=(r,), pad=(r // 2 + r % 2,),
            )
        return x
