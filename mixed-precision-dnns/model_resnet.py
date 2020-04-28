# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

"""
ResNet primitives and full network models.
"""
import re
import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import parametric_quantization as PQ
import nnabla.functions as F
from nnabla.parameter import get_parameter_or_create
from nnabla.initializer import ConstantInitializer

from nnabla.logger import logger
from functools import partial


def find_delta(w, bw):
    """ Finds optimal quantization step size for FP quantization """
    maxabs_w = np.max(np.abs(w.d)) + np.finfo(np.float32).eps

    if bw > 4:
        return 2**(np.ceil(np.log2(maxabs_w/(2**(bw-1)-1))))
    else:
        return 2**(np.floor(np.log2(maxabs_w/(2**(bw-1)-1))))


def get_quantizers(cfg, test, pname, with_bias=True):
    """ determine quantization functions """

    if cfg.w_quantize in ['fp',
                          'parametric_fp_b_xmax', 'parametric_fp_d_xmax', 'parametric_fp_d_b',
                          'pow2',
                          'parametric_pow2_b_xmax', 'parametric_pow2_b_xmin', 'parametric_pow2_xmin_xmax']:
        # set delta to weights range
        if pname in nn.get_parameters():
            delta = find_delta(nn.get_parameters()[pname], cfg.w_bitwidth)
        else:
            delta = cfg.w_stepsize

        xmax = delta * (2 ** (cfg.w_bitwidth - 1) - 1)
        if 'pow2' in cfg.w_quantize:
            xmax = 2. ** np.round(np.log2(xmax))
            xmin = xmax / 2. ** (2. ** (cfg.w_bitwidth-1) - 1)

            xmin = np.clip(xmin, cfg.w_xmin_min + 1e-5, cfg.w_xmin_max - 1e-5)
            xmax = np.clip(xmax, cfg.w_xmax_min + 1e-5, cfg.w_xmax_max - 1e-5)

        if not test:
            print(f'Quantized affine/conv initialized to delta={delta}, xmax={xmax}')

    quantization_b = None
    if cfg.w_quantize == 'fp':
        quantization_w = partial(F.fixed_point_quantize, sign=True,
                                 n=cfg.w_bitwidth, delta=delta)
        quantization_b = partial(F.fixed_point_quantize, sign=True,
                                 n=cfg.w_bitwidth, delta=delta)
    elif cfg.w_quantize == 'parametric_fp_b_xmax':
        quantization_w = partial(PQ.parametric_fixed_point_quantize_b_xmax, sign=True,
                                 n_init=cfg.w_bitwidth,
                                 n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                 xmax_init=xmax,
                                 xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                 name=re.sub('quantized_[^/]*/W$', 'Wquant', pname))
        if with_bias:
            quantization_b = partial(PQ.parametric_fixed_point_quantize_b_xmax, sign=True,
                                     n_init=cfg.w_bitwidth,
                                     n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                     xmax_init=xmax,
                                     xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                     name=re.sub('quantized_[^/]*/W$', 'bquant', pname))
    elif cfg.w_quantize == 'parametric_fp_d_xmax':
        quantization_w = partial(PQ.parametric_fixed_point_quantize_d_xmax, sign=True,
                                 d_init=delta,
                                 d_min=cfg.w_stepsize_min, d_max=cfg.w_stepsize_max,
                                 xmax_init=xmax,
                                 xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                 name=re.sub('quantized_[^/]*/W$', 'Wquant', pname))
        if with_bias:
            quantization_b = partial(PQ.parametric_fixed_point_quantize_d_xmax, sign=True,
                                     d_init=delta,
                                     d_min=cfg.w_stepsize_min, d_max=cfg.w_stepsize_max,
                                     xmax_init=xmax,
                                     xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                     name=re.sub('quantized_[^/]*/W$', 'bquant', pname))
    elif cfg.w_quantize == 'parametric_fp_d_b':
        quantization_w = partial(PQ.parametric_fixed_point_quantize_d_b, sign=True,
                                 n_init=cfg.w_bitwidth,
                                 n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                 d_init=delta,
                                 d_min=cfg.w_stepsize_min, d_max=cfg.w_stepsize_max,
                                 name=re.sub('quantized_[^/]*/W$', 'Wquant', pname))
        if with_bias:
            quantization_b = partial(PQ.parametric_fixed_point_quantize_d_b, sign=True,
                                     n_init=cfg.w_bitwidth,
                                     n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                     d_init=delta,
                                     d_min=cfg.w_stepsize_min, d_max=cfg.w_stepsize_max,
                                     name=re.sub('quantized_[^/]*/W$', 'bquant', pname))
    elif cfg.w_quantize == 'pow2':
        quantization_w = partial(F.pow2_quantize, sign=True, with_zero=False,
                                 n=cfg.w_bitwidth, m=np.round(np.log2(xmax)))
        if with_bias:
            quantization_b = partial(F.pow2_quantize, sign=True, with_zero=False,
                                     n=cfg.w_bitwidth, m=np.round(np.log2(xmax)))
    elif cfg.w_quantize == 'parametric_pow2_b_xmax':
        quantization_w = partial(PQ.parametric_pow2_quantize_b_xmax, sign=True, with_zero=False,
                                 n_init=cfg.w_bitwidth,
                                 n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                 xmax_init=xmax,
                                 xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                 name=re.sub('quantized_[^/]*/W$', 'Wquant', pname))
        if with_bias:
            quantization_b = partial(PQ.parametric_pow2_quantize_b_xmax, sign=True, with_zero=False,
                                     n_init=cfg.w_bitwidth,
                                     n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                     xmax_init=xmax,
                                     xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                     name=re.sub('quantized_[^/]*/W$', 'bquant', pname))
    elif cfg.w_quantize == 'parametric_pow2_b_xmin':
        quantization_w = partial(PQ.parametric_pow2_quantize_b_xmin, sign=True, with_zero=False,
                                 n_init=cfg.w_bitwidth,
                                 n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                 xmin_init=xmin,
                                 xmin_min=cfg.w_xmin_min, xmin_max=cfg.w_xmin_max,
                                 name=re.sub('quantized_[^/]*/W$', 'Wquant', pname))
        if with_bias:
            quantization_b = partial(PQ.parametric_pow2_quantize_b_xmin, sign=True, with_zero=False,
                                     n_init=cfg.w_bitwidth,
                                     n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                     xmin_init=xmin,
                                     xmin_min=cfg.w_xmin_min, xmin_max=cfg.w_xmin_max,
                                     name=re.sub('quantized_[^/]*/W$', 'bquant', pname))
    elif cfg.w_quantize == 'parametric_pow2_xmin_xmax':
        quantization_w = partial(PQ.parametric_pow2_quantize_xmin_xmax, sign=True, with_zero=False,
                                 xmin_init=xmin,
                                 xmin_min=cfg.w_xmin_min, xmin_max=cfg.w_xmin_max,
                                 xmax_init=xmax,
                                 xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                 name=re.sub('quantized_[^/]*/W$', 'Wquant', pname))
        if with_bias:
            quantization_b = partial(PQ.parametric_pow2_quantize_xmin_xmax, sign=True, with_zero=False,
                                     xmin_init=xmin,
                                     xmin_min=cfg.w_xmin_min, xmin_max=cfg.w_xmin_max,
                                     xmax_init=xmax,
                                     xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                     name=re.sub('quantized_[^/]*/W$', 'bquant', pname))
    else:
        quantization_w = None
        quantization_b = None

    return quantization_w, quantization_b


def affi(x, n_outmaps, cfg, test, w_init, b_init, name=None):
    if name is None:
        pname = "quantized_affine/W"
    else:
        pname = "{}/quantized_affine/W".format(name)

    quantization_w, quantization_b = get_quantizers(cfg=cfg, test=test, pname=pname)

    return PQ.quantized_affine(x, n_outmaps,
                               name=name,
                               w_init=w_init, b_init=b_init,
                               quantization_w=quantization_w,
                               quantization_b=quantization_b)


def conv(x, outmaps, kernel, cfg, test, name=None, pad=None, stride=None,
         with_bias=True, w_init=None, b_init=None):

    if name is None:
        pname = "quantized_conv/W"
    else:
        pname = "{}/quantized_conv/W".format(name)

    quantization_w, quantization_b = get_quantizers(cfg=cfg, test=test, pname=pname, with_bias=with_bias)

    return PQ.quantized_convolution(x, outmaps, kernel,
                                    name=name,
                                    pad=pad, stride=stride,
                                    with_bias=with_bias,
                                    w_init=w_init, b_init=b_init,
                                    quantization_w=quantization_w,
                                    quantization_b=quantization_b)


def nonl(x, cfg, inplace=False):
    # for convenience, store size of x (this allows us to compute the number of activations)
    _s = get_parameter_or_create('Asize', (), ConstantInitializer(np.prod(x.shape[1:])), need_grad=False)

    # get stepsize/maximum value
    delta = cfg.a_stepsize
    xmax = delta * (2. ** cfg.a_bitwidth - 1)

    if cfg.a_quantize is not None and 'pow2' in cfg.a_quantize:
        xmax = 2. ** np.round(np.log2(xmax))
        xmin = xmax / 2. ** (2. ** (cfg.a_bitwidth-1) - 1)

        xmin = np.clip(xmin, cfg.a_xmin_min + 1e-5, cfg.a_xmin_max - 1e-5)
        xmax = np.clip(xmax, cfg.a_xmax_min + 1e-5, cfg.a_xmax_max - 1e-5)

    print(f'We use default delta ({delta, xmax}) for quantized nonlinearity.')

    if cfg.a_quantize == 'fp_relu':
        return F.fixed_point_quantize(x, sign=False,
                                      n=cfg.a_bitwidth, delta=cfg.a_stepsize)
    elif cfg.a_quantize == 'parametric_fp_b_xmax_relu':
        return PQ.parametric_fixed_point_quantize_b_xmax(x, sign=False,
                                                         n_init=cfg.a_bitwidth,
                                                         n_min=cfg.a_bitwidth_min, n_max=cfg.a_bitwidth_max,
                                                         xmax_init=xmax,
                                                         xmax_min=cfg.a_xmax_min, xmax_max=cfg.a_xmax_max,
                                                         name='Aquant')
    elif cfg.a_quantize == 'parametric_fp_d_xmax_relu':
        return PQ.parametric_fixed_point_quantize_d_xmax(x, sign=False,
                                                         d_init=delta,
                                                         d_min=cfg.a_stepsize_min, d_max=cfg.a_stepsize_max,
                                                         xmax_init=xmax,
                                                         xmax_min=cfg.a_xmax_min, xmax_max=cfg.a_xmax_max,
                                                         name='Aquant')
    elif cfg.a_quantize == 'parametric_fp_d_b_relu':
        return PQ.parametric_fixed_point_quantize_d_b(x, sign=False,
                                                      n_init=cfg.a_bitwidth,
                                                      n_min=cfg.a_bitwidth_min, n_max=cfg.a_bitwidth_max,
                                                      d_init=delta,
                                                      d_min=cfg.a_stepsize_min, d_max=cfg.a_stepsize_max,
                                                      name='Aquant')
    elif cfg.a_quantize == 'pow2_relu':
        return F.pow2_quantize(x, sign=False, with_zero=True,
                               n=cfg.a_bitwidth, m=np.round(np.log2(xmax)))
    elif cfg.a_quantize == 'parametric_pow2_b_xmax_relu':
        return PQ.parametric_pow2_quantize_b_xmax(x, sign=False, with_zero=True,
                                                  n_init=cfg.a_bitwidth,
                                                  n_min=cfg.a_bitwidth_min, n_max=cfg.a_bitwidth_max,
                                                  xmax_init=xmax,
                                                  xmax_min=cfg.a_xmax_min, xmax_max=cfg.a_xmax_max,
                                                  name='Aquant')
    elif cfg.a_quantize == 'parametric_pow2_b_xmin_relu':
        return PQ.parametric_pow2_quantize_b_xmin(x, sign=False, with_zero=True,
                                                  n_init=cfg.a_bitwidth,
                                                  n_min=cfg.a_bitwidth_min, n_max=cfg.a_bitwidth_max,
                                                  xmin_init=xmin,
                                                  xmin_min=cfg.a_xmin_min, xmin_max=cfg.a_xmax_max,
                                                  name='Aquant')
    elif cfg.a_quantize == 'parametric_pow2_xmin_xmax_relu':
        return PQ.parametric_pow2_quantize_xmin_xmax(x, sign=False, with_zero=True,
                                                     xmin_init=xmin,
                                                     xmin_min=cfg.a_xmin_min, xmin_max=cfg.a_xmax_max,
                                                     xmax_init=xmax,
                                                     xmax_min=cfg.a_xmax_min, xmax_max=cfg.a_xmax_max,
                                                     name='Aquant')
    else:
        return F.relu(x, inplace=inplace)


def shortcut(x, ochannels, stride, cfg, test):
    ichannels = x.shape[1]
    use_conv = cfg.shortcut_type.lower() == 'c'
    if ichannels != ochannels:
        if cfg.shortcut_type.lower() == 'b':
            use_conv = True
    if use_conv:
        # Convolution does everything.
        # Matching channels, striding.
        with nn.parameter_scope("shortcut_conv"):
            x = conv(x, ochannels, (1, 1), cfg, test,
                     stride=stride, with_bias=False)
            x = PF.batch_normalization(x, batch_stat=not test)
    else:
        if stride != (1, 1):
            # Stride
            x = F.average_pooling(x, (1, 1), stride)
        if ichannels != ochannels:
            # Zero-padding to channel axis
            ishape = x.shape
            zeros = F.constant(
                0, (ishape[0], ochannels - ichannels) + ishape[-2:])
            x = F.concatenate(x, zeros, axis=1)
    return x


def basicblock(x, ochannels, stride, cfg, test):
    def bn(h):
        return PF.batch_normalization(h, batch_stat=not test)

    with nn.parameter_scope("basicblock1"):
        h = nonl(bn(conv(x, ochannels, (3, 3), cfg, test,
                         pad=(1, 1), stride=stride, with_bias=False)),
                 cfg, inplace=True)
    with nn.parameter_scope("basicblock2"):
        h = bn(conv(h, ochannels, (3, 3), cfg, test, pad=(1, 1), with_bias=False))
    with nn.parameter_scope("basicblock_s"):
        s = shortcut(x, ochannels, stride, cfg, test)
    return nonl(F.add2(h, s, inplace=True), cfg, inplace=True)


def layer(x, block, ochannels, count, stride, cfg, test):
    for i in range(count):
        with nn.parameter_scope("layer{}".format(i + 1)):
            x = block(x, ochannels, stride if i ==
                      0 else (1, 1), cfg, test)
    return x


#-----------------------------------------resnets for different datasets-----------------------------------------------------

#-----------------------------------------------resnets for cifar10----------------------------------------------------------
def resnet_cifar10(x, num_classes, cfg, test):
    """
    Args:
        x : Variable
        num_classes : Number of classes of outputs
        cfg : network configuration
    """
    layers = {
        20: ((3, 3, 3), basicblock, 1),
        32: ((5, 5, 5), basicblock, 1),
        44: ((7, 7, 7), basicblock, 1),
        56: ((9, 9, 9), basicblock, 1),
        110: ((18, 18, 18), basicblock, 1)}

    counts, block, ocoef = layers[cfg.num_layers]
    logger.debug(x.shape)
    with nn.parameter_scope("conv1"):
        stride = (1, 1)
        r = conv(x, 16, (3, 3), cfg, test,
                 pad=(1, 1), stride=stride, with_bias=False)
        r = nonl(PF.batch_normalization(
            r, batch_stat=not test), cfg, inplace=True)
    hidden = {}
    hidden['r0'] = r
    ochannels = [16, 32, 64]
    strides = [1, 2, 2]
    logger.debug(r.shape)
    for i in range(3):
        with nn.parameter_scope("res{}".format(i + 1)):
            r = layer(r, block, ochannels[i] * ocoef,
                      counts[i], (strides[i], strides[i]), cfg, test)
        hidden['r{}'.format(i + 1)] = r
        logger.debug(r.shape)
    r = F.average_pooling(r, r.shape[-2:])
    with nn.parameter_scope("fc"):
        stdv = 1. / np.sqrt(np.prod(r.shape[1:]))
        init = nn.initializer.UniformInitializer(lim=(-stdv, stdv))
        r = affi(r, num_classes, cfg, test, w_init=init, b_init=init)
        if cfg.scale_layer:
            s = get_parameter_or_create('scale_layer', shape=(1,1), initializer=np.ones((1,1), dtype=np.float32), need_grad=True)
            r = s * r
    logger.debug(r.shape)
    return r, hidden

