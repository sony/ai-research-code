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

import math
import sys
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.parameter import get_parameter_or_create


def conv2d(conv_input, out_channels, kernel_size, stride, name='', dilation=None, pad=0):
    '''
    Define simple 2d Convolution
    '''
    kernel = (kernel_size, kernel_size)
    stride = (stride, stride)
    pad = (pad, pad)
    dilation = 1 if dilation is None else dilation
    dilation = (dilation, dilation)
    return PF.convolution(conv_input, out_channels, kernel=kernel, stride=stride, dilation=dilation, pad=pad, name=name)


def bnc_gate(inp, out_ch, gate_filter_size=(3, 3), gate_pad=1, stride=(1, 1), test=False):
    '''
    Define BNC_Gate : Batch-Normalization, Convolution and Sigmoid Gate
    '''
    with nn.parameter_scope('out_gate'):
        bn_out = PF.batch_normalization(inp, batch_stat=not test, name='bn')
        gate = F.sigmoid(conv2d(bn_out, out_ch, kernel_size=gate_filter_size,
                                stride=stride, pad=gate_pad, name='conv_gate/conv'))
        filt = conv2d(bn_out, out_ch, kernel_size=gate_filter_size,
                      stride=stride, pad=gate_pad, name='conv_filt/conv')
    return gate * filt


def bn_conv_block(inp, growth_rate, kernel_size=3, dilation=1, pad=1, stride=1, test=False):
    '''
    Define Simple Batch-Normalization and Convolution Block
    '''
    with nn.parameter_scope('bottleneck'):
        h = PF.batch_normalization(inp, batch_stat=not test, name='norm1')
        h = F.relu(h, inplace=True)
        out = conv2d(h, growth_rate, kernel_size=kernel_size, stride=stride,
                     name='conv1', dilation=dilation, pad=pad)
    return out


def dilated_dense_block(inp, growth_rate, num_layers, scope_name, kernel_size=3, pad=1, dilation=True, test=False):
    '''
    Define Dilated Dense Block
    '''
    with nn.parameter_scope(scope_name):
        with nn.parameter_scope('initial_layer'):
            h = bn_conv_block(inp, growth_rate*num_layers, dilation=1,
                              kernel_size=kernel_size, pad=pad, test=test)

        if num_layers > 1:
            lst = []
            for i in range(num_layers):
                # Split Variable(h) and append them in lst.
                lst.append(h[:, i*growth_rate:(i+1)*growth_rate])

            def update(inp_, n):
                for j in range(num_layers-n-1):
                    lst[j+1+n] += inp_[:, j*growth_rate:(j+1)*growth_rate]
            for i in range(num_layers-1):
                d = int(2**(i+1)) if dilation else 1
                with nn.parameter_scope('layers/layer%s' % (i+1)):
                    tmp = bn_conv_block(lst[i], growth_rate*(num_layers-i-1),
                                        dilation=d, kernel_size=kernel_size, pad=pad*d, test=test)
                    update(tmp, i)
            # concatenate the splitted and updated Variables from the lst
            h = F.concatenate(*lst, axis=1)
    return h[:, -growth_rate:]


def upsampling_layer(inp, num_input_features, comp_rate=1., test=False):
    '''
    Define Upsampling Layer
    '''
    num_filters = int(math.ceil(comp_rate * num_input_features))
    with nn.parameter_scope('upsample'):
        h = PF.batch_normalization(inp, batch_stat=not test, name='norm')
        out = PF.deconvolution(h, num_filters, kernel=(
            2, 2), stride=(2, 2), name='transconv')
    return out


def d3_block(inp, growth_rate, num_layers, n_blocks, kernel_size=3, pad=1, dilation=True, test=False):
    '''
    Define D3Block
    '''
    h = dilated_dense_block(inp, growth_rate*n_blocks, num_layers, scope_name='initial_block',
                            kernel_size=kernel_size, pad=pad, dilation=dilation, test=test)
    if n_blocks > 1:
        lst = []
        for i in range(n_blocks):
            lst.append(h[:, i*growth_rate:(i+1)*growth_rate])

        def update(inp_, n):
            for j in range(n_blocks-n-1):
                lst[j+1+n] += inp_[:, j*growth_rate:(j+1)*growth_rate]

        for i in range(n_blocks-1):
            tmp = dilated_dense_block(lst[i], growth_rate*(n_blocks-i-1), num_layers, scope_name='layers/layer%s' % (
                i+1), kernel_size=kernel_size, pad=pad, dilation=dilation, test=test)
            update(tmp, i)
        h = F.concatenate(*lst, axis=1)
    return h[:, -growth_rate:]


def md3_block_ds(inp, in_channels, ks, n_layers, n_blocks, comp_rates, kernel_size=3, pad=1, dilation=True, name='', test=False):
    '''
    Define MD3BlockDS
    '''
    if not len(ks) == len(n_layers):
        print('length of ks and n_layers should be match.')
    if min(len(ks), len(n_layers)) % 2 == 0:
        sys.stderr.write('length of ks, n_layers and comp_rates must be odd.')
    ds_len = (len(n_layers) - 1) // 2
    ds = []
    out = inp
    out_layers = []

    with nn.parameter_scope(name + '/ds_layers'):
        # Down-sampling path
        n_channels = in_channels
        dense_blk_cnt = 0
        ds_concat_channels = []
        for k, nl, comp, b in zip(ks[:ds_len], n_layers[:ds_len], comp_rates[:ds_len], n_blocks[:ds_len]):
            if nl > 0:
                with nn.parameter_scope('dense_block%s' % dense_blk_cnt):
                    out = d3_block(out, k, nl, b, kernel_size=kernel_size,
                                   pad=pad, dilation=dilation, test=test)

                ds_concat_channels.append(k)
                n_channels = k
            else:
                out = inp
                ds_concat_channels.append(n_channels)
            ds.append(out)

            out = F.average_pooling(out, kernel=(2, 2), stride=(2, 2))

            if comp < 1.:
                n_channels = int(math.ceil(comp * n_channels))
            dense_blk_cnt += 1

    # concatenation happens in reverse order, so reverse the list
    ds = ds[::-1]

    # bottleneck block
    with nn.parameter_scope(name + '/bottleneck_block'):
        out = d3_block(out, ks[ds_len], n_layers[ds_len], n_blocks[ds_len],
                       kernel_size=kernel_size, pad=pad, dilation=dilation, test=test)
    dense_blk_cnt += 1
    out_layers.append(out)
    ds_concat_channels = ds_concat_channels[::-1]
    n_channels = ks[ds_len]

    # Up-sampling path
    cnt = 0
    with nn.parameter_scope(name + '/us_layers'):
        for k, nl, comp, b, i in zip(ks[ds_len + 1:], n_layers[ds_len + 1:], comp_rates[ds_len:], n_blocks[ds_len:], range(ds_len)):
            with nn.parameter_scope('upsample%s' % i):
                out = upsampling_layer(out, n_channels, comp, test)
                out = F.concatenate(out, ds[cnt], axis=1)
                cnt += 1
            if comp < 1.:
                n_channels = int(math.ceil(comp * n_channels))
            n_channels += ds_concat_channels[i]
            if nl > 0:
                with nn.parameter_scope('dense_block%s' % dense_blk_cnt):
                    out = d3_block(out, k, nl, b, kernel_size=kernel_size,
                                   pad=pad, dilation=dilation, test=test)
                    out_layers.append(out)
            n_channels = k
            dense_blk_cnt += 1

    return out_layers


def d3_net(inp, hp, test=False):
    '''
    Define D3Net
    '''
    in_offset = get_parameter_or_create(
        'in_offset', (1, 1, 1, hp['fft_size']//2+1))
    in_scale = get_parameter_or_create(
        'in_scale', (1, 1, 1, hp['fft_size']//2+1))
    decode_scale = get_parameter_or_create(
        'decode_scale', (1, 1, 1, hp['valid_signal_idx']))
    decode_bias = get_parameter_or_create(
        'decode_bias', (1, 1, 1, hp['valid_signal_idx']))

    dilation = hp['dilation'] if 'dilation' in hp else True
    out_channels = hp['n_channels']
    if 'kernel_size' in hp:
        kernel_size = hp['kernel_size']
    else:
        kernel_size = 3
    pad = (kernel_size-1)//2
    valid_signal_idx = hp['valid_signal_idx']
    band_split_idxs = hp['band_split_idxs'] + [hp['valid_signal_idx']]
    inp = F.transpose(inp, (0, 2, 1, 3))
    scaled_inp = (inp - in_offset)/in_scale

    max_final_k = 0
    for k in hp['dens_k']:
        if max_final_k < k[-1]:
            max_final_k = k[-1]
    i = 0
    band_idx_start = 0
    band_out = []
    band_dense_out = []

    # Low ~ middle bands
    for num_init_features, dens_k, num_layer_block, b_n_block, comp_rates in zip(hp['num_init_features'], hp['dens_k'], hp['num_layer_blocks'], hp['b_n_blocks'], hp['comp_rates']):
        x_band = scaled_inp[:, :, :, band_idx_start:band_split_idxs[i]]
        x_band = conv2d(x_band, num_init_features, kernel_size=kernel_size,
                        stride=1, name='features_init/%s' % i, pad=pad)
        dense_band = md3_block_ds(x_band, num_init_features, dens_k, num_layer_block, b_n_block, comp_rates,
                                  kernel_size=kernel_size, pad=pad, dilation=dilation, name='dense_band/%s' % i, test=test)
        band_dense_out.append(dense_band[::-1])
        if max_final_k > hp['dens_k'][i][-1]:
            h = PF.batch_normalization(
                band_dense_out[-1][0], batch_stat=not test, name='match_fm_conv/%s/norm' % i)
            out = conv2d(h, max_final_k, kernel_size=1, stride=1,
                         name='match_fm_conv/%s/conv' % i, pad=0)
            band_out.append(out)
        else:
            band_out.append(band_dense_out[-1][0])
        num_features = max_final_k
        band_idx_start = band_split_idxs[i]
        i += 1
    num_features += hp['f_dens_k'][-1]

    # full bands
    full = conv2d(scaled_inp[:, :, :, :valid_signal_idx], hp['f_num_init_features'], kernel_size=kernel_size,
                  stride=1, name='features_init_full', dilation=dilation, pad=pad)
    full = md3_block_ds(full, hp['f_num_init_features'], hp['f_dens_k'], hp['f_num_layer_block'],
                        hp['f_n_blocks'], hp['f_comp_rates'], kernel_size=kernel_size, pad=pad,
                        dilation=dilation, name='dense_full', test=test)

    # concat low~middle bands and then with full bands
    concat_bands = F.concatenate(*band_out, axis=3)
    concat_full = F.concatenate(*[concat_bands, full[-1]], axis=1)

    # Final dense block
    final = dilated_dense_block(
        concat_full, hp['ttl_dens_k'], hp['ttl_num_layer_block'], kernel_size=kernel_size, scope_name='final_dense', test=test)
    out = bnc_gate(final, out_channels, gate_filter_size=1,
                   gate_pad=0, stride=1, test=test)
    out = out * decode_scale + decode_bias
    out = F.relu(out)
    out = F.concatenate(*[out, inp[:, :, :, valid_signal_idx:]], axis=3)
    out = F.transpose(out, (0, 2, 1, 3))
    return out
