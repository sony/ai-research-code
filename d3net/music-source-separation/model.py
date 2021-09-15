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
D3net Architecture definition for MSS.
'''

import math
import os
import sys
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.parameter import get_parameter_or_create
import nnabla.initializer as I
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from d3net_basic_blocks import D3NetBase


def stft(x, n_fft=4096, n_hop=1024, center=True, patch_length=None):
    '''
    Multichannel STFT
    Input: (nb_samples, nb_channels, nb_timesteps)
    Output: (nb_samples, nb_channels, nb_bins, nb_frames),
            (nb_samples, nb_channels, nb_bins, nb_frames)
    '''
    nb_samples, nb_channels, _ = x.shape
    x = F.reshape(x, (nb_samples*nb_channels, -1))
    real, imag = F.stft(x, n_fft, n_hop, n_fft,
                        window_type='hanning', center=center, pad_mode='reflect')
    real = F.reshape(real, (nb_samples, nb_channels, n_fft // 2 + 1, -1))
    imag = F.reshape(imag, (nb_samples, nb_channels, n_fft // 2 + 1, -1))

    if patch_length is not None:
        # slice 256(patch_length) frames from 259 frames
        return real[..., :patch_length], imag[..., :patch_length]
    return real, imag


def spectogram(real, imag, power=1, mono=True):
    '''
    Input:  (nb_samples, nb_channels, nb_bins, nb_frames),
            (nb_samples, nb_channels, nb_bins, nb_frames)
    Output: (nb_samples, nb_frames, nb_channels, nb_bins)
    '''
    spec = ((real ** 2) + (imag ** 2)) ** (power / 2.0)
    if mono:
        spec = F.mean(spec, axis=1, keepdims=True)

    return F.transpose(spec, ((0, 3, 1, 2)))


class D3NetMSS(D3NetBase):
    def __init__(self, hparams, comm=None, test=False, recompute=False, init_method=None, input_mean=None, input_scale=None):
        super(D3NetMSS, self).__init__(comm=comm, test=test,
                                       recompute=recompute, init_method=init_method)
        self.hparams = hparams
        if input_mean is None or input_scale is None:
            input_mean = np.zeros((1, 1, 1, self.hparams['fft_size']//2+1))
            input_scale = np.ones((1, 1, 1, self.hparams['fft_size']//2+1))
        else:
            input_mean = input_mean.reshape(
                (1, 1, 1, self.hparams['fft_size']//2+1))
            input_scale = input_scale.reshape(
                (1, 1, 1, self.hparams['fft_size']//2+1))

        self.in_offset = get_parameter_or_create(
            'in_offset', shape=input_mean.shape, initializer=input_mean)
        self.in_scale = get_parameter_or_create(
            'in_scale', shape=input_scale.shape, initializer=input_scale)
        self.decode_scale = get_parameter_or_create(
            'decode_scale', (1, 1, 1, self.hparams['valid_signal_idx']), initializer=I.ConstantInitializer(value=1))
        self.decode_bias = get_parameter_or_create(
            'decode_bias', (1, 1, 1, self.hparams['valid_signal_idx']), initializer=I.ConstantInitializer(value=1))

    def dilated_dense_block_2(self, inp, growth_rate, num_layers, scope_name, name='bottleneck'):
        '''
        Dilated Dense Block-2
        '''
        with nn.parameter_scope(scope_name):
            with nn.parameter_scope('initial_layer'):
                out = self.bn_conv_block(
                    inp, growth_rate*num_layers, name=name)
            return self.dilated_dense_block(out, growth_rate, num_layers, name=name)

    def upsampling_layer(self, inp, num_input_features, comp_rate=1.):
        '''
        Define Upsampling Layer
        '''
        num_filters = int(math.ceil(comp_rate * num_input_features))
        with nn.parameter_scope('upsample'):
            out = self.batch_norm(inp, name='norm')
            out = PF.deconvolution(out, num_filters, kernel=(
                2, 2), stride=(2, 2), name='transconv')
        return out

    def d3_block(self, inp, growth_rate, num_layers, n_blocks):
        '''
        Define D3Block
        '''
        out = self.dilated_dense_block_2(
            inp, growth_rate*n_blocks, num_layers, scope_name='initial_block')
        if n_blocks > 1:
            lst = []
            for i in range(n_blocks):
                lst.append(out[:, i*growth_rate:(i+1)*growth_rate])

            def update(inp_, n):
                for j in range(n_blocks-n-1):
                    lst[j+1+n] += inp_[:, j*growth_rate:(j+1)*growth_rate]
            for i in range(n_blocks-1):
                tmp = self.dilated_dense_block_2(
                    lst[i], growth_rate*(n_blocks-i-1), num_layers, scope_name='layers/layer%s' % (i+1))
                update(tmp, i)
            out = F.concatenate(*lst, axis=1)
        return out[:, -growth_rate:]

    def md3_block_ds(self, inp, in_channels, ks, n_layers, n_blocks, comp_rates, name=''):
        '''
        Define MD3BlockDS
        '''
        if not len(ks) == len(n_layers):
            print('length of ks and n_layers should be match.')
        if min(len(ks), len(n_layers)) % 2 == 0:
            sys.stderr.write(
                'length of ks, n_layers and comp_rates must be odd.')
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
                with nn.parameter_scope('dense_block%s' % dense_blk_cnt):
                    out = self.d3_block(out, k, nl, b)
                ds_concat_channels.append(k)
                n_channels = k
                ds.append(out)
                out = F.average_pooling(out, kernel=(2, 2), stride=(2, 2))
                if comp < 1.:
                    n_channels = int(math.ceil(comp * n_channels))
                dense_blk_cnt += 1

        # concatenation happens in reverse order, so reverse the list
        ds = ds[::-1]

        # bottleneck block
        with nn.parameter_scope(name + '/bottleneck_block'):
            out = self.d3_block(
                out, ks[ds_len], n_layers[ds_len], n_blocks[ds_len])
        dense_blk_cnt += 1
        out_layers.append(out)
        ds_concat_channels = ds_concat_channels[::-1]
        n_channels = ks[ds_len]

        # Up-sampling path
        cnt = 0
        with nn.parameter_scope(name + '/us_layers'):
            for k, nl, comp, b, i in zip(ks[ds_len + 1:], n_layers[ds_len + 1:], comp_rates[ds_len:], n_blocks[ds_len:], range(ds_len)):
                with nn.parameter_scope('upsample%s' % i):
                    out = self.upsampling_layer(out, n_channels, comp)
                    out = F.concatenate(out, ds[cnt], axis=1)
                    cnt += 1
                if comp < 1.:
                    n_channels = int(math.ceil(comp * n_channels))
                n_channels += ds_concat_channels[i]
                with nn.parameter_scope('dense_block%s' % dense_blk_cnt):
                    out = self.d3_block(out, k, nl, b)
                    out_layers.append(out)
                n_channels = k
                dense_blk_cnt += 1

        return out_layers

    def __call__(self, inp):
        '''
        Define D3Net
        '''

        valid_signal_idx = self.hparams['valid_signal_idx']
        band_split_idxs = self.hparams['band_split_idxs'] + \
            [self.hparams['valid_signal_idx']]

        inp = F.transpose(inp, (0, 2, 1, 3))

        scaled_inp = (inp - self.in_offset)/self.in_scale

        max_final_k = 0
        for k in self.hparams['dens_k']:
            if max_final_k < k[-1]:
                max_final_k = k[-1]
        i = 0
        band_idx_start = 0
        band_out = []
        band_dense_out = []

        # Low ~ middle bands
        for num_init_features, dens_k, num_layer_block, b_n_block, comp_rates in zip(self.hparams['num_init_features'], self.hparams['dens_k'], self.hparams['num_layer_blocks'], self.hparams['b_n_blocks'], self.hparams['comp_rates']):
            x_band = scaled_inp[:, :, :, band_idx_start:band_split_idxs[i]]
            x_band = self.conv2d(x_band, num_init_features, kernel_size=3,
                                 stride=1, name='features_init/%s' % i, pad=1)
            dense_band = self.md3_block_ds(
                x_band, num_init_features, dens_k, num_layer_block, b_n_block, comp_rates, name='dense_band/%s' % i)
            band_dense_out.append(dense_band[::-1])
            if max_final_k > self.hparams['dens_k'][i][-1]:
                h = self.batch_norm(
                    band_dense_out[-1][0], name='match_fm_conv/%s/norm' % i)
                out = self.conv2d(h, max_final_k, kernel_size=1,
                                  stride=1, name='match_fm_conv/%s/conv' % i)
                band_out.append(out)
            else:
                band_out.append(band_dense_out[-1][0])
            band_idx_start = band_split_idxs[i]
            i += 1

        # full bands
        full = self.conv2d(scaled_inp[:, :, :, :valid_signal_idx], self.hparams['f_num_init_features'], kernel_size=3,
                           stride=1, name='features_init_full', pad=1)
        full = self.md3_block_ds(full, self.hparams['f_num_init_features'], self.hparams['f_dens_k'], self.hparams['f_num_layer_block'],
                                 self.hparams['f_n_blocks'], self.hparams['f_comp_rates'], name='dense_full')

        # concat low~middle bands and then with full bands
        concat_bands = F.concatenate(*band_out, axis=3)
        concat_full = F.concatenate(*[concat_bands, full[-1]], axis=1)

        # Final dense block
        final = self.dilated_dense_block_2(
            concat_full, self.hparams['ttl_dens_k'], self.hparams['ttl_num_layer_block'], scope_name='final_dense')

        # Define BNC_Gate : Batch-Normalization, Convolution and Sigmoid Gate
        with nn.parameter_scope('out_gate'):
            bn_out = self.batch_norm(final, name='bn')
            gate = F.sigmoid(self.conv2d(
                bn_out, self.hparams['n_channels'], kernel_size=1, stride=1, name='conv_gate/conv'))
            filt = self.conv2d(
                bn_out, self.hparams['n_channels'], kernel_size=1, stride=1, name='conv_filt/conv')

        out = gate * filt
        out = out * self.decode_scale + self.decode_bias
        out = F.relu(out)
        out = F.concatenate(*[out, inp[:, :, :, valid_signal_idx:]], axis=3)
        out = F.transpose(out, (0, 2, 1, 3))
        return out
