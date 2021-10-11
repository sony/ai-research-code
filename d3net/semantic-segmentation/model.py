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
D3net Architecture definition.
'''

import os
import sys
import nnabla as nn
import nnabla.functions as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from d3net_basic_blocks import BasicLayers, D3NetBase


class D3NetBC(D3NetBase):
    '''
    D3Net backbone.

    N.Takahashi, et al. "Densely connected multidilated convolutional networks for dense prediction tasks", CVPR2021
    arXiv: https://arxiv.org/abs/2011.11844
    '''

    def __init__(self, hparams, comm=None, test=False, recompute=False, init_method=None):
        super(D3NetBC, self).__init__(comm=comm, test=test,
                                      recompute=recompute, init_method=init_method)
        self.hparams = hparams
        self.x_list = []

    def transition(self, inp, out_channels):
        '''
        Transitional Layer Between D3-Blocks
        '''
        out = F.relu(self.batch_norm(inp, name='bn'), inplace=True)
        out = self.conv2d(out, out_channels, kernel_size=1,
                          stride=1, bias=False, name='conv')
        out = F.average_pooling(out, kernel=(2, 2))
        return out

    def dilated_dense_block_2(self, inp, growth_rate, num_layers, out_block, block_comp, name='bottle_neck'):
        '''
        Dilated Dense Block-2
        '''
        out = inp
        bc_ch = growth_rate * 4

        with nn.parameter_scope('d2block'):
            with nn.parameter_scope('initial_layer'):
                if bc_ch < inp.shape[1]:
                    with nn.parameter_scope('bc_layer'):
                        out = self.bn_conv_block(
                            out, bc_ch, name=name, kernel_size=1, pad=0)

                with nn.parameter_scope('init_layer'):
                    out = self.bn_conv_block(
                        out, growth_rate*num_layers, name=name, kernel_size=3, pad=1)
            out = self.dilated_dense_block(
                out, growth_rate, num_layers, name=name, kernel_size=3, out_block=out_block)

        with nn.parameter_scope('block_dim_reduction'):
            out = self.batch_norm(out, name='norm1')
            out = F.relu(out, inplace=True)
            out = self.conv2d(out, int(growth_rate*out_block*block_comp),
                              kernel_size=1, stride=1, bias=False, name='conv1')
        return out

    def d3_block(self, inp, i):
        '''
        Definition of D3 Block
        '''
        growth_rate = self.hparams['dens_k'][i]
        num_layers = self.hparams['num_layers'][i]
        n_blocks = self.hparams['n_blocks'][i]
        out_block = self.hparams['dense_n_out_layer_block'][i]
        block_comp = self.hparams['block_comp'][i]
        intermediate_out_ch = self.hparams['intermediate_out_ch'][i]

        out = inp

        with nn.parameter_scope('dense%s' % (i+1)):
            with nn.parameter_scope('d2_block'):
                for j in range(n_blocks):
                    with nn.parameter_scope('block%s' % j):
                        block_out = self.dilated_dense_block_2(
                            out, growth_rate, num_layers, out_block, block_comp)
                    out = F.concatenate(out, block_out, axis=1)

        if i == 3:
            intermediate_out = self.conv2d(
                out, intermediate_out_ch, kernel_size=1, stride=1, bias=True, name='final_out%s/%s' % (i+1, 0))
            self.x_list.append(F.relu(self.batch_norm(
                intermediate_out, name='final_out%s/%s' % (i+1, 1)), inplace=True))
            return None

        intermediate_out = self.conv2d(out, intermediate_out_ch, kernel_size=1,
                                       stride=1, bias=True, name='intermediate_out%s/%s' % (i+1, 0))
        self.x_list.append(F.relu(self.batch_norm(
            intermediate_out, name='intermediate_out%s/%s' % (i+1, 1)), inplace=True))

        trans_ch = out.shape[1] // self.hparams['trans_comp_factor']
        with nn.parameter_scope('transition%s' % (i+1)):
            out = self.transition(out, trans_ch)
        return out

    def __call__(self, img):
        out = self.conv2d(img, self.hparams['num_init_features'],
                          kernel_size=3, stride=2, bias=False, name='conv1', pad=1)
        out = F.relu(self.batch_norm(out, name='bn1'), inplace=True)
        out = self.conv2d(out, self.hparams['num_init_features'],
                          kernel_size=3, stride=2, bias=False, name='conv2', pad=1)

        # scale 1
        out1 = self.d3_block(out, i=0)

        # scale 2
        out2 = self.d3_block(out1, i=1)

        # scale 3
        out3 = self.d3_block(out2, i=2)

        # scale 1
        out4 = self.d3_block(out3, i=3)

        return self.x_list


class FCNHead(BasicLayers):
    '''
    Fully Convolution Networks for Semantic Segmentation; decodes the extracted features into a semantic segmentation map
    This head is implementation of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    '''

    def __init__(self, hparams, output_size, comm=None, test=False, recompute=False, init_method=None):
        super(FCNHead, self).__init__(comm=comm, test=test,
                                      recompute=recompute, init_method=init_method)
        self.hparams = hparams
        self.output_size = output_size

    def __call__(self, features):
        upsampled_inputs = [F.interpolate(x, output_size=features[0].shape[2:], mode='linear',
                                          align_corners=False, half_pixel=True) for x in features]
        inputs = F.concatenate(*upsampled_inputs, axis=1)
        out = self.conv2d(inputs, self.hparams['channels'], kernel_size=1,
                          stride=1, bias=False, name='convs/0/conv')
        out = F.relu(self.batch_norm(out, name='convs/0/bn'))
        out = self.conv2d(out, self.hparams['num_classes'], kernel_size=1,
                          stride=1, bias=True, name='conv_seg')
        out = F.interpolate(out, output_size=self.output_size, mode='linear',
                            align_corners=False, half_pixel=True)
        if self.test:
            return F.softmax(out, axis=1)
        return out


def d3net_segmentation(img, hparams, test=False, recompute=False):
    '''
    Get Semantic Segmentation map of img by applying D3Net backbone and Fully Convolutional Network
    '''
    # get the encoded features by D3Net Backbone
    with nn.parameter_scope('backbone'):
        d3net = D3NetBC(hparams, test=test, recompute=recompute)
        features = d3net(img)

    # decode the extracted features into a semantic segmentation map
    with nn.parameter_scope('decode_head'):
        fcn_head = FCNHead(
            hparams, output_size=img.shape[2:], test=test, recompute=recompute, init_method='normal')
        seg_map = fcn_head(features)

    return seg_map
