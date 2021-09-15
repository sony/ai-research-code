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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I

'''
D3net Basic Blocks definition.
'''


class BasicLayers(object):
    '''
    Define 2D-Convolution Layer abd BatchNormalization Layer
    '''

    def __init__(self, comm=None, test=False, recompute=False, init_method=None):
        self.comm = comm
        self.test = test
        self.recompute = recompute
        self.init_method = init_method

    def conv2d(self, conv_input, out_channels, kernel_size, stride, bias=True, name='', dilation=1, pad=0):
        '''
        Define 2D-Convolution Layer
        '''
        if self.init_method == 'xavier':
            sigma = I.calc_normal_std_glorot(
                conv_input.shape[1], out_channels, kernel=(kernel_size, kernel_size))
            w_init = I.NormalInitializer(sigma)
        elif self.init_method == 'normal':
            w_init = I.NormalInitializer(sigma=0.01)
        else:
            w_init = None
        conv_out = PF.convolution(conv_input, out_channels, kernel=(kernel_size, kernel_size), stride=(
            stride, stride), with_bias=bias, dilation=(dilation, dilation), pad=(pad, pad), name=name, w_init=w_init)
        conv_out.apply(recompute=self.recompute)
        return conv_out

    def batch_norm(self, inp, name):
        '''
        Define BatchNormalization Layer
        '''
        if self.comm is not None:
            return PF.sync_batch_normalization(inp, comm=self.comm, group='world', batch_stat=not self.test, name=name)
        return PF.batch_normalization(inp, batch_stat=not self.test, name=name)


class D3NetBase(BasicLayers):
    '''
    A base class of D3Net.
    '''

    def __init__(self, comm=None, test=False, recompute=False, init_method=None):
        super(D3NetBase, self).__init__(comm=comm, test=test,
                                        recompute=recompute, init_method=init_method)

    def bn_conv_block(self, inp, growth_rate, name, kernel_size=3, dilation=1, pad=1, stride=1):
        '''
        Define Simple Batch-Normalization and Convolution Block
        '''
        with nn.parameter_scope(name):
            # Conv 3x3
            out = self.batch_norm(inp, name='norm1')
            out = F.relu(out, inplace=True)
            out = self.conv2d(out, growth_rate, kernel_size=kernel_size,
                              stride=stride, name='conv1', dilation=dilation, pad=pad)
        return out

    def dilated_dense_block(self, inp, growth_rate, num_layers, name, kernel_size=3, out_block=1):
        '''
        Dilated Dense Block
        '''
        out = inp
        if num_layers > 1:
            lst = []
            for i in range(num_layers):
                # Split Variable(h) and append them in lst.
                lst.append(inp[:, i*growth_rate:(i+1)*growth_rate])

            def update(inp_, n):
                for j in range(num_layers-n-1):
                    lst[j+1+n] += inp_[:, j*growth_rate:(j+1)*growth_rate]

            for i in range(num_layers-1):
                d = int(2**(i+1))
                with nn.parameter_scope('layers/layer%s' % (i+1)):
                    update(self.bn_conv_block(
                        lst[i], growth_rate*(num_layers-i-1), name, dilation=d, kernel_size=kernel_size, pad=d), i)

            # concatenate the splitted and updated Variables from the lst
            out = F.concatenate(*lst, axis=1)
        return out[:, -out_block*growth_rate:]
