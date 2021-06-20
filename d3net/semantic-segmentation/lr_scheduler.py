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
PolynomialScheduler Code
'''


class PolynomialScheduler:
    '''
    PolynomialScheduler: Polynominal decay
    '''

    def __init__(self, hparams):
        '''
        Args:
            hparams (dict): Hyper-Parameters
        '''
        self.base_lr = hparams['lr']
        self.min_lr = hparams['min_lr']
        self.max_iter = hparams['max_iter']
        self.power = hparams['power']

    def get_learning_rate(self, i):
        '''
        Get learning rate with polymomial decay based on current iteration.
        Args:
            i (int): current iteration (starting with 0).
        Returns:
            float: Learning rate
        '''
        coeff = (1.0 - i * 1.0 / self.max_iter) ** self.power
        return (self.base_lr - self.min_lr) * coeff + self.min_lr
