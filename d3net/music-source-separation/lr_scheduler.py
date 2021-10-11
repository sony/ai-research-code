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
AnnealingScheduler for D3Net based MSS training.
'''


class AnnealingScheduler(object):
    """Annealing Learning Rate Scheduler.
    Args:
        init_lr (float): Initial learning rate.
        warmup (int): The number of iterations for warm-up.
        anneal_steps (list of int, optional): Iteration at which to anneal the learning rate.
            Defaults to ().
        anneal_factor (float, optional): Factor by which to anneal the learning rate.
            Defaults to 0.1.
    """

    def __init__(self, init_lr=0.001, anneal_steps=(), anneal_factor=0.1):
        self.init_lr = init_lr
        self.anneal_steps = anneal_steps
        self.anneal_factor = anneal_factor

    def get_learning_rate(self, cur_iter):
        r"""Get learning rate based on current iteration.
        Args:
            iter (int): Current iteration (starting with 0).
        Returns:
            float: Learning rate
        """
        step = cur_iter + 1
        power = len([x for x in self.anneal_steps if step >= x])
        lr = self.init_lr*(self.anneal_factor ** power)
        return lr
