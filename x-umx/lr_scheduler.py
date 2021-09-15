# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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
ReduceLROnPlateau Learning Rate Scheduler code.
'''

import math


class ReduceLROnPlateau():
    """
    Reduce learning rate when loss has stopped reducing
    If loss does not reduce for 'patience' number of epochs, 
    learning rate is reduced by specified factor.
    Args:
        mode (str): either `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. 
            Default: 'min'.
        lr (float): learning rate before scheduler is put to work
            Default: 0
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. 
            Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        compare_mode (str): One of `rel`, `abs`. 
            In `rel` mode, current value must be '> best * ( 1 + delta)' or 
            '< best * ( 1 - delta)' to consider it is significant.
            In `abs` mode, current value must be '> best + delta' or 
            '< best - delta' to consider it is significant.
            Default: 'rel'
        cooldown (int): Number of epochs to wait before fiddling with learning 
            rate if it has already been reduced.
            Default: 0.
        min_lr (float): Lower bound on the learning rate. 
            Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. 
            Default: 1e-8.
    Example:
        >>> scheduler = ReduceLROnPlateau('min')
        >>> for epoch in range(epochs):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     lr = scheduler.update_lr(val_loss, epoch)
    """

    def __init__(self, mode='min', lr=0, factor=0.1, patience=10,
                 compare_mode='rel', cooldown=10,
                 min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        self.lr = lr
        self._last_lr = None
        self.min_lr = min_lr
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.compare_mode = compare_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self.delta = 1e-4
        self._init_is_better(mode=mode, compare_mode=compare_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def update_lr(self, cur_metric, epoch=None):

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(cur_metric, self.best):
            self.best = cur_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        self._last_lr = self.lr

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.lr

    def _reduce_lr(self):
        new_lr = max(self.lr * self.factor, self.min_lr)
        if self._last_lr - new_lr > self.eps:
            self._last_lr = self.lr
            self.lr = new_lr

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.compare_mode == 'rel':
            return a < best * (1. - self.delta)

        elif self.mode == 'min' and self.compare_mode == 'abs':
            return a < best - self.delta

        elif self.mode == 'max' and self.compare_mode == 'rel':
            return a > best * (self.delta + 1.)

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.delta

    def _init_is_better(self, mode, compare_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if compare_mode not in {'rel', 'abs'}:
            raise ValueError('compare mode ' + compare_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = math.inf
        else:  # mode == 'max':
            self.mode_worse = -(math.inf)

        self.mode = mode
        self.compare_mode = compare_mode
