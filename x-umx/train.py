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
MSS Training code using X-UMX/UMX.
'''

import os
import nnabla as nn
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context, import_extension_module
from nnabla.utils.data_iterator import data_iterator
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from numpy.random import RandomState, seed
from tqdm import trange
from lr_scheduler import ReduceLROnPlateau
from comm import CommunicatorWrapper
from args import get_train_args
from model import get_model
from data import load_datasources
import utils
seed(42)


def train():
    # Check NNabla version
    if utils.get_nnabla_version_integer() < 11900:
        raise ValueError(
            'Please update the nnabla version to v1.19.0 or latest version since memory efficiency of core engine is improved in v1.19.0')

    parser, args = get_train_args()

    # Get context.
    ctx = get_extension_context(args.context, device_id=args.device_id)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)
    ext = import_extension_module(args.context)

    # Monitors
    # setting up monitors for logging
    monitor_path = args.output
    monitor = Monitor(monitor_path)

    monitor_best_epoch = MonitorSeries(
        'Best epoch', monitor, interval=1)
    monitor_traing_loss = MonitorSeries(
        'Training loss', monitor, interval=1)
    monitor_validation_loss = MonitorSeries(
        'Validation loss', monitor, interval=1)
    monitor_lr = MonitorSeries(
        'learning rate', monitor, interval=1)
    monitor_time = MonitorTimeElapsed(
        "training time per iteration", monitor, interval=1)

    if comm.rank == 0:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    # Initialize DataIterator for MUSDB18.
    train_source, valid_source, args = load_datasources(parser, args)

    train_iter = data_iterator(
        train_source,
        args.batch_size,
        RandomState(args.seed),
        with_memory_cache=False,
    )

    valid_iter = data_iterator(
        valid_source,
        1,
        RandomState(args.seed),
        with_memory_cache=False,
    )

    if comm.n_procs > 1:
        train_iter = train_iter.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

        valid_iter = valid_iter.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    # Calculate maxiter per GPU device.
    # Change max_iter, learning_rate and weight_decay according no. of gpu devices for multi-gpu training.
    default_batch_size = 16
    train_scale_factor = (comm.n_procs * args.batch_size) / default_batch_size
    max_iter = int((train_source._size // args.batch_size) // comm.n_procs)
    weight_decay = args.weight_decay * train_scale_factor
    args.lr = args.lr * train_scale_factor

    # Calculate the statistics (mean and variance) of the dataset
    scaler_mean, scaler_std = utils.get_statistics(args, train_source)

    # clear cache memory
    ext.clear_memory_cache()

    max_bin = utils.bandwidth_to_max_bin(
        train_source.sample_rate, args.nfft, args.bandwidth
    )

    # Get X-UMX/UMX computation graph and variables as namedtuple
    model = get_model(args, scaler_mean, scaler_std, max_bin=max_bin)

    # Create Solver and set parameters.
    solver = S.Adam(args.lr)
    solver.set_parameters(nn.get_parameters())

    # Initialize Early Stopping
    es = utils.EarlyStopping(patience=args.patience)

    # Initialize LR Scheduler (ReduceLROnPlateau)
    lr_scheduler = ReduceLROnPlateau(
        lr=args.lr, factor=args.lr_decay_gamma, patience=args.lr_decay_patience)
    best_epoch = 0

    # AverageMeter for mean loss calculation over the epoch
    losses = utils.AverageMeter()

    # Training loop.
    for epoch in trange(args.epochs):
        # TRAINING
        losses.reset()
        for batch in range(max_iter):
            model.mixture_audio.d, model.target_audio.d = train_iter.next()
            solver.zero_grad()
            model.loss.forward(clear_no_need_grad=True)
            if comm.n_procs > 1:
                all_reduce_callback = comm.get_all_reduce_callback()
                model.loss.backward(clear_buffer=True,
                                    communicator_callbacks=all_reduce_callback)
            else:
                model.loss.backward(clear_buffer=True)
            solver.weight_decay(weight_decay)
            solver.update()
            losses.update(model.loss.d.copy(), args.batch_size)
        training_loss = losses.get_avg()

        # clear cache memory
        ext.clear_memory_cache()

        # VALIDATION
        losses.reset()
        for batch in range(int(valid_source._size // comm.n_procs)):
            x, y = valid_iter.next()
            dur = int(valid_source.sample_rate * args.valid_dur)
            sp, cnt = 0, 0
            loss_tmp = nn.NdArray()
            loss_tmp.zero()
            while 1:
                model.vmixture_audio.d = x[Ellipsis, sp:sp+dur]
                model.vtarget_audio.d = y[Ellipsis, sp:sp+dur]
                model.vloss.forward(clear_no_need_grad=True)
                cnt += 1
                sp += dur
                loss_tmp += model.vloss.data
                if x[Ellipsis, sp:sp+dur].shape[-1] < dur or x.shape[-1] == cnt*dur:
                    break
            loss_tmp = loss_tmp / cnt
            if comm.n_procs > 1:
                comm.all_reduce(loss_tmp, division=True, inplace=True)
            losses.update(loss_tmp.data.copy(), 1)
        validation_loss = losses.get_avg()

        # clear cache memory
        ext.clear_memory_cache()

        lr = lr_scheduler.update_lr(validation_loss, epoch=epoch)
        solver.set_learning_rate(lr)
        stop = es.step(validation_loss)

        if comm.rank == 0:
            monitor_best_epoch.add(epoch, best_epoch)
            monitor_traing_loss.add(epoch, training_loss)
            monitor_validation_loss.add(epoch, validation_loss)
            monitor_lr.add(epoch, lr)
            monitor_time.add(epoch)

            if validation_loss == es.best:
                best_epoch = epoch
                # save best model
                if args.umx_train:
                    nn.save_parameters(os.path.join(
                        args.output, 'best_umx.h5'))
                else:
                    nn.save_parameters(os.path.join(
                        args.output, 'best_xumx.h5'))

        if args.umx_train:
            # Early stopping for UMX after `args.patience` (140) number of epochs
            if stop:
                print("Apply Early Stopping")
                break


if __name__ == '__main__':
    train()
