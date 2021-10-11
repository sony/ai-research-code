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
MSS Training code using D3Net.
'''

import os
import re
import argparse
import yaml
from numpy.random import RandomState, seed
import nnabla as nn
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context, import_extension_module
from nnabla.utils.data_iterator import data_iterator
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
import nnabla.functions as F
from lr_scheduler import AnnealingScheduler
from comm import CommunicatorWrapper
from model import stft, spectogram, D3NetMSS
from data import load_datasources
from util import AverageMeter, get_statistics
from args import get_train_args


def get_nnabla_version_integer():
    r = list(map(int, re.match('^(\d+)\.(\d+)\.(\d+)', nn.__version__).groups()))
    return r[0] * 10000 + r[1] * 100 + r[2]


def train():
    # Check NNabla version
    if get_nnabla_version_integer() < 11900:
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
    monitor_path = os.path.join(args.output, args.target)
    monitor = Monitor(monitor_path)

    monitor_traing_loss = MonitorSeries(
        'Training loss', monitor, interval=1)
    monitor_lr = MonitorSeries(
        'learning rate', monitor, interval=1)
    monitor_time = MonitorTimeElapsed(
        "training time per epoch", monitor, interval=1)

    if comm.rank == 0:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    # Initialize DataIterator for MUSDB.
    train_source, args = load_datasources(parser, args)

    train_iter = data_iterator(
        train_source,
        args.batch_size,
        RandomState(args.seed),
        with_memory_cache=False
    )

    if comm.n_procs > 1:
        train_iter = train_iter.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    # Change max_iter, learning_rate and weight_decay according no. of gpu devices for multi-gpu training.
    default_batch_size = 6
    train_scale_factor = (comm.n_procs * args.batch_size) / default_batch_size

    max_iter = int(train_source._size // (comm.n_procs * args.batch_size))
    weight_decay = args.weight_decay * train_scale_factor
    args.lr = args.lr * train_scale_factor

    print(f"max_iter per GPU-device:{max_iter}")

    # Calculate the statistics (mean and variance) of the dataset
    scaler_mean, scaler_std = get_statistics(args, train_source)

    # clear cache memory
    ext.clear_memory_cache()

    # Create input variables.
    mixture_audio = nn.Variable(
        [args.batch_size] + list(train_source._get_data(0)[0].shape))
    target_audio = nn.Variable(
        [args.batch_size] + list(train_source._get_data(0)[1].shape))

    with open(f"./configs/{args.target}.yaml") as file:
        # Load target specific Hyper parameters
        hparams = yaml.load(file, Loader=yaml.FullLoader)

    # create training graph
    mix_spec = spectogram(
        *stft(mixture_audio,
              n_fft=hparams['fft_size'], n_hop=hparams['hop_size'], patch_length=256),
        mono=(hparams['n_channels'] == 1))
    target_spec = spectogram(
        *stft(target_audio,
              n_fft=hparams['fft_size'], n_hop=hparams['hop_size'], patch_length=256),
        mono=(hparams['n_channels'] == 1))

    with nn.parameter_scope(args.target):
        d3net = D3NetMSS(hparams, comm=comm.comm, input_mean=scaler_mean,
                         input_scale=scaler_std, init_method='xavier')
        pred_spec = d3net(mix_spec)

    loss = F.mean(F.squared_error(pred_spec, target_spec))
    loss.persistent = True

    # Create Solver and set parameters.
    solver = S.Adam(args.lr)
    solver.set_parameters(nn.get_parameters())

    # Initialize LR Scheduler (AnnealingScheduler)
    lr_scheduler = AnnealingScheduler(
        init_lr=args.lr, anneal_steps=[40], anneal_factor=0.1)

    # AverageMeter for mean loss calculation over the epoch
    losses = AverageMeter()

    for epoch in range(args.epochs):
        # TRAINING
        losses.reset()
        for batch in range(max_iter):
            mixture_audio.d, target_audio.d = train_iter.next()
            solver.zero_grad()
            loss.forward(clear_no_need_grad=True)
            if comm.n_procs > 1:
                all_reduce_callback = comm.get_all_reduce_callback()
                loss.backward(clear_buffer=True,
                              communicator_callbacks=all_reduce_callback)
            else:
                loss.backward(clear_buffer=True)
            solver.weight_decay(weight_decay)
            solver.update()
            losses.update(loss.d.copy(), args.batch_size)
        training_loss = losses.get_avg()

        # clear cache memory
        ext.clear_memory_cache()

        lr = lr_scheduler.get_learning_rate(epoch)
        solver.set_learning_rate(lr)

        if comm.rank == 0:
            monitor_traing_loss.add(epoch, training_loss)
            monitor_lr.add(epoch, lr)
            monitor_time.add(epoch)

            # save intermediate weights
            nn.save_parameters(f"{os.path.join(args.output, args.target)}.h5")

    if comm.rank == 0:
        # save final weights
        nn.save_parameters(f"{os.path.join(args.output, args.target)}_final.h5")


if __name__ == '__main__':
    train()
