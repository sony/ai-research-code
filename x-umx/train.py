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

import os
import nnabla as nn
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context, import_extension_module
from nnabla.utils.data_iterator import data_iterator
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from numpy.random import RandomState, seed
from tqdm import trange
from loss import mse_loss, sdr_loss
from lr_scheduler import ReduceLROnPlateau
from comm import CommunicatorWrapper
from args import get_train_args
from model import OpenUnmix_CrossNet, STFT, Spectrogram
from data import load_datasources
import utils
seed(42)

os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = str(1)

def train():
    # Check NNabla version
    from utils import get_nnabla_version_integer
    if get_nnabla_version_integer() < 11500:
        raise ValueError(
            'This does not work with nnabla version less than 1.15.0 due to [a bug](https://github.com/sony/nnabla/pull/760). Please update the nnabla version.')

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
        print("Mixing coef. is {}, i.e., MDL = {}*TD-Loss + FD-Loss".format(args.mcoef, args.mcoef))
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    # Initialize DataIterator for MUSDB.
    train_source, valid_source, args = load_datasources(parser, args)

    train_iter = data_iterator(
        train_source,
        args.batch_size,
        RandomState(args.seed),
        with_memory_cache=False,
        with_file_cache=False
    )

    valid_iter = data_iterator(
        valid_source,
        1,
        RandomState(args.seed),
        with_memory_cache=False,
        with_file_cache=False
    )

    if comm.n_procs > 1:
        train_iter = train_iter.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

        valid_iter = valid_iter.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    # Calculate maxiter per GPU device.
    max_iter = int((train_source._size // args.batch_size) // comm.n_procs)
    weight_decay = args.weight_decay * comm.n_procs

    print("max_iter", max_iter)

    scaler_mean, scaler_std = utils.get_statistics(args, train_source)

    max_bin = utils.bandwidth_to_max_bin(
        train_source.sample_rate, args.nfft, args.bandwidth
    )

    unmix = OpenUnmix_CrossNet(
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_channels=args.nb_channels,
        hidden_size=args.hidden_size,
        n_fft=args.nfft,
        n_hop=args.nhop,
        max_bin=max_bin
    )

    # Create input variables.
    mixture_audio = nn.Variable(
        [args.batch_size] + list(train_source._get_data(0)[0].shape))
    target_audio = nn.Variable(
        [args.batch_size] + list(train_source._get_data(0)[1].shape))

    vmixture_audio = nn.Variable([1] + [2, valid_source.sample_rate * args.valid_dur])
    vtarget_audio = nn.Variable([1] + [8, valid_source.sample_rate * args.valid_dur])

    # create training graph
    mix_spec, M_hat, pred = unmix(mixture_audio)
    Y = Spectrogram(
        *STFT(target_audio, n_fft=unmix.n_fft, n_hop=unmix.n_hop),
        mono=(unmix.nb_channels == 1)
    )
    loss_f = mse_loss(mix_spec, M_hat, Y)
    loss_t = sdr_loss(mixture_audio, pred, target_audio)
    loss = args.mcoef * loss_t + loss_f
    loss.persistent = True

    # Create Solver and set parameters.
    solver = S.Adam(args.lr)
    solver.set_parameters(nn.get_parameters())

    # create validation graph
    vmix_spec, vM_hat, vpred = unmix(vmixture_audio, test=True)
    vY = Spectrogram(
        *STFT(vtarget_audio, n_fft=unmix.n_fft, n_hop=unmix.n_hop),
        mono=(unmix.nb_channels == 1)
    )
    vloss_f = mse_loss(vmix_spec, vM_hat, vY)
    vloss_t = sdr_loss(vmixture_audio, vpred, vtarget_audio)
    vloss = args.mcoef * vloss_t + vloss_f
    vloss.persistent = True

    # Initialize Early Stopping
    es = utils.EarlyStopping(patience=args.patience)

    # Initialize LR Scheduler (ReduceLROnPlateau)
    lr_scheduler = ReduceLROnPlateau(
        lr=args.lr, factor=args.lr_decay_gamma, patience=args.lr_decay_patience)
    best_epoch = 0

    # Training loop.
    for epoch in trange(args.epochs):
        # TRAINING
        losses = utils.AverageMeter()
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
        training_loss = losses.avg

        # clear cache memory
        ext.clear_memory_cache()

        # VALIDATION
        vlosses = utils.AverageMeter()
        for batch in range(int(valid_source._size // comm.n_procs)):
            x, y = valid_iter.next()
            dur = int(valid_source.sample_rate * args.valid_dur)
            sp, cnt = 0, 0
            loss_tmp = nn.NdArray()
            loss_tmp.zero()
            while 1:
                vmixture_audio.d = x[Ellipsis, sp:sp+dur]
                vtarget_audio.d = y[Ellipsis, sp:sp+dur]
                vloss.forward(clear_no_need_grad=True)
                cnt += 1
                sp += dur
                loss_tmp += vloss.data
                if x[Ellipsis, sp:sp+dur].shape[-1] < dur or x.shape[-1] == cnt*dur:
                    break
            loss_tmp = loss_tmp / cnt
            if comm.n_procs > 1:
                comm.all_reduce(loss_tmp, division=True, inplace=True)
            vlosses.update(loss_tmp.data.copy(), 1)
        validation_loss = vlosses.avg

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

            # save current model and states
            nn.save_parameters(os.path.join(
                args.output, 'checkpoint_xumx.h5'))
            solver.save_states(os.path.join(
                args.output, 'xumx_states.h5'))

            if validation_loss == es.best:
                # save best model
                nn.save_parameters(os.path.join(
                    args.output, 'best_xumx.h5'))
                best_epoch = epoch

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == '__main__':
    train()
