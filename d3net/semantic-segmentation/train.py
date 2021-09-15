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
D3Net Semantic Segmentation Training Code
'''

import os
import argparse
import yaml
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
import nnabla.monitor as M
from nnabla.ext_utils import get_extension_context
from comm import CommunicatorWrapper
from segmentation_data import data_iterator_cityscapes
from model import d3net_segmentation
from lr_scheduler import PolynomialScheduler


def get_args(description='D3Net Semantic Segmentation Training'):
    '''
    Get command line arguments.
    Arguments set the default values of command line arguments.
    '''
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--data-dir', '-d', type=str, required=True,
                        help='root path of the dataset')
    parser.add_argument('--output-dir', '-o', default='./d3net',
                        help='Output directory for saving the model parameters')
    parser.add_argument('--log-interval', type=int,
                        default=50, help="Interval for saving training logs")
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension modules('cpu', 'cudnn')")
    parser.add_argument('--pretrained', '-p', type=str, required=True,
                        help="Path to pre-trained D3Net backbone weights")
    parser.add_argument('--config-file', '-cfg', type=str, default='./configs/D3Net_L.yaml',
                        help="Configuration file('D3Net_L.yaml', 'D3Net_S.yaml')")
    parser.add_argument('--recompute', '-r', action='store_true', default=False,
                        help='If True, reduces the memory usage for training at the cost of additional training time ')
    return parser.parse_args()


def train():
    '''
    Run D3Net Semantic Segmentation Training
    '''
    args = get_args()
    # Load D3Net Hyper parameters (D3Net-L or D3Net-S)
    with open(args.config_file) as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)

    # Get context.
    ctx = get_extension_context(args.context, device_id=0)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)

    # Change max_iter, learning_rate and weight_decay according no. of gpu devices for multi-gpu training.
    default_batch_size = 8
    train_scale_factor = comm.n_procs * \
        (hparams['batch_size'] / default_batch_size)
    hparams['max_iter'] = int(hparams['max_iter'] // train_scale_factor)
    hparams['lr'] = hparams['lr'] * train_scale_factor
    hparams['min_lr'] = hparams['min_lr'] * train_scale_factor
    hparams['weight_decay'] = hparams['weight_decay'] * comm.n_procs

    # ---------------------
    # Create data iterators
    # ---------------------
    rng = np.random.RandomState()
    data = data_iterator_cityscapes(
        hparams['batch_size'], args.data_dir, rng=rng, train=True)

    if comm.n_procs > 1:
        data = data.slice(rng=rng, num_of_slices=comm.n_procs,
                          slice_pos=comm.rank)

    if comm.rank == 0:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

    # Create monitors
    monitor = M.Monitor(args.output_dir)
    monitor_training_loss = M.MonitorSeries(
        'Training loss', monitor, interval=args.log_interval)
    monitor_lr = M.MonitorSeries(
        'Learning rate', monitor, interval=args.log_interval)
    monitor_time = M.MonitorTimeElapsed(
        "Training time per iteration", monitor, interval=args.log_interval)

    # ---------------------
    # Create Training Graph
    # ---------------------
    # Create input variables
    image = nn.Variable(
        (hparams['batch_size'], 3, hparams['image_height'], hparams['image_width']))
    seg_gt = nn.Variable(
        (hparams['batch_size'], 1, hparams['image_height'], hparams['image_width']))
    mask = nn.Variable(
        (hparams['batch_size'], 1, hparams['image_height'], hparams['image_width']))

    # D3Net prediction/output
    seg_pred = d3net_segmentation(image, hparams, recompute=args.recompute)

    # Configure loss
    loss = F.mean(F.softmax_cross_entropy(seg_pred, seg_gt, axis=1) * mask)
    loss.persistent = True

    # Create Solver
    solver = S.Momentum(hparams['lr'], hparams['momentum'])
    solver.set_parameters(nn.get_parameters())

    # Initialize LR Scheduler
    lr_scheduler = PolynomialScheduler(hparams)

    if args.pretrained is not None:
        # Initialize the D3Net backbone weights
        with nn.parameter_scope('backbone'):
            nn.load_parameters(args.pretrained)

    # -------------
    # Training loop
    # -------------
    for i in range(hparams['max_iter']):
        image.d, seg_gt.d, mask.d = data.next()
        solver.zero_grad()
        lr = lr_scheduler.get_learning_rate(i)
        solver.set_learning_rate(lr)
        loss.forward(clear_no_need_grad=True)

        if comm.n_procs > 1:
            all_reduce_callback = comm.get_all_reduce_callback()
            loss.backward(clear_buffer=True,
                          communicator_callbacks=all_reduce_callback)
        else:
            loss.backward(clear_buffer=True)
        solver.weight_decay(hparams['weight_decay'])
        solver.update()

        if comm.rank == 0:
            # Log monitors
            monitor_training_loss.add(i, loss.d.copy())
            monitor_lr.add(i, lr)
            monitor_time.add(i)

            if (i % hparams['save_interval']) == 0:
                # Save intermediate model parameters
                nn.save_parameters(os.path.join(
                    args.output_dir, "model_param_%08d.h5" % i))
                solver.save_states(os.path.join(
                    args.output_dir, "solver_states.h5"))

    if comm.rank == 0:
        # save final model parameters
        nn.save_parameters(os.path.join(args.output_dir, "final.h5"))


if __name__ == '__main__':
    train()
